"""Metadata signal: EXIF/XMP parsing, editor and AI-generator markers (matched via
guarded regex patterns so ordinary words like "influx" or "Dallas" cannot trigger
a false positive), thumbnail consistency, and JPEG quantization-table quality
estimation.

Only :mod:`PIL` and the MIT-licensed :mod:`piexif` (used solely to decode the
embedded EXIF thumbnail) are required.
"""

from __future__ import annotations

import io
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import piexif
from PIL import Image

from imgforensics.core.base import BaseDetector
from imgforensics.core.image import ForensicImage
from imgforensics.core.registry import register
from imgforensics.core.types import DetectionResult, Label

# Standard ("Annex K") JPEG luminance quantization table at quality 50, used as
# the libjpeg scaling-formula reference for jpeg_quality_estimate. Order does
# not matter for that estimate (only the mean of the 64 coefficients is
# used), but it is also reused, in this natural row-major order, by the
# exact-match classification below -- where order matters, see
# _classify_quant_tables.
_ANNEX_K_LUMA_TABLE = (
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
)  # fmt: skip

# Standard ("Annex K") JPEG chrominance quantization table at quality 50
# (libjpeg jcparam.c std_chrominance_quant_tbl), natural row-major order --
# see _ANNEX_K_LUMA_TABLE.
_ANNEX_K_CHROMA_TABLE = (
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
)  # fmt: skip

_EDITOR_NAMES = (
    "Photoshop",
    "Lightroom",
    "GIMP",
    "Affinity",
    "Snapseed",
    "Pixelmator",
    "Capture One",
    "Luminar",
    "Canva",
    "Picsart",
    "Facetune",
)

_AI_MARKER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Stable Diffusion", re.compile(r"stable[\s_-]*diffusion", re.IGNORECASE)),
    ("SDXL", re.compile(r"(?<![a-z0-9])sdxl(?![a-z0-9])", re.IGNORECASE)),
    ("Midjourney", re.compile(r"midjourney", re.IGNORECASE)),
    ("DALL-E", re.compile(r"dall[\s·•_-]?e(?![a-z])", re.IGNORECASE)),
    ("Adobe Firefly", re.compile(r"(?<![a-z0-9])firefly(?![a-z0-9])", re.IGNORECASE)),
    (
        "Google Imagen",
        re.compile(r"google\s*imagen|(?<![a-z0-9])imagen[\s_-]*\d", re.IGNORECASE),
    ),
    (
        "FLUX",
        re.compile(
            r"(?<![a-z0-9])flux(?:[\s_.-]*(?:1|dev|schnell|pro|kontext))(?![a-z0-9])"
            r"|black[\s-]*forest[\s-]*labs",
            re.IGNORECASE,
        ),
    ),
    ("ComfyUI", re.compile(r"comfyui", re.IGNORECASE)),
    ("NovelAI", re.compile(r"novelai", re.IGNORECASE)),
    ("Leonardo.Ai", re.compile(r"leonardo[\s._-]*ai(?![a-z])", re.IGNORECASE)),
    ("Ideogram", re.compile(r"(?<![a-z0-9])ideogram(?![a-z0-9])", re.IGNORECASE)),
    ("Automatic1111", re.compile(r"automatic1111|sd[\s_-]*webui", re.IGNORECASE)),
    (
        "IPTC digitalSourceType algorithmicMedia",
        re.compile(r"algorithmicmedia|compositesynthetic", re.IGNORECASE),
    ),
)

# EXIF/TIFF tag ids used here (see the EXIF 2.3 spec).
_TAG_MAKE = 0x010F
_TAG_MODEL = 0x0110
_TAG_SOFTWARE = 0x0131
_TAG_EXIF_IFD = 0x8769
_TAG_DATETIME_ORIGINAL = 0x9003
_TAG_LENS_MODEL = 0xA434


def _clean_str(value: Any) -> str | None:
    """Decode/strip an EXIF text value; return ``None`` when empty."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        value = value.strip("\x00").strip()
    return value or None


def _extract_exif_fields(img: Image.Image) -> dict[str, Any]:
    """Pull the handful of EXIF fields the metadata signal cares about."""
    exif = img.getexif()
    has_exif = bool(exif)
    software = _clean_str(exif.get(_TAG_SOFTWARE)) if exif else None
    fields: dict[str, Any] = {
        "has_exif": has_exif,
        "camera_make": _clean_str(exif.get(_TAG_MAKE)) if exif else None,
        "camera_model": _clean_str(exif.get(_TAG_MODEL)) if exif else None,
        "software": software,
        "datetime_original": None,
        "lens_model": None,
    }
    if exif:
        try:
            exif_sub = exif.get_ifd(_TAG_EXIF_IFD)
        except Exception:
            exif_sub = {}
        fields["datetime_original"] = _clean_str(exif_sub.get(_TAG_DATETIME_ORIGINAL))
        fields["lens_model"] = _clean_str(exif_sub.get(_TAG_LENS_MODEL))
    return fields


def _text_chunks(img: Image.Image) -> dict[str, str]:
    """Collect PNG text/info string chunks (``img.text`` and ``img.info``)."""
    chunks: dict[str, str] = {}
    text_attr = getattr(img, "text", None)
    if isinstance(text_attr, dict):
        for key, value in text_attr.items():
            if isinstance(value, bytes):
                value = value.decode("utf-8", "replace")
            chunks[key] = str(value)
    for key, value in img.info.items():
        if not isinstance(key, str) or key in chunks:
            continue
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8", "replace")
            except Exception:
                continue
        if isinstance(value, str):
            chunks[key] = value
    return chunks


def _xmp_text(img: Image.Image) -> str | None:
    xmp = img.info.get("xmp")
    if not xmp:
        return None
    return xmp.decode("utf-8", "replace") if isinstance(xmp, bytes) else str(xmp)


def _editor_markers(img: Image.Image, software: str | None) -> list[str]:
    """Detect metadata traces left by known photo editors."""
    markers: list[str] = []
    if software:
        for name in _EDITOR_NAMES:
            if name.lower() in software.lower():
                markers.append(f"Software tag mentions {name!r} ({software})")
                break
    if img.info.get("photoshop"):
        markers.append("Photoshop APP13 segment present")

    xmp_text = _xmp_text(img)
    if xmp_text:
        xmp_lower = xmp_text.lower()
        if "xmpmm:history" in xmp_lower:
            markers.append("XMP xmpMM:History present")
        if "stevt:action" in xmp_lower and "edited" in xmp_lower:
            markers.append('XMP stEvt:action="edited" present')
        if "photoshop:" in xmp_lower:
            markers.append("XMP photoshop: namespace present")
        match = re.search(r'creatortool[^>"]*[>"]([^<"]+)[<"]', xmp_text, re.IGNORECASE)
        if match:
            tool = match.group(1)
            for name in _EDITOR_NAMES:
                if name.lower() in tool.lower():
                    markers.append(f"XMP CreatorTool names {name!r} ({tool})")
                    break
    return markers


def _find_ai_terms(text: str) -> list[str]:
    """Return the labels of every AI-generator marker pattern found in ``text``."""
    return [label for label, pattern in _AI_MARKER_PATTERNS if pattern.search(text)]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _ai_markers(img: Image.Image, software: str | None) -> list[str]:
    """Detect metadata traces naming a known generative-AI tool."""
    markers: list[str] = []
    chunks = _text_chunks(img)

    if "parameters" in chunks:
        markers.append(
            "PNG text chunk 'parameters' present (Automatic1111 / Stable Diffusion WebUI)"
        )
    if "prompt" in chunks:
        markers.append("PNG text chunk 'prompt' present (ComfyUI)")
    if "workflow" in chunks:
        markers.append("PNG text chunk 'workflow' present (ComfyUI)")

    candidates: dict[str, str] = {}
    if software:
        candidates["Software"] = software
    for key in ("Software", "Comment", "Description", "comment", "description"):
        if key in chunks:
            candidates.setdefault(key, chunks[key])

    for label, value in candidates.items():
        for term in _find_ai_terms(value):
            markers.append(f"{label} mentions {term!r}")

    xmp_text = _xmp_text(img)
    if xmp_text:
        for term in _find_ai_terms(xmp_text):
            markers.append(f"XMP packet mentions {term!r}")

    return _dedupe(markers)


def _thumbnail_check(original: Image.Image, raw: bytes) -> dict[str, Any]:
    """Compare the embedded EXIF thumbnail against the full-resolution image."""
    result: dict[str, Any] = {
        "thumbnail_mismatch": False,
        "thumbnail_mad": None,
        "thumbnail_ncc": None,
    }
    try:
        exif_dict = piexif.load(raw)
    except Exception:
        # Not a JPEG/TIFF, or no parseable EXIF: nothing to compare, no error.
        return result

    thumb_bytes = exif_dict.get("thumbnail")
    if not thumb_bytes:
        return result

    try:
        with Image.open(io.BytesIO(thumb_bytes)) as thumb_img:
            thumb_img.load()
            thumb_gray = np.asarray(thumb_img.convert("L"), dtype=np.float64)
        thumb_size = (int(thumb_gray.shape[1]), int(thumb_gray.shape[0]))
        main_resized = original.convert("RGB").resize(thumb_size, Image.Resampling.BILINEAR)
        main_gray = np.asarray(main_resized.convert("L"), dtype=np.float64)
    except Exception as exc:
        result["thumbnail_error"] = f"failed to decode embedded thumbnail: {exc}"
        return result

    mad = float(np.mean(np.abs(main_gray - thumb_gray)) / 255.0)
    main_centered = main_gray - main_gray.mean()
    thumb_centered = thumb_gray - thumb_gray.mean()
    denom = float(np.sqrt(np.sum(main_centered**2)) * np.sqrt(np.sum(thumb_centered**2)))
    ncc = float(np.sum(main_centered * thumb_centered) / denom) if denom > 0 else 0.0

    result["thumbnail_mad"] = mad
    result["thumbnail_ncc"] = ncc
    result["thumbnail_mismatch"] = bool(ncc < 0.6 or mad > 0.15)
    return result


def _estimate_jpeg_quality(luma_table: Sequence[int]) -> int:
    """Estimate JPEG quality from a luma quantization table.

    Uses the standard libjpeg scaling formula: the quantization table at
    quality ``q`` is ``base * scale / 100`` where ``scale = 200 - 2q`` for
    ``q >= 50`` and ``scale = 5000 / q`` for ``q < 50``. We estimate ``scale``
    as the ratio of the mean of the actual table to the mean of the Annex K
    base table (order-independent, so it works regardless of zig-zag vs.
    natural ordering) and invert it, clamped to [1, 100].
    """
    mean_table = float(np.mean(np.asarray(luma_table, dtype=np.float64)))
    mean_base = float(np.mean(np.asarray(_ANNEX_K_LUMA_TABLE, dtype=np.float64)))
    if mean_base <= 0:
        return 50
    scale_factor = (mean_table / mean_base) * 100.0
    quality = (200.0 - scale_factor) / 2.0 if scale_factor <= 100.0 else 5000.0 / scale_factor
    return int(round(min(100.0, max(1.0, quality))))


def _ijg_scale_factor(quality: int) -> int:
    """The libjpeg ``jpeg_quality_scaling`` scale factor for ``quality`` (1-100)."""
    quality = max(1, min(100, quality))
    return 5000 // quality if quality < 50 else 200 - quality * 2


def _ijg_scaled_table(base: Sequence[int], quality: int) -> tuple[int, ...]:
    """The libjpeg ``jpeg_add_quant_table`` scaling of ``base`` at ``quality``.

    ``temp = (base[i] * scale + 50) / 100``, clamped to ``[1, 255]`` (the
    ``force_baseline`` clamp libjpeg applies for 8-bit JPEGs).
    """
    scale = _ijg_scale_factor(quality)
    return tuple(max(1, min(255, (coeff * scale + 50) // 100)) for coeff in base)


# Every (luma, chroma) IJG-standard table pair for quality 1..100, keyed for
# exact-match lookup. Built ascending so that if two qualities round to the
# same table pair (only possible near quality 100, where clamping saturates
# many coefficients at 1), the higher quality wins.
_STANDARD_QUALITY_TABLES: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {
    (_ijg_scaled_table(_ANNEX_K_LUMA_TABLE, q), _ijg_scaled_table(_ANNEX_K_CHROMA_TABLE, q)): q
    for q in range(1, 101)
}


def _classify_quant_tables(quantization: dict[int, list[int]]) -> dict[str, Any]:
    """Classify a JPEG's luma+chroma quantization tables as IJG-standard or not.

    ``jpeg_quant_standard`` is ``True`` when both the luma (table 0) and
    chroma (table 1) tables exactly equal the libjpeg-computed pair for some
    quality 1-100 (see ``_STANDARD_QUALITY_TABLES``), in which case
    ``jpeg_quant_quality_exact`` is that quality. It is ``False`` when both
    tables are present but do not match any standard pair -- non-standard
    tables typically come from cameras or Adobe products, which use their
    own quantization tables rather than the plain libjpeg scaling formula.
    It is ``None`` (unknown) when there is no separate chroma table to pair
    the luma table with (e.g. a grayscale JPEG).
    """
    luma = quantization.get(0)
    chroma = quantization.get(1)
    if luma is None or chroma is None:
        return {"jpeg_quant_standard": None, "jpeg_quant_quality_exact": None}
    quality = _STANDARD_QUALITY_TABLES.get((tuple(luma), tuple(chroma)))
    if quality is not None:
        return {"jpeg_quant_standard": True, "jpeg_quant_quality_exact": quality}
    return {"jpeg_quant_standard": False, "jpeg_quant_quality_exact": None}


def _jpeg_quality_info(img: Image.Image) -> dict[str, Any]:
    quantization = getattr(img, "quantization", None)
    if img.format != "JPEG" or not quantization:
        return {
            "jpeg_quality_estimate": None,
            "jpeg_quant_tables_count": None,
            "jpeg_quant_standard": None,
            "jpeg_quant_quality_exact": None,
        }
    luma_table = quantization.get(0) or next(iter(quantization.values()))
    return {
        "jpeg_quality_estimate": _estimate_jpeg_quality(luma_table),
        "jpeg_quant_tables_count": len(quantization),
        **_classify_quant_tables(quantization),
    }


@register("metadata")
class MetadataSignal(BaseDetector):
    """Classical signal built purely from image metadata.

    Reads EXIF, XMP, PNG text chunks, the Photoshop APP13 segment, the
    embedded EXIF thumbnail, and (for JPEG) the quantization tables from
    :meth:`ForensicImage.open_original`. Requires the original encoded bytes;
    when ``image.raw`` is ``None`` (e.g. built via ``ForensicImage.from_pil``)
    the signal abstains with score 0.5 and
    ``details["reason"] = "no encoded file available"``.

    Score rules, evaluated in order (first match wins):

    1. ``ai_markers`` non-empty -> score 0.95, label "fake". Metadata names a
       known generative-AI tool (Stable Diffusion, Midjourney, ComfyUI, ...),
       matched via guarded regex patterns (see ``_AI_MARKER_PATTERNS``) so that
       ordinary words are never mistaken for a marker.
    2. ``thumbnail_mismatch`` is ``True`` -> score 0.85, label "fake". The
       embedded EXIF thumbnail does not match the full-resolution image, a
       classic sign the image content was swapped after the thumbnail was
       generated.
    3. ``editor_markers`` non-empty -> score 0.70, label "fake". Metadata
       names a known photo editor (Software tag, Photoshop APP13, XMP
       history/CreatorTool).
    4. ``has_exif`` is ``True`` and ``camera_make`` is set (and, by the
       ordering above, no editor markers were found) -> score 0.30, label
       "real". Camera-shot metadata with no edit trace.
    5. Otherwise -> score 0.50, label "uncertain". No usable metadata
       evidence; most sharing platforms strip EXIF, so this is inconclusive
       rather than informative of anything.

    Every computed field is placed in ``details``; this signal never raises,
    catching unexpected errors into ``details["error"]`` with score 0.5.
    """

    name = "metadata"

    def predict(self, image: ForensicImage) -> DetectionResult:
        if image.raw is None:
            return DetectionResult(
                detector=self.name,
                score=0.5,
                label="uncertain",
                details={"reason": "no encoded file available"},
            )
        try:
            return self._predict_from_raw(image.raw)
        except Exception as exc:  # never raise: record and abstain
            return DetectionResult(
                detector=self.name,
                score=0.5,
                label="uncertain",
                details={"error": f"{type(exc).__name__}: {exc}"},
            )

    def _predict_from_raw(self, raw: bytes) -> DetectionResult:
        with Image.open(io.BytesIO(raw)) as img:
            fmt = img.format
            fields = _extract_exif_fields(img)
            editor_markers = _editor_markers(img, fields["software"])
            ai_markers = _ai_markers(img, fields["software"])
            thumb = _thumbnail_check(img, raw)
            jpeg_info = _jpeg_quality_info(img)

        details: dict[str, Any] = {
            "format": fmt,
            **fields,
            "editor_markers": editor_markers,
            "ai_markers": ai_markers,
            **thumb,
            **jpeg_info,
        }

        score: float
        label: Label
        if ai_markers:
            score, label = 0.95, "fake"
        elif thumb.get("thumbnail_mismatch"):
            score, label = 0.85, "fake"
        elif editor_markers:
            score, label = 0.70, "fake"
        elif fields["has_exif"] and fields["camera_make"]:
            score, label = 0.30, "real"
        else:
            score, label = 0.50, "uncertain"
            details["note"] = "no metadata evidence; metadata is stripped by most platforms"

        return DetectionResult(detector=self.name, score=score, label=label, details=details)
