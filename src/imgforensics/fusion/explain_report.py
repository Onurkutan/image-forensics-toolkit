"""Explanation-report builder: one folder per ``analyze`` run.

``docs/ROADMAP.md``, Phase 5: turns a single image's detector results (and,
optionally, a fitted :class:`~imgforensics.fusion.stacking.Fuser`) into a
self-contained report folder -- ``report.json`` (machine-readable), a
``<detector>_heatmap.png`` / ``<detector>_overlay.png`` pair per detector
that produced a heatmap, and ``report.md`` (a human-readable summary with
plain-language cards, reusing :data:`~imgforensics.fusion.report.DETECTOR_NOTES`).

Pure ``numpy`` + ``Pillow`` -- no plotting library -- so :func:`apply_colormap`
implements a small built-in 5-stop colour ramp instead of depending on
matplotlib.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from imgforensics import __version__
from imgforensics.core.image import ForensicImage
from imgforensics.core.types import DetectionResult
from imgforensics.fusion.report import DETECTOR_NOTES, GENERIC_NOTE, explain
from imgforensics.fusion.stacking import Band, Fuser
from imgforensics.utils.image_io import image_hash
from imgforensics.utils.jsonsafe import to_jsonable

#: Longest-side cap the RGB overlay is downscaled to before blending. Only
#: the overlay (meant for eyeballing) is capped this way; the grayscale
#: heatmap PNG next to it is always written at the heatmap's own (full)
#: resolution, matching ``imgforensics analyze --save-heatmaps``.
OVERLAY_MAX_SIDE = 1024

#: Heatmap value above which the overlay draws a thin one-pixel outline
#: around the region's boundary.
_OUTLINE_THRESHOLD = 0.5

#: Opacity of the colour wash at heatmap value 1.0, scaling linearly down to
#: 0 at heatmap value 0 -- so a heatmap of all zeros leaves every pixel
#: exactly unchanged and untouched regions stay visually clear.
_MAX_OVERLAY_ALPHA = 0.6

_OUTLINE_COLOR = (255.0, 255.0, 255.0)

_SCORE_BAR_WIDTH = 20
_TOP_CONTRIBUTIONS_SHOWN = 5
_DETAIL_VALUE_LIMIT = 100

#: Detector names shown ahead of the classical signals in ``report.md``, in
#: this order (see the module docstring for the rest of the card ordering:
#: fused verdict, then learned, then localizers, then signals by descending
#: score). Kept as small local sets, the same way
#: :mod:`imgforensics.fusion.report`'s ``DETECTOR_NOTES`` is keyed by plain
#: registry names, rather than importing ``imgforensics.detectors``/
#: ``imgforensics.localization`` just for a name list.
_LEARNED_DETECTOR_NAMES = frozenset({"dinov2_head"})
_LOCALIZER_DETECTOR_NAMES = frozenset({"iml_vit", "catnet_v2"})

#: Curated "most informative" detail keys per detector, shown in ``report.md``'s
#: two-column table -- the full ``details`` dict already lives in
#: ``report.json`` for anything not shown here.
_DETAIL_KEYS_BY_DETECTOR: dict[str, tuple[str, ...]] = {
    "metadata": (
        "ai_markers",
        "editor_markers",
        "camera_make",
        "camera_model",
        "software",
        "jpeg_quality_estimate",
        "jpeg_quant_standard",
        "thumbnail_mismatch",
    ),
    "c2pa": (
        "manifest_present",
        "digital_source_type",
        "actions",
        "validation_status",
        "is_valid",
        "claim_generator",
    ),
    "ela": ("mean_ela", "p99_ela", "high_region_fraction", "block_inhomogeneity"),
    "copy_move": (
        "accepted",
        "matched_fraction",
        "dominant_shift",
        "dominant_votes",
        "candidate_pairs",
    ),
    "jpeg_ghost": ("ghost_fraction", "dominant_quality", "qualities"),
    "double_jpeg": (
        "grid_strength",
        "secondary_grid_detected",
        "double_quantization_suspected",
        "jpeg_history_detected",
    ),
    "sd_watermark": ("best_payload", "best_agreement", "matched"),
    "dinov2_head": ("n_crops", "per_crop", "backbone"),
}

#: Fallback keys for a localizer not named above (``iml_vit``/``catnet_v2``
#: both report these three shapes): "max/mean/area", per the module contract.
_LOCALIZER_CORE_KEYS = ("max_prob", "mean_prob")


# --------------------------------------------------------------------------
# Colormap and overlay blending
# --------------------------------------------------------------------------

#: A small built-in 5-stop blue -> cyan -> green -> yellow -> red ramp (a
#: cheap stand-in for viridis/jet), (position, r, g, b) rows.
_COLORMAP_STOPS = np.array(
    [
        [0.00, 0.0, 0.0, 255.0],
        [0.25, 0.0, 255.0, 255.0],
        [0.50, 0.0, 255.0, 0.0],
        [0.75, 255.0, 255.0, 0.0],
        [1.00, 255.0, 0.0, 0.0],
    ]
)


def apply_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Map a float heatmap in [0, 1] to an 8-bit HxWx3 RGB image.

    Uses the built-in 5-stop ramp (see :data:`_COLORMAP_STOPS`): blue (cold,
    0) through cyan, green, yellow, to red (hot, 1), linearly interpolated
    per channel with :func:`numpy.interp`. Values outside [0, 1] are clipped
    first.
    """
    clipped = np.clip(heatmap, 0.0, 1.0).astype(np.float64)
    positions = _COLORMAP_STOPS[:, 0]
    channels = [np.interp(clipped, positions, _COLORMAP_STOPS[:, channel]) for channel in (1, 2, 3)]
    return np.stack(channels, axis=-1).astype(np.uint8)


def _region_outline(mask: np.ndarray) -> np.ndarray:
    """One-pixel-wide interior boundary of ``mask`` (a 2-D boolean array).

    ``True`` where ``mask`` is ``True`` but at least one of its four
    (up/down/left/right) neighbours is ``False`` -- a cheap 4-connectivity
    erosion via array shifts, no ``scipy`` dependency. A pixel at the image
    border is only compared against the neighbours it actually has.
    """
    eroded = mask.copy()
    eroded[1:, :] &= mask[:-1, :]
    eroded[:-1, :] &= mask[1:, :]
    eroded[:, 1:] &= mask[:, :-1]
    eroded[:, :-1] &= mask[:, 1:]
    return mask & ~eroded


def blend_overlay(
    rgb: np.ndarray, heatmap: np.ndarray, *, max_alpha: float = _MAX_OVERLAY_ALPHA
) -> np.ndarray:
    """Blend a colour-mapped ``heatmap`` over ``rgb``, alpha proportional to heatmap value.

    Args:
        rgb: HxWx3 uint8 image.
        heatmap: HxW float array in [0, 1], the same shape as ``rgb``.
        max_alpha: Opacity of the colour wash at heatmap value 1.0; scales
            linearly to 0 at heatmap value 0, so a heatmap of all zeros
            leaves ``rgb`` completely unchanged and untouched regions stay
            visually clear.

    Returns:
        HxWx3 uint8 image: ``rgb`` blended with the colour-mapped heatmap,
        plus a thin one-pixel outline around the boundary of the region
        where ``heatmap > 0.5``.

    Raises:
        ValueError: if ``rgb`` and ``heatmap`` have different HxW shapes.
    """
    if rgb.shape[:2] != heatmap.shape:
        raise ValueError(f"shape mismatch: rgb {rgb.shape[:2]} vs heatmap {heatmap.shape}")

    base = rgb.astype(np.float64)
    colors = apply_colormap(heatmap).astype(np.float64)
    alpha = (np.clip(heatmap, 0.0, 1.0) * max_alpha)[..., None]
    blended = base * (1.0 - alpha) + colors * alpha

    outline = _region_outline(heatmap > _OUTLINE_THRESHOLD)
    blended[outline] = _OUTLINE_COLOR

    return np.clip(blended, 0, 255).astype(np.uint8)


def _resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear-resize a float ``heatmap`` to ``size`` = ``(width, height)``, clipped to [0, 1]."""
    image = Image.fromarray(heatmap.astype(np.float32), mode="F")
    resized = image.resize(size, Image.Resampling.BILINEAR)
    return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)


def _downscale_for_overlay(image: Image.Image, max_side: int) -> tuple[Image.Image, bool]:
    """Downscale ``image`` so its longer side is at most ``max_side``, if needed.

    Returns ``(image, False)`` unchanged when already within the cap.
    """
    width, height = image.size
    long_side = max(width, height)
    if long_side <= max_side:
        return image, False
    scale = max_side / long_side
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.BILINEAR), True


def _build_overlay_image(rgb_image: Image.Image, heatmap: np.ndarray) -> tuple[Image.Image, bool]:
    """Build the RGB overlay for one detector's heatmap, downscaling large images first.

    Returns ``(overlay_image, was_downscaled)``.
    """
    downscaled_image, was_downscaled = _downscale_for_overlay(rgb_image, OVERLAY_MAX_SIDE)
    heatmap_for_overlay = (
        _resize_heatmap(heatmap, downscaled_image.size) if was_downscaled else heatmap
    )
    rgb_array = np.asarray(downscaled_image.convert("RGB"), dtype=np.uint8)
    blended = blend_overlay(rgb_array, heatmap_for_overlay)
    return Image.fromarray(blended, mode="RGB"), was_downscaled


def _save_heatmap_png(heatmap: np.ndarray, path: Path) -> None:
    """Write ``heatmap`` (float, [0, 1]) as an 8-bit grayscale PNG, full resolution."""
    array_8bit = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(array_8bit, mode="L").save(path)


# --------------------------------------------------------------------------
# report.json
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportPaths:
    """Paths written by one :func:`build_report` call."""

    out_dir: Path
    json_path: Path
    markdown_path: Path
    heatmap_paths: dict[str, Path]
    overlay_paths: dict[str, Path]


def _fusion_block(fuser: Fuser, results: Sequence[DetectionResult]) -> dict[str, Any]:
    scores = {result.detector: result.score for result in results}
    probability = fuser.predict(scores)
    label = fuser.predict_label(scores)
    contributions = explain(fuser, scores)
    return {
        "probability": probability,
        "label": label,
        "band": {"low": fuser.band.low, "high": fuser.band.high},
        "contributions": [
            {
                "detector": contribution.detector,
                "score": contribution.score,
                "weight": contribution.weight,
                "contribution": contribution.contribution,
                "present": contribution.present,
                "note": contribution.note,
            }
            for contribution in contributions
        ],
    }


def build_report(
    image: ForensicImage,
    results: list[DetectionResult],
    *,
    fuser: Fuser | None,
    out_dir: Path,
    source_name: str,
) -> ReportPaths:
    """Build a self-contained report folder for one ``analyze`` run.

    Writes, into ``out_dir`` (created if needed):

    - ``<detector>_heatmap.png`` and ``<detector>_overlay.png`` for every
      result that carries a heatmap (see :func:`blend_overlay`).
    - ``report.json``: image info, per-detector entries, the fusion block
      (only present when ``fuser`` is given), package version and a created
      timestamp.
    - ``report.md``: the same information as a human-readable Markdown
      summary with plain-language cards (see :func:`_render_markdown`).

    Args:
        image: The analyzed image.
        results: Every detector's :class:`DetectionResult` from this run.
        fuser: A fitted fuser to compute a fused-verdict card/block from, or
            ``None`` to omit fusion entirely (report still builds fine).
        out_dir: Directory to write the report into.
        source_name: Display name for the image (typically the input file's
            name), recorded in ``report.json``'s ``image.name`` and in
            ``report.md``'s title/info table.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = image_hash(image.rgb)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")

    heatmap_paths: dict[str, Path] = {}
    overlay_paths: dict[str, Path] = {}
    overlay_meta: dict[str, dict[str, Any]] = {}

    for result in results:
        if result.heatmap is None:
            continue

        heatmap_path = out_dir / f"{result.detector}_heatmap.png"
        _save_heatmap_png(result.heatmap, heatmap_path)
        heatmap_paths[result.detector] = heatmap_path

        overlay_image, downscaled = _build_overlay_image(image.rgb, result.heatmap)
        overlay_path = out_dir / f"{result.detector}_overlay.png"
        overlay_image.save(overlay_path)
        overlay_paths[result.detector] = overlay_path
        overlay_meta[result.detector] = {
            "width": overlay_image.width,
            "height": overlay_image.height,
            "downscaled": downscaled,
        }

    fusion_block = _fusion_block(fuser, results) if fuser is not None else None

    document: dict[str, Any] = {
        "image": {
            "name": source_name,
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "sha256": digest,
        },
        "detectors": [
            {
                "detector": result.detector,
                "score": result.score,
                "label": result.label,
                "elapsed_ms": result.elapsed_ms,
                "details": to_jsonable(result.details),
                "heatmap": (
                    heatmap_paths[result.detector].name
                    if result.detector in heatmap_paths
                    else None
                ),
                "overlay": (
                    {"file": overlay_paths[result.detector].name, **overlay_meta[result.detector]}
                    if result.detector in overlay_paths
                    else None
                ),
            }
            for result in results
        ],
        "overlay_max_side": OVERLAY_MAX_SIDE,
        "package_version": __version__,
        "created": created,
    }
    if fusion_block is not None:
        document["fusion"] = fusion_block

    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    markdown_path = out_dir / "report.md"
    markdown_path.write_text(
        _render_markdown(
            image=image,
            results=results,
            band=fuser.band if fuser is not None else None,
            fusion_block=fusion_block,
            overlay_paths=overlay_paths,
            source_name=source_name,
            digest=digest,
        ),
        encoding="utf-8",
    )

    return ReportPaths(
        out_dir=out_dir,
        json_path=json_path,
        markdown_path=markdown_path,
        heatmap_paths=heatmap_paths,
        overlay_paths=overlay_paths,
    )


# --------------------------------------------------------------------------
# report.md
# --------------------------------------------------------------------------


def _score_bar(score: float, width: int = _SCORE_BAR_WIDTH) -> str:
    """A text score bar, e.g. ``[#######.............]``."""
    filled = int(round(float(np.clip(score, 0.0, 1.0)) * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _escape_cell(text: str) -> str:
    """Escape a Markdown table cell's pipe characters."""
    return text.replace("|", "\\|")


def _format_value(value: Any) -> str:
    """Render one detail value as a short, single-line Markdown table cell."""
    if value is None:
        text = "-"
    elif isinstance(value, float):
        text = f"{value:.4g}"
    elif isinstance(value, list | tuple):
        text = ", ".join(_format_value(item) for item in value) if value else "-"
    elif isinstance(value, dict):
        text = json.dumps(to_jsonable(value), separators=(",", ":"))
    else:
        text = str(value)
    if len(text) > _DETAIL_VALUE_LIMIT:
        text = text[: _DETAIL_VALUE_LIMIT - 1] + "\N{HORIZONTAL ELLIPSIS}"
    return _escape_cell(text)


def _is_localizer_details(details: Mapping[str, Any]) -> bool:
    return set(_LOCALIZER_CORE_KEYS) <= details.keys()


def _detail_rows(detector: str, details: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Pick the "most informative" ``details`` entries to show for ``detector``.

    Uses the curated per-detector key list (:data:`_DETAIL_KEYS_BY_DETECTOR`)
    when one exists; otherwise, a detector reporting the localizer shape
    (``max_prob``/``mean_prob``, e.g. ``iml_vit``/``catnet_v2``, or a future
    one) shows those plus any ``area_fraction_above_*`` key; anything else
    falls back to every detail key except the free-text ``note``/``error``
    fields (already surfaced via :func:`_caution_line` and the plain-language
    note).
    """
    keys = _DETAIL_KEYS_BY_DETECTOR.get(detector)
    if keys is None:
        if _is_localizer_details(details):
            area_keys = tuple(sorted(k for k in details if k.startswith("area_fraction_above_")))
            keys = _LOCALIZER_CORE_KEYS + area_keys
        else:
            keys = tuple(key for key in details if key not in {"note", "error"})

    rows: list[tuple[str, str]] = []
    for key in keys:
        if key not in details or details[key] is None:
            continue
        rows.append((key, _format_value(details[key])))
    return rows


def _caution_line(result: DetectionResult) -> str | None:
    """A one-line caution for a detector that abstained (score exactly 0.5, a ``reason`` given)."""
    reason = result.details.get("reason")
    if result.score == 0.5 and reason:
        return f"*Caution:* this detector abstained -- {reason}"
    return None


def _band_sentence(probability: float, label: str, band: Band) -> str:
    """One plain-language sentence explaining why the fuser landed on ``label``."""
    if label == "fake":
        return (
            f"probability {probability:.2f} is above the abstain band's high threshold "
            f"({band.high:.2f}), so the fuser calls this **fake**."
        )
    if label == "real":
        return (
            f"probability {probability:.2f} is below the abstain band's low threshold "
            f"({band.low:.2f}), so the fuser calls this **real**."
        )
    return (
        f"probability {probability:.2f} falls inside the abstain band "
        f"[{band.low:.2f}, {band.high:.2f}], so the fuser reports **uncertain** rather than "
        "guessing."
    )


def _fusion_card(fusion_block: Mapping[str, Any], band: Band) -> list[str]:
    probability = float(fusion_block["probability"])
    label = str(fusion_block["label"])
    lines = [
        "## Fused verdict",
        "",
        f"**probability = {probability:.2f}** -- {_band_sentence(probability, label, band)}",
        "",
        "| detector | score | contribution | note |",
        "|---|---|---|---|",
    ]
    for contribution in fusion_block["contributions"][:_TOP_CONTRIBUTIONS_SHOWN]:
        score_text = f"{contribution['score']:.2f}"
        if not contribution["present"]:
            score_text += " (missing)"
        lines.append(
            f"| {contribution['detector']} | {score_text} | "
            f"{contribution['contribution']:+.3f} | {_escape_cell(str(contribution['note']))} |"
        )
    lines.append("")
    return lines


def _group_rank(detector: str) -> int:
    if detector in _LEARNED_DETECTOR_NAMES:
        return 0
    if detector in _LOCALIZER_DETECTOR_NAMES:
        return 1
    return 2


def _ordered_results(results: Sequence[DetectionResult]) -> list[DetectionResult]:
    """Card order: learned detectors, then localizers, then signals -- each by descending score."""
    return sorted(results, key=lambda result: (_group_rank(result.detector), -result.score))


def _detector_card(result: DetectionResult, overlay_paths: Mapping[str, Path]) -> list[str]:
    lines = [
        f"## {result.detector}",
        "",
        f"`{_score_bar(result.score)}` score = {result.score:.2f}  label = **{result.label}**",
        "",
    ]

    rows = _detail_rows(result.detector, result.details)
    if rows:
        lines += ["| Field | Value |", "|---|---|"]
        lines += [f"| {_escape_cell(key)} | {value} |" for key, value in rows]
        lines.append("")

    if result.detector in overlay_paths:
        lines.append(f"![{result.detector} overlay]({overlay_paths[result.detector].name})")
        lines.append("")

    note = DETECTOR_NOTES.get(result.detector, GENERIC_NOTE)
    lines.append(f"*What this means:* {note}")

    caution = _caution_line(result)
    if caution is not None:
        lines.append("")
        lines.append(caution)

    lines.append("")
    return lines


def _render_markdown(
    *,
    image: ForensicImage,
    results: Sequence[DetectionResult],
    band: Band | None,
    fusion_block: Mapping[str, Any] | None,
    overlay_paths: Mapping[str, Path],
    source_name: str,
    digest: str,
) -> str:
    lines = [
        f"# Forensic report: {source_name}",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| file | {_escape_cell(source_name)} |",
        f"| size | {image.width}x{image.height} |",
        f"| format | {image.format or 'unknown'} |",
        f"| sha256 | {digest[:12]} |",
        "",
    ]

    if fusion_block is not None:
        assert band is not None  # fusion_block is only built alongside a Fuser
        lines += _fusion_card(fusion_block, band)

    for result in _ordered_results(results):
        lines += _detector_card(result, overlay_paths)

    return "\n".join(lines) + "\n"
