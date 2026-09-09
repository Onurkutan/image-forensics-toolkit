"""Invisible-watermark signal: detects the ``dwtDct`` watermark that Stable
Diffusion reference pipelines embed via the CompVis/Stability
`invisible-watermark` package (MIT, github.com/ShieldMnt/invisible-watermark).

This signal does *not* depend on that package -- its default ``rivaGan``
backend pulls in ``torch``, ``onnxruntime`` and ``opencv-python`` (this
project uses ``opencv-python-headless``, so those may not even be installed
correctly side by side). Instead it vendors just the pure numpy/opencv/
PyWavelets ``dwtDct`` codec (the ``EmbedMaxDct`` class, unrelated to the
GAN-based backend) -- see :mod:`imgforensics.signals._vendor.dwtdct`.

Decoding always returns exactly as many bits as asked for, watermarked or
not (there is no "absent" marker), so a bare decode is meaningless on its
own. Instead, for each known Stable Diffusion payload below, this signal
decodes with that payload's exact bit length and measures the fraction of
decoded bits that agree with the known payload ("agreement"). Each bit is
independently close to a coin flip on a non-watermarked image, so agreement
there concentrates near 0.5; a real embedding pushes it towards 1.0. A
payload counts as "matched" when its agreement clears a payload-specific
threshold (see ``_KNOWN_PAYLOADS``), chosen well above the ~0.5 chance floor:
0.95 for the 48-bit payload (chance of 24+/48 correct by luck alone is
negligible) and 0.90 for the 136-bit payload (same reasoning, looser only
because empirically the longer payload's per-bit noise floor sits a little
higher after any lossy step).

Blind spot measured directly (see tests/test_signals_watermark.py): the
``dwtDct`` scheme embeds only in the chroma (U/V) planes (``EmbedMaxDct``'s
default ``scales=(0, 36, 36)`` zeroes out luma), so it is destroyed by *any*
JPEG re-encode using the default 4:2:0 chroma subsampling -- even at quality
100 -- and separately by heavier quantization at lower qualities. A
non-match therefore says nothing about whether an image passed through a
Stable Diffusion pipeline that watermarks its *lossless* (PNG) output; it
only says the watermark, if any, did not survive whatever came after.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from imgforensics.core.base import BaseDetector
from imgforensics.core.image import ForensicImage
from imgforensics.core.registry import register
from imgforensics.core.types import DetectionResult, Label
from imgforensics.signals._vendor.dwtdct import EmbedMaxDct
from imgforensics.utils.image_io import to_numpy

# 48-bit SDXL / Stability reference-pipeline payload. Verified against
# diffusers `src/diffusers/pipelines/stable_diffusion_xl/watermark.py`
# (itself "Copied from https://github.com/Stability-AI/generative-models/blob/
# 613af104c6b85184091d42d374fef420eddb356d/scripts/demo/streamlit_helpers.py#L66"):
#     WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
#     WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
#     encoder.set_watermark("bits", WATERMARK_BITS); encoder.encode(image, "dwtDct")
_SDXL_WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
_SDXL_BITS: tuple[int, ...] = tuple(int(bit) for bit in bin(_SDXL_WATERMARK_MESSAGE)[2:])

# 136-bit (17-byte) CompVis Stable Diffusion v1 payload. Verified against
# CompVis/stable-diffusion `scripts/txt2img.py`:
#     wm = "StableDiffusionV1"
#     wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
#     ...
#     img = wm_encoder.encode(img, 'dwtDct')  # put_watermark()
# `WatermarkEncoder.set_by_bytes` (imwatermark/watermark.py) unpacks the
# UTF-8 bytes into individual bits MSB-first via ``np.unpackbits``, which is
# reproduced here.
_COMPVIS_SD_V1_PAYLOAD = b"StableDiffusionV1"
_COMPVIS_SD_V1_BITS: tuple[int, ...] = tuple(
    int(bit) for bit in np.unpackbits(np.frombuffer(_COMPVIS_SD_V1_PAYLOAD, dtype=np.uint8))
)

# name -> (payload bits, minimum bit-agreement fraction to count as "matched")
_KNOWN_PAYLOADS: dict[str, tuple[tuple[int, ...], float]] = {
    "sdxl_48bit": (_SDXL_BITS, 0.95),
    "compvis_sdv1_136bit": (_COMPVIS_SD_V1_BITS, 0.90),
}

# The reference encoders/decoders refuse images smaller than this (see
# WatermarkEncoder.encode / WatermarkDecoder.decode in imwatermark/watermark.py);
# below this the DWT sub-bands don't contain enough blocks to carry the
# longer payload reliably.
_MIN_PIXELS = 256 * 256


def _bit_agreement(rgb: np.ndarray, bits: tuple[int, ...]) -> float:
    """Fraction of ``bits`` that a dwtDct decode of ``rgb`` (HxWx3, RGB, uint8) agrees with."""
    bgr = rgb[:, :, ::-1]
    decoder = EmbedMaxDct(wm_len=len(bits))
    decoded = decoder.decode(bgr)
    target = np.array(bits, dtype=bool)
    return float(np.mean(decoded == target))


@register("sd_watermark")
class InvisibleWatermarkSignal(BaseDetector):
    """Decodes the Stable Diffusion ``dwtDct`` invisible watermark, if present.

    Reads ``image.rgb`` (no encoded original required) and, for each payload
    in ``_KNOWN_PAYLOADS``, decodes that many bits and measures agreement
    with the known payload (see module docstring for why bit-agreement,
    rather than a plain decode, is the right test). Requires the image to be
    at least 256x256, matching the reference implementation's own minimum.

    Score rules (first match wins):

    1. Any known payload's agreement clears its threshold ("matched") ->
       score 0.95, label "fake".
    2. Otherwise -> score 0.45, label "uncertain". Most images, including
       most AI-generated ones from generators other than Stable Diffusion,
       carry no watermark at all, and any lossy re-encoding (JPEG, resize)
       destroys this particular scheme even when it was originally present
       -- so a non-match is weak evidence at best, never proof of
       authenticity.

    Every computed field is placed in ``details``; this signal never raises,
    catching unexpected errors into ``details["error"]`` with score 0.5.
    """

    name = "sd_watermark"

    def predict(self, image: ForensicImage) -> DetectionResult:
        try:
            return self._predict(image)
        except Exception as exc:  # never raise: record and abstain
            return DetectionResult(
                detector=self.name,
                score=0.5,
                label="uncertain",
                details={"error": f"{type(exc).__name__}: {exc}"},
            )

    def _predict(self, image: ForensicImage) -> DetectionResult:
        width, height = image.width, image.height
        if width * height < _MIN_PIXELS:
            return DetectionResult(
                detector=self.name,
                score=0.5,
                label="uncertain",
                details={
                    "reason": (
                        f"image too small ({width}x{height}); dwtDct decoding "
                        "requires at least 256x256, matching the reference codec"
                    )
                },
            )

        rgb = to_numpy(image.rgb)

        agreements: dict[str, float] = {}
        matched: dict[str, bool] = {}
        for payload_name, (bits, threshold) in _KNOWN_PAYLOADS.items():
            agreement = _bit_agreement(rgb, bits)
            agreements[payload_name] = agreement
            matched[payload_name] = agreement >= threshold

        best_payload = max(agreements, key=lambda name: agreements[name])
        best_agreement = agreements[best_payload]
        any_matched = any(matched.values())

        details: dict[str, Any] = {
            "agreements": agreements,
            "matched": matched,
            "best_payload": best_payload,
            "best_agreement": best_agreement,
        }

        score: float
        label: Label
        if any_matched:
            score, label = 0.95, "fake"
        else:
            score, label = 0.45, "uncertain"
            details["note"] = (
                "no known SD watermark; absence is expected for most generators and after resizing"
            )

        return DetectionResult(detector=self.name, score=score, label=label, details=details)
