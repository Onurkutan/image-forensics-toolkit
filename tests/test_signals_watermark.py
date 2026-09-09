"""Tests for imgforensics.signals.watermark.InvisibleWatermarkSignal.

Robustness note (measured on 256x256 synthetic images, 3 seeds): the
vendored ``dwtDct`` scheme embeds only in the chroma (U/V) planes (see
``EmbedMaxDct``'s default ``scales=(0, 36, 36)``, which zeroes out the luma
channel), so it is destroyed by *any* JPEG re-encode that applies the
default 4:2:0 chroma subsampling -- even at quality 100 -- and additionally
by heavier quantization at lower qualities even without subsampling.
Round-trip "matched" tests below therefore save as PNG (lossless; also how
Stable Diffusion reference pipelines actually save their watermarked
output), matching the vendored codec's own contract. A dedicated JPEG q75
re-save test measures the resulting agreement directly: it dropped to ~0.44
(chance level, well under the 0.95 threshold) across all three seeds tried,
confirming the collapse is not seed-dependent.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from imgforensics.core.image import ForensicImage
from imgforensics.signals._vendor.dwtdct import EmbedMaxDct
from imgforensics.signals.watermark import (
    _COMPVIS_SD_V1_BITS,
    _SDXL_BITS,
    InvisibleWatermarkSignal,
)


def _synthetic_rgb(seed: int, size: int = 256) -> np.ndarray:
    """A smooth, photo-like (not flat, not pure noise) synthetic RGB image."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    base = np.sin(x / 20.0) * 40 + np.cos(y / 15.0) * 40 + 128
    channels = [base + rng.normal(0, 10, base.shape) for _ in range(3)]
    return np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)


def _embed(rgb: np.ndarray, bits: tuple[int, ...]) -> np.ndarray:
    """Embed ``bits`` into ``rgb`` (HxWx3 RGB uint8) with the vendored dwtDct encoder."""
    bgr = rgb[:, :, ::-1].copy()
    encoded_bgr = EmbedMaxDct(list(bits), wm_len=len(bits)).encode(bgr)
    return encoded_bgr[:, :, ::-1]


def _to_png_forensic_image(rgb: np.ndarray) -> ForensicImage:
    """Lossless round trip -- how Stable Diffusion pipelines actually save output."""
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    return ForensicImage.from_bytes(buffer.getvalue())


def _to_jpeg_forensic_image(rgb: np.ndarray, *, quality: int) -> ForensicImage:
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="JPEG", quality=quality)
    return ForensicImage.from_bytes(buffer.getvalue())


def test_sdxl_48bit_watermark_is_matched() -> None:
    rgb = _embed(_synthetic_rgb(seed=0), _SDXL_BITS)
    fi = _to_png_forensic_image(rgb)

    result = InvisibleWatermarkSignal().predict(fi)

    assert result.details["matched"]["sdxl_48bit"] is True
    assert result.details["agreements"]["sdxl_48bit"] >= 0.95
    assert result.score == pytest.approx(0.95)
    assert result.label == "fake"


def test_compvis_sdv1_136bit_watermark_is_matched() -> None:
    rgb = _embed(_synthetic_rgb(seed=1), _COMPVIS_SD_V1_BITS)
    fi = _to_png_forensic_image(rgb)

    result = InvisibleWatermarkSignal().predict(fi)

    assert result.details["matched"]["compvis_sdv1_136bit"] is True
    assert result.details["agreements"]["compvis_sdv1_136bit"] >= 0.90
    assert result.score == pytest.approx(0.95)
    assert result.label == "fake"


def test_unwatermarked_image_is_not_matched() -> None:
    rgb = _synthetic_rgb(seed=2)
    fi = _to_png_forensic_image(rgb)

    result = InvisibleWatermarkSignal().predict(fi)

    assert not any(result.details["matched"].values())
    # Chance level for an unrelated image is ~0.5; well under either threshold.
    assert result.details["best_agreement"] < 0.90
    assert result.score == pytest.approx(0.45)
    assert result.label == "uncertain"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_watermark_does_not_survive_jpeg_q75_resave(seed: int) -> None:
    rgb = _embed(_synthetic_rgb(seed=seed), _SDXL_BITS)
    fi = _to_jpeg_forensic_image(rgb, quality=75)

    result = InvisibleWatermarkSignal().predict(fi)

    # Measured agreement after one JPEG q75 re-save (default 4:2:0 chroma
    # subsampling) was ~0.44 (chance level) across seeds 0-2, consistently
    # below the 0.95 match threshold. This scheme embeds only in the chroma
    # planes, so it does not even need heavy quantization to collapse --
    # subsampling alone is enough (see the module-level docstring above).
    assert result.details["matched"]["sdxl_48bit"] is False
    assert result.details["agreements"]["sdxl_48bit"] < 0.70


def test_image_smaller_than_256_is_uncertain_with_reason() -> None:
    rgb = _synthetic_rgb(seed=3, size=128)
    fi = _to_jpeg_forensic_image(rgb, quality=95)

    result = InvisibleWatermarkSignal().predict(fi)

    assert result.score == pytest.approx(0.5)
    assert result.label == "uncertain"
    assert "reason" in result.details
    assert "256x256" in result.details["reason"]
