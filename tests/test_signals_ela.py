"""Tests for imgforensics.signals.ela.ELASignal and ela_map.

Two fixtures exercise both directions: a patch more compressed than its
surroundings (lower ELA) and one less compressed (higher ELA). ELA exposes
a difference in compression history either way.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from imgforensics.core.image import ForensicImage
from imgforensics.signals.ela import ELASignal, ela_map

_SIZE = 256
_PATCH_BOX = (96, 96, 160, 160)  # 64x64 region


def _gradient_noise_image(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0, 255, _SIZE).astype(np.float32)
    gradient = np.tile(ramp, (_SIZE, 1))
    gradient_rgb = np.stack([gradient, gradient.T, (gradient + gradient.T) / 2], axis=2)
    noise = rng.normal(0, 5, gradient_rgb.shape)
    array = np.clip(gradient_rgb + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _tampered_jpeg_bytes() -> bytes:
    base = _gradient_noise_image()

    patch = base.crop(_PATCH_BOX)
    patch_buffer = io.BytesIO()
    patch.save(patch_buffer, format="JPEG", quality=50)
    patch_buffer.seek(0)
    with Image.open(patch_buffer) as reloaded_patch:
        patch_q50 = reloaded_patch.convert("RGB")

    tampered = base.copy()
    tampered.paste(patch_q50, _PATCH_BOX)

    out_buffer = io.BytesIO()
    tampered.save(out_buffer, format="JPEG", quality=90)
    return out_buffer.getvalue()


def _resave_as_jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as reloaded:
        return reloaded.convert("RGB")


def _bright_patch_jpeg_bytes() -> bytes:
    """The classic ELA direction: a never-recompressed patch pasted into a
    background that has already been through two rounds of lossy JPEG
    compression (q75, q75). The background has lost most of its
    high-frequency content and barely changes under a further recompression;
    the fresh patch still has it to lose, so it lights up more than the
    background.
    """
    base = _gradient_noise_image()

    once = _resave_as_jpeg(base, 75)
    twice = _resave_as_jpeg(once, 75)

    composite = twice.copy()
    composite.paste(base.crop(_PATCH_BOX), _PATCH_BOX)

    out_buffer = io.BytesIO()
    composite.save(out_buffer, format="JPEG", quality=90)
    return out_buffer.getvalue()


def _region_means(heatmap: np.ndarray) -> tuple[float, float]:
    x0, y0, x1, y1 = _PATCH_BOX
    inside = heatmap[y0:y1, x0:x1]
    mask = np.ones(heatmap.shape, dtype=bool)
    mask[y0:y1, x0:x1] = False
    return float(inside.mean()), float(heatmap[mask].mean())


def test_ela_map_shape_and_range() -> None:
    fi = ForensicImage.from_bytes(_tampered_jpeg_bytes())

    heatmap = ela_map(fi.rgb, quality=95)

    assert heatmap.shape == (fi.height, fi.width)
    assert heatmap.dtype == np.float32
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_ela_separates_tampered_patch_from_surroundings() -> None:
    fi = ForensicImage.from_bytes(_tampered_jpeg_bytes())

    heatmap = ela_map(fi.rgb, quality=95)
    inside, outside = _region_means(heatmap)

    # Clear, robust separation between the recompressed patch and the rest
    # of the image (see module docstring for the direction).
    assert outside - inside > 0.05


def test_ela_highlights_uncompressed_patch_pasted_into_compressed_background() -> None:
    fi = ForensicImage.from_bytes(_bright_patch_jpeg_bytes())

    heatmap = ela_map(fi.rgb, quality=95)
    inside, outside = _region_means(heatmap)

    # Opposite direction from the tampered fixture above: here the patch is
    # the less-compressed region, so it shows *more* error than the twice
    # re-compressed background.
    assert inside - outside > 0.05


def test_ela_signal_score_and_label() -> None:
    fi = ForensicImage.from_bytes(_tampered_jpeg_bytes())

    result = ELASignal().predict(fi)

    assert 0.3 <= result.score <= 0.65
    assert result.label in {"real", "fake", "uncertain"}
    assert result.heatmap is not None
    assert result.heatmap.shape == (fi.height, fi.width)
    assert result.details["quality"] == 95
    assert "mean_ela" in result.details
    assert "p99_ela" in result.details
    assert "high_region_fraction" in result.details
    assert "note" in result.details
    assert np.isfinite(result.details["block_inhomogeneity"])


def test_ela_works_on_png_input() -> None:
    image = _gradient_noise_image(seed=1)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    fi = ForensicImage.from_bytes(buffer.getvalue())

    assert fi.format == "PNG"
    result = ELASignal().predict(fi)

    assert result.heatmap is not None
    assert result.heatmap.dtype == np.float32
    assert result.heatmap.min() >= 0.0
    assert result.heatmap.max() <= 1.0


def test_ela_works_on_from_pil_input() -> None:
    image = _gradient_noise_image(seed=2)
    fi = ForensicImage.from_pil(image)

    assert fi.raw is None
    result = ELASignal().predict(fi)

    assert result.heatmap is not None
    assert 0.3 <= result.score <= 0.65


def test_ela_map_pure_function_reuse() -> None:
    image = _gradient_noise_image(seed=3)

    heatmap_default = ela_map(image)
    heatmap_q80 = ela_map(image, quality=80)

    assert heatmap_default.shape == heatmap_q80.shape
    assert heatmap_default.dtype == np.float32 == heatmap_q80.dtype


def test_ela_map_handles_flat_image_without_dividing_by_zero() -> None:
    flat = Image.new("RGB", (32, 32), color=(128, 128, 128))

    heatmap = ela_map(flat, quality=95)

    assert np.isfinite(heatmap).all()
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
