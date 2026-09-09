"""Tests for imgforensics.signals.jpeg_ghost.JPEGGhostSignal and _ghost_analysis.

Calibration note (measured on the fixtures below, 512x512, patch at
(192,192)-(320,320) -- see module docstring in jpeg_ghost.py for how the
implementation refines the textbook algorithm): the spliced
patch (previously saved at q60, pasted into a q90 background) gives
``dominant_quality == 90`` and a patch best-quality median of 60.0 (58/64
blocks land exactly on 60), with heatmap depth ~1.7 inside the patch versus
~0 (slightly negative) outside -- effectively unbounded contrast, comfortably
over the required 2x. An untouched, never-JPEG-compressed image gave
``ghost_fraction`` between 0.017 and 0.031 and score between 0.43 and 0.53
across several seeds tried, comfortably under the 0.05 / 0.55 ceilings.
"""

from __future__ import annotations

import io
import time

import numpy as np
from conftest import natural_like_image
from PIL import Image

from imgforensics.core.image import ForensicImage
from imgforensics.signals.jpeg_ghost import JPEGGhostSignal, _ghost_analysis

_SIZE = (512, 512)
_PATCH_BOX = (192, 192, 320, 320)  # 128x128, 16px-aligned
_Q_PATCH = 60
_Q_BACKGROUND = 90


def _spliced_image(seed: int = 0) -> Image.Image:
    base = natural_like_image(size=_SIZE, seed=seed)

    bg_buffer = io.BytesIO()
    base.save(bg_buffer, format="JPEG", quality=_Q_BACKGROUND)
    bg_buffer.seek(0)
    background = Image.open(bg_buffer).convert("RGB")

    patch = base.crop(_PATCH_BOX)
    patch_buffer = io.BytesIO()
    patch.save(patch_buffer, format="JPEG", quality=_Q_PATCH)
    patch_buffer.seek(0)
    patch_q60 = Image.open(patch_buffer).convert("RGB")

    composite = background.copy()
    composite.paste(patch_q60, _PATCH_BOX[:2])
    return composite


def _region_means(heatmap: np.ndarray) -> tuple[float, float]:
    x0, y0, x1, y1 = _PATCH_BOX
    inside = heatmap[y0:y1, x0:x1]
    mask = np.ones(heatmap.shape, dtype=bool)
    mask[y0:y1, x0:x1] = False
    return float(inside.mean()), float(heatmap[mask].mean())


def test_ghost_analysis_finds_spliced_patch_quality() -> None:
    composite = _spliced_image()

    analysis = _ghost_analysis(composite)

    assert abs(analysis["dominant_quality"] - _Q_BACKGROUND) <= 5

    block = 16
    px0, py0, px1, py1 = (coord // block for coord in _PATCH_BOX)
    patch_best = analysis["best_quality_map"][py0:py1, px0:px1]
    assert abs(float(np.median(patch_best)) - _Q_PATCH) <= 5

    assert analysis["best_quality_map"].shape == (_SIZE[1] // block, _SIZE[0] // block)


def test_jpeg_ghost_signal_heatmap_highlights_patch() -> None:
    fi = ForensicImage.from_pil(_spliced_image())

    result = JPEGGhostSignal().predict(fi)

    assert result.heatmap is not None
    assert result.heatmap.shape == (fi.height, fi.width)
    inside, outside = _region_means(result.heatmap)
    assert inside > 2 * max(outside, 1e-6)

    assert result.details["dominant_quality"] == 90 or (
        abs(result.details["dominant_quality"] - 90) <= 5
    )
    assert 0.3 <= result.score <= 0.7
    assert "best_quality_map_shape" in result.details
    assert "best_quality_histogram" in result.details


def test_jpeg_ghost_untouched_png_scores_low() -> None:
    fi = ForensicImage.from_pil(natural_like_image(size=_SIZE, seed=1))

    result = JPEGGhostSignal().predict(fi)

    assert result.details["ghost_fraction"] < 0.05, result.details
    assert result.score <= 0.55, result.details


def test_jpeg_ghost_runs_fast_on_12mp_image() -> None:
    fi = ForensicImage.from_pil(natural_like_image(size=(3000, 4000), seed=2))

    start = time.perf_counter()
    result = JPEGGhostSignal().predict(fi)
    elapsed = time.perf_counter() - start

    assert elapsed < 4.0, f"jpeg_ghost took {elapsed:.2f}s on a 12MP image"
    assert result.details["scale"] < 1.0
    assert result.heatmap is not None
    assert result.heatmap.shape == (fi.height, fi.width)


def test_jpeg_ghost_never_raises_on_tiny_image() -> None:
    tiny = Image.new("RGB", (8, 8), color=(120, 120, 120))
    fi = ForensicImage.from_pil(tiny)

    result = JPEGGhostSignal().predict(fi)

    assert result.label in {"real", "fake", "uncertain"}
