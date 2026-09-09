"""Tests for imgforensics.signals.copymove.CopyMoveSignal.

The synthetic copy-move fixture pastes a patch at an offset that is a
multiple of the block step (4 px): the block-matching grid only ever samples
positions on that step, so a pasted region is only *guaranteed* to line up
exactly with a grid-sampled source block when the paste offset itself is a
multiple of the step -- a real-world limitation of this class of algorithm,
not an artefact of the test.
"""

from __future__ import annotations

import io
import time

import numpy as np
from conftest import natural_like_image
from PIL import Image, ImageDraw

from imgforensics.core.image import ForensicImage
from imgforensics.signals.copymove import CopyMoveSignal

_SIZE = (512, 512)
_PATCH_BOX = (40, 40, 136, 136)  # 96x96
_OFFSET = (256, 200)  # multiple of the 4px block step; euclidean dist ~325px


def _copy_move_image(seed: int = 0) -> Image.Image:
    base = natural_like_image(size=_SIZE, seed=seed)
    dest_xy = (_PATCH_BOX[0] + _OFFSET[0], _PATCH_BOX[1] + _OFFSET[1])
    patch = base.crop(_PATCH_BOX)
    tampered = base.copy()
    tampered.paste(patch, dest_xy)
    return tampered


def _dest_box() -> tuple[int, int, int, int]:
    x0, y0 = _PATCH_BOX[0] + _OFFSET[0], _PATCH_BOX[1] + _OFFSET[1]
    return x0, y0, x0 + 96, y0 + 96


def _region_means(heatmap: np.ndarray) -> tuple[float, float]:
    """(mean inside the two 96x96 source/dest regions, mean outside both)."""
    sx0, sy0, sx1, sy1 = _PATCH_BOX
    dx0, dy0, dx1, dy1 = _dest_box()
    inside = np.concatenate([heatmap[sy0:sy1, sx0:sx1].ravel(), heatmap[dy0:dy1, dx0:dx1].ravel()])
    mask = np.ones(heatmap.shape, dtype=bool)
    mask[sy0:sy1, sx0:sx1] = False
    mask[dy0:dy1, dx0:dx1] = False
    return float(inside.mean()), float(heatmap[mask].mean())


def _checkerboard_image(size: tuple[int, int] = _SIZE, square: int = 32) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    colors = [(210, 70, 60), (60, 90, 210)]
    for i in range(0, width, square):
        for j in range(0, height, square):
            color = colors[((i // square) + (j // square)) % 2]
            draw.rectangle([i, j, i + square, j + square], fill=color)
    return image


def test_copy_move_detects_duplicated_patch() -> None:
    fi = ForensicImage.from_pil(_copy_move_image())

    result = CopyMoveSignal().predict(fi)

    assert result.details["accepted"] is True
    dx, dy = result.details["dominant_shift"]
    assert abs(dx - _OFFSET[0]) <= 4
    assert abs(dy - _OFFSET[1]) <= 4
    assert result.score == 0.85
    assert result.label == "fake"

    assert result.heatmap is not None
    assert result.heatmap.shape == (fi.height, fi.width)
    inside, outside = _region_means(result.heatmap)
    assert inside > 3 * max(outside, 1e-6)


def test_copy_move_survives_jpeg_q85_resave() -> None:
    tampered = _copy_move_image()
    buffer = io.BytesIO()
    tampered.save(buffer, format="JPEG", quality=85)
    fi = ForensicImage.from_bytes(buffer.getvalue())

    result = CopyMoveSignal().predict(fi)

    assert result.details["accepted"] is True, (
        f"copy-move should still be detected after a q85 JPEG re-save; details={result.details}"
    )
    dx, dy = result.details["dominant_shift"]
    assert abs(dx - _OFFSET[0]) <= 4
    assert abs(dy - _OFFSET[1]) <= 4


def test_copy_move_untouched_image_not_accepted() -> None:
    fi = ForensicImage.from_pil(natural_like_image(size=_SIZE, seed=1))

    result = CopyMoveSignal().predict(fi)

    assert result.details["accepted"] is False
    assert result.score == 0.45
    assert result.label == "uncertain"
    assert "note" in result.details


def test_copy_move_periodic_tile_not_accepted() -> None:
    fi = ForensicImage.from_pil(_checkerboard_image())

    result = CopyMoveSignal().predict(fi)

    assert result.details["accepted"] is False, (
        "a periodic checkerboard should be rejected by the flat-block filter, "
        f"min_distance gate, or shift-consistency guard; details={result.details}"
    )


def test_copy_move_runs_fast_on_12mp_image() -> None:
    fi = ForensicImage.from_pil(natural_like_image(size=(3000, 4000), seed=2))

    start = time.perf_counter()
    result = CopyMoveSignal().predict(fi)
    elapsed = time.perf_counter() - start

    assert elapsed < 4.0, f"copy_move took {elapsed:.2f}s on a 12MP image"
    assert result.details["scale"] < 1.0
    assert result.heatmap is not None
    assert result.heatmap.shape == (fi.height, fi.width)


def test_copy_move_never_raises_on_tiny_image() -> None:
    tiny = Image.new("RGB", (8, 8), color=(120, 120, 120))
    fi = ForensicImage.from_pil(tiny)

    result = CopyMoveSignal().predict(fi)

    assert result.label in {"real", "fake", "uncertain"}
    assert "error" not in result.details
