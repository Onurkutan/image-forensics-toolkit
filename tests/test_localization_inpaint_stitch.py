"""Tests for the ``dino_inpaint`` localizer's tile stitching.

Also torch-free. The property under test is the one the tiling exists to
guarantee: cutting a map into overlapping tiles and averaging them back
together reproduces the map, so a heatmap has no seam at a tile boundary and
the padding an undersized image needed never reaches the caller.
"""

from __future__ import annotations

import numpy as np
import pytest

from imgforensics.localization.dino_inpaint import (
    CROP_SIZE,
    TILE_STRIDE,
    pad_for_tiles,
    stitch_tiles,
    tile_origins,
)


def _probability_map(height: int, width: int, seed: int = 0) -> np.ndarray:
    """A smooth, strictly-inside-[0, 1] map, like a real heatmap."""
    rng = np.random.default_rng(seed)
    rows = np.linspace(0.05, 0.95, height, dtype=np.float64)[:, None]
    columns = np.linspace(0.95, 0.05, width, dtype=np.float64)[None, :]
    return np.clip((rows + columns) / 2 + rng.normal(0, 0.01, (height, width)), 0.01, 0.99)


def _cut(source: np.ndarray, tile_height: int, tile_width: int) -> list:
    return [
        (top, left, source[top : top + tile_height, left : left + tile_width])
        for top in tile_origins(source.shape[0], tile_height, TILE_STRIDE)
        for left in tile_origins(source.shape[1], tile_width, TILE_STRIDE)
    ]


@pytest.mark.parametrize("shape", [(1024, 1024), (448, 448), (683, 1024), (2048, 462)])
def test_stitching_overlapping_tiles_reproduces_the_map(shape: tuple[int, int]) -> None:
    source = _probability_map(*shape)
    tile_height = min(CROP_SIZE, shape[0])
    tile_width = min(CROP_SIZE, shape[1])

    stitched = stitch_tiles(_cut(source, tile_height, tile_width), shape)

    assert stitched.shape == shape
    assert stitched.dtype == np.float32
    assert np.abs(stitched.astype(np.float64) - source).max() < 1e-6


def test_a_padded_small_image_is_cropped_back_to_its_own_pixels() -> None:
    pixels = np.zeros((256, 256, 3), dtype=np.uint8)
    padded, (pad_top, pad_left) = pad_for_tiles(pixels)
    assert padded.shape[:2] == (266, 266)

    source = _probability_map(266, 266, seed=1)
    tiles = _cut(source, 266, 266)
    assert len(tiles) == 1

    stitched = stitch_tiles(tiles, (266, 266), crop=(pad_top, pad_left, 256, 256))

    assert stitched.shape == (256, 256)
    expected = source[pad_top : pad_top + 256, pad_left : pad_left + 256]
    assert np.abs(stitched.astype(np.float64) - expected).max() < 1e-6


def test_an_oblong_small_image_is_padded_per_axis_not_to_a_square() -> None:
    pixels = np.zeros((256, 384, 3), dtype=np.uint8)

    padded, (pad_top, pad_left) = pad_for_tiles(pixels)

    # 256 -> 266 (19 patches) and 384 -> 392 (28 patches), each on its own:
    # padding the height to 392 as well would invent 136 rows of reflection.
    assert padded.shape[:2] == (266, 392)
    assert (pad_top, pad_left) == (5, 4)

    source = _probability_map(266, 392, seed=2)
    tiles = _cut(source, 266, 392)
    assert len(tiles) == 1

    stitched = stitch_tiles(tiles, (266, 392), crop=(pad_top, pad_left, 256, 384))

    assert stitched.shape == (256, 384)
    expected = source[pad_top : pad_top + 256, pad_left : pad_left + 384]
    assert np.abs(stitched.astype(np.float64) - expected).max() < 1e-6


def test_stitching_averages_disagreeing_tiles_rather_than_letting_one_win() -> None:
    left_tile = np.full((4, 4), 0.2)
    right_tile = np.full((4, 4), 0.6)

    stitched = stitch_tiles([(0, 0, left_tile), (0, 2, right_tile)], (4, 6))

    assert stitched[0, 0] == pytest.approx(0.2)
    assert stitched[0, 5] == pytest.approx(0.6)
    # The two-pixel overlap is the mean of the two, not either of them.
    assert stitched[0, 2] == pytest.approx(0.4)
    assert stitched[0, 3] == pytest.approx(0.4)


def test_stitching_clips_into_the_probability_range() -> None:
    stitched = stitch_tiles([(0, 0, np.array([[-0.5, 1.5]]))], (1, 2))

    assert stitched.tolist() == [[0.0, 1.0]]


def test_stitching_refuses_an_empty_or_incomplete_cover() -> None:
    with pytest.raises(ValueError, match="at least one tile"):
        stitch_tiles([], (4, 4))

    with pytest.raises(ValueError, match="uncovered"):
        stitch_tiles([(0, 0, np.zeros((2, 2)))], (4, 4))
