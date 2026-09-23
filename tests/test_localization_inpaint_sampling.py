"""Tests for the ``dino_inpaint`` localizer's crop sampling and patch grid.

Deliberately torch-free: :mod:`imgforensics.localization.dino_inpaint` imports
no ``torch`` at module scope (it is imported on every CLI invocation for its
registration side effect), and these three functions are the part of the
design that is worth pinning down independently of any model -- where a
training crop lands, what target it carries, and how an image is cut up for
inference.
"""

from __future__ import annotations

import numpy as np
import pytest

from imgforensics.localization.dino_inpaint import (
    CROP_ALIGN,
    CROP_SIZE,
    PATCH,
    TILE_STRIDE,
    pad_for_tiles,
    patch_targets,
    sample_crop_box,
    tile_origins,
)

_DRAWS = 2000


# --- patch_targets() ---------------------------------------------------------


def test_patch_targets_reports_each_patchs_exact_mask_fraction() -> None:
    mask = np.zeros((4 * PATCH, 3 * PATCH), dtype=bool)
    # A rectangle covering one whole patch, half of the patch to its right,
    # and a quarter of the patch below it.
    mask[0:PATCH, 0:PATCH] = True
    mask[0:PATCH, PATCH : PATCH + PATCH // 2] = True
    mask[PATCH : PATCH + PATCH // 2, 0 : PATCH // 2] = True

    targets = patch_targets(mask)

    assert targets.shape == (4, 3)
    assert targets.dtype == np.float32
    assert targets[0, 0] == pytest.approx(1.0)
    assert targets[0, 1] == pytest.approx(0.5)
    assert targets[1, 0] == pytest.approx(0.25)
    assert targets[0, 2] == pytest.approx(0.0)
    assert targets[2:].sum() == pytest.approx(0.0)


def test_patch_targets_hard_mode_thresholds_at_half() -> None:
    mask = np.zeros((PATCH, 4 * PATCH), dtype=bool)
    mask[:, 0:PATCH] = True  # 1.0
    mask[0 : PATCH // 2 + 1, PATCH : 2 * PATCH] = True  # just above 0.5
    mask[0 : PATCH // 2, 2 * PATCH : 3 * PATCH] = True  # exactly 0.5

    soft = patch_targets(mask, mode="soft")
    hard = patch_targets(mask, mode="hard")

    assert soft[0].tolist() == pytest.approx([1.0, 0.5 + 1 / PATCH, 0.5, 0.0])
    # Exactly 0.5 is *not* inpainted: the rule is strictly greater, matching
    # the threshold every pixel metric in this project uses.
    assert hard[0].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_patch_targets_grid_is_the_image_divided_by_the_patch_size() -> None:
    assert patch_targets(np.zeros((CROP_SIZE, CROP_SIZE))).shape == (32, 32)
    # Rows and columns past the last whole patch are dropped, not partially
    # averaged: 683 px is 48 patches and 11 forgotten pixels.
    assert patch_targets(np.zeros((683, 1024))).shape == (48, 73)


def test_patch_targets_also_reduces_a_float_heatmap() -> None:
    heatmap = np.linspace(0.0, 1.0, PATCH * PATCH, dtype=np.float32).reshape(PATCH, PATCH)

    reduced = patch_targets(heatmap)

    assert reduced.shape == (1, 1)
    assert reduced[0, 0] == pytest.approx(float(heatmap.mean()), abs=1e-6)


# --- sample_crop_box() -------------------------------------------------------


def test_sample_crop_box_stays_inside_the_image_and_on_the_alignment_grid() -> None:
    rng = np.random.default_rng(0)

    for _ in range(_DRAWS):
        top, left = sample_crop_box(512, 1024, CROP_SIZE, rng)
        assert 0 <= top <= 512 - CROP_SIZE
        assert 0 <= left <= 1024 - CROP_SIZE
        assert top % CROP_ALIGN == 0
        assert left % CROP_ALIGN == 0


def test_sample_crop_box_guided_draws_always_cover_the_requested_pixel() -> None:
    rng = np.random.default_rng(1)
    height, width = 1024, 1024

    for _ in range(_DRAWS):
        pixel = (int(rng.integers(0, height)), int(rng.integers(0, width)))
        top, left = sample_crop_box(height, width, CROP_SIZE, rng, contains=pixel)
        assert top <= pixel[0] < top + CROP_SIZE
        assert left <= pixel[1] < left + CROP_SIZE
        assert 0 <= top <= height - CROP_SIZE
        assert 0 <= left <= width - CROP_SIZE


def test_sample_crop_box_guided_draws_do_not_center_the_pixel() -> None:
    rng = np.random.default_rng(2)
    pixel = (500, 500)

    offsets = np.asarray(
        [sample_crop_box(1024, 1024, CROP_SIZE, rng, contains=pixel)[0] for _ in range(_DRAWS)]
    )
    within_crop = pixel[0] - offsets

    assert np.all(offsets % CROP_ALIGN == 0)
    # The pixel lands almost anywhere in the crop, not in the middle of it.
    assert within_crop.min() < CROP_SIZE * 0.15
    assert within_crop.max() > CROP_SIZE * 0.85
    near_center = np.abs(within_crop - CROP_SIZE / 2) < CROP_SIZE * 0.1
    assert near_center.mean() < 0.3
    assert len(np.unique(offsets)) > 40


def test_sample_crop_box_guided_draws_near_the_edge_still_cover_the_pixel() -> None:
    # 683 - 448 = 235, which is not a multiple of 8, so the bottom rows have
    # no aligned corner that covers them; coverage wins over alignment there.
    rng = np.random.default_rng(3)

    for row in range(676, 683):
        top, _ = sample_crop_box(683, 1024, CROP_SIZE, rng, contains=(row, 10))
        assert top <= row < top + CROP_SIZE
        assert 0 <= top <= 683 - CROP_SIZE


def test_sample_crop_box_on_an_axis_shorter_than_the_crop_starts_at_zero() -> None:
    rng = np.random.default_rng(4)

    assert sample_crop_box(300, 300, CROP_SIZE, rng) == (0, 0)
    assert sample_crop_box(300, 1024, CROP_SIZE, rng)[0] == 0


# --- tile_origins() / pad_for_tiles() ----------------------------------------


def test_tile_origins_cover_the_extent_with_the_last_one_pulled_back() -> None:
    assert tile_origins(448, CROP_SIZE) == [0]
    assert tile_origins(266, 266) == [0]
    assert tile_origins(1024, CROP_SIZE) == [0, 336, 576]
    assert tile_origins(2048, CROP_SIZE) == [0, 336, 672, 1008, 1344, 1600]

    for extent in (462, 1024, 2044, 4096):
        origins = tile_origins(extent, CROP_SIZE)
        assert origins[0] == 0
        assert origins[-1] + CROP_SIZE == extent
        # No gap: consecutive tiles always overlap.
        assert all(
            later <= earlier + CROP_SIZE
            for earlier, later in zip(origins, origins[1:], strict=False)
        )


def test_tile_origins_stay_on_the_patch_grid_for_a_whole_patch_extent() -> None:
    for extent in (448, 1008, 1512, 3024):
        origins = tile_origins(extent, min(CROP_SIZE, extent))
        assert all(origin % PATCH == 0 for origin in origins), origins
    assert TILE_STRIDE % PATCH == 0


def test_pad_for_tiles_pads_a_small_image_up_to_whole_patches_only() -> None:
    small = np.zeros((256, 256, 3), dtype=np.uint8)

    padded, (top, left) = pad_for_tiles(small)

    # 256 -> 266 = 19 patches, not 448: the padding is invented pixels.
    assert padded.shape == (266, 266, 3)
    assert (top, left) == (5, 5)


def test_pad_for_tiles_leaves_an_image_at_or_above_the_tile_size_alone() -> None:
    large = np.zeros((683, 1024, 3), dtype=np.uint8)

    padded, offset = pad_for_tiles(large)

    assert padded is large
    assert offset == (0, 0)


def test_pad_for_tiles_pads_only_the_axis_that_is_short() -> None:
    wide = np.zeros((200, 1024, 3), dtype=np.uint8)

    padded, (top, left) = pad_for_tiles(wide)

    assert padded.shape == (210, 1024, 3)
    assert (top, left) == (5, 0)


def test_pad_for_tiles_gives_each_short_axis_its_own_target() -> None:
    # Both sides are under the tile size but need different amounts: padding
    # to the larger of the two would invent 136 rows of reflection.
    padded, (top, left) = pad_for_tiles(np.zeros((256, 384, 3), dtype=np.uint8))

    assert padded.shape == (266, 392, 3)
    assert (top, left) == (5, 4)

    tall, offset = pad_for_tiles(np.zeros((384, 256, 3), dtype=np.uint8))
    assert tall.shape == (392, 266, 3)
    assert offset == (4, 5)


def test_pad_for_tiles_reflects_rather_than_inventing_a_constant_border() -> None:
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)

    padded, (top, left) = pad_for_tiles(pixels)

    # BORDER_REFLECT_101: the padded row mirrors the image without repeating
    # its edge row, so row `top - 1` equals image row 1.
    assert np.array_equal(padded[top - 1], padded[top + 1])
    assert np.array_equal(padded[top : top + 256, left : left + 256], pixels)
