"""Tests for imgforensics.eval.preprocess: crop helpers and training-time augmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from conftest import natural_like_image

from imgforensics.eval.preprocess import (
    AugmentationConfig,
    augment,
    center_crop,
    grid_crops,
    random_crops,
)

_SIZE = (96, 64)


def test_center_crop_exact_size_when_image_larger() -> None:
    image = natural_like_image(size=(128, 128), seed=1)
    cropped = center_crop(image, 64)
    assert cropped.size == (64, 64)


def test_center_crop_pads_when_image_smaller() -> None:
    image = natural_like_image(size=(40, 30), seed=1)
    cropped = center_crop(image, 64)
    assert cropped.size == (64, 64)


def test_center_crop_is_centered() -> None:
    image = natural_like_image(size=(100, 100), seed=1)
    array = np.asarray(image)
    cropped = np.asarray(center_crop(image, 50))
    expected = array[25:75, 25:75]
    assert np.array_equal(cropped, expected)


def test_random_crops_returns_n_crops_of_requested_size() -> None:
    image = natural_like_image(size=_SIZE, seed=1)
    crops = random_crops(image, size=32, n=5, seed=0)
    assert len(crops) == 5
    assert all(crop.size == (32, 32) for crop in crops)


def test_random_crops_is_deterministic_given_seed() -> None:
    image = natural_like_image(size=_SIZE, seed=1)
    crops_a = random_crops(image, size=32, n=4, seed=42)
    crops_b = random_crops(image, size=32, n=4, seed=42)
    assert all(
        np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(crops_a, crops_b, strict=True)
    )


def test_random_crops_differ_across_seeds() -> None:
    image = natural_like_image(size=_SIZE, seed=1)
    crops_a = random_crops(image, size=32, n=4, seed=1)
    crops_b = random_crops(image, size=32, n=4, seed=2)
    assert any(
        not np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(crops_a, crops_b, strict=True)
    )


def test_grid_crops_are_non_overlapping_row_major_and_capped() -> None:
    image = natural_like_image(size=(64, 64), seed=1)
    crops = grid_crops(image, size=32, max_crops=100)
    assert len(crops) == 4  # 2x2 non-overlapping tiles
    assert all(crop.size == (32, 32) for crop in crops)

    capped = grid_crops(image, size=32, max_crops=2)
    assert len(capped) == 2


def test_grid_crops_pads_small_image() -> None:
    image = natural_like_image(size=(20, 20), seed=1)
    crops = grid_crops(image, size=32, max_crops=10)
    assert len(crops) == 1
    assert crops[0].size == (32, 32)


def test_augmentation_config_from_yaml(tmp_path: Path) -> None:
    yaml_text = """
    jpeg_quality: [40, 90]
    noise_sigma: [1.0, 4.0]
    cutout:
      size: 16
      count: 2
    p: 1.0
    """
    config_path = tmp_path / "aug.yaml"
    config_path.write_text(yaml_text, encoding="utf-8")

    config = AugmentationConfig.from_yaml(config_path)

    assert config.jpeg_quality == (40, 90)
    assert config.noise_sigma == (1.0, 4.0)
    assert config.cutout == {"size": 16, "count": 2}
    assert config.p == 1.0
    assert config.webp_quality is None


def test_augment_with_all_off_returns_same_size_image() -> None:
    image = natural_like_image(size=_SIZE, seed=1)
    config = AugmentationConfig()
    rng = np.random.default_rng(0)
    result = augment(image, config, rng)
    assert result.size == image.size
    assert result.mode == "RGB"


def test_augment_with_everything_on_is_reproducible_given_rng_state() -> None:
    image = natural_like_image(size=_SIZE, seed=1)
    config = AugmentationConfig(
        jpeg_quality=(30, 60),
        webp_quality=(30, 60),
        gaussian_blur_sigma=(0.5, 2.0),
        downscale_upscale=(0.5, 0.9),
        noise_sigma=(1.0, 5.0),
        cutout={"size": 8, "count": 2},
        p=1.0,
    )
    out_a = augment(image, config, np.random.default_rng(7))
    out_b = augment(image, config, np.random.default_rng(7))
    assert np.array_equal(np.asarray(out_a), np.asarray(out_b))
    assert out_a.size == image.size


def test_augment_with_p_zero_never_changes_pixels() -> None:
    image = natural_like_image(size=_SIZE, seed=1)
    config = AugmentationConfig(
        jpeg_quality=(30, 60),
        noise_sigma=(1.0, 5.0),
        p=0.0,
    )
    rng = np.random.default_rng(3)
    result = augment(image, config, rng)
    assert np.array_equal(np.asarray(result), np.asarray(image.convert("RGB")))
