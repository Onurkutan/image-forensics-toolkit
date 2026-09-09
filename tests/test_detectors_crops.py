"""Tests for the torch-free surface of imgforensics.detectors.

Everything here is pure PIL/numpy on purpose -- the crop policy, the backbone
registry and the weights-cache configuration are the half of the
learned-detector pipeline that must keep working (and keep being tested)
without the optional ``ml`` extra installed, so these tests run in the default
CI job. The last two tests pin that guarantee down: one imports the package in
a subprocess where ``torch`` and ``timm`` are made unimportable, the other
checks that the ``features`` CLI commands say so and exit 1 rather than
raising :class:`ImportError` at the user.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from conftest import natural_like_image
from PIL import Image
from typer.testing import CliRunner

from imgforensics import cli as cli_module
from imgforensics.cli import app
from imgforensics.detectors import crops as crops_module
from imgforensics.detectors.backbones import (
    BACKBONES,
    WEIGHTS_DIR_ENV,
    _apply_weights_dir_env,
    get_backbone,
)
from imgforensics.detectors.crops import CropPolicy, crops_for, to_array

_CROP_SIZE = 224


def _as_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def test_center_mode_returns_one_crop_of_exact_size() -> None:
    image = natural_like_image(size=(500, 300), seed=1)
    result = crops_for(image, CropPolicy(size=_CROP_SIZE, mode="center", max_crops=4))

    assert len(result) == 1
    assert result[0].size == (_CROP_SIZE, _CROP_SIZE)
    repeated = crops_for(image, CropPolicy(size=_CROP_SIZE, mode="center"))
    assert np.array_equal(_as_array(result[0]), _as_array(repeated[0]))


def test_grid_mode_takes_tiles_closest_to_the_center_first() -> None:
    # 3x3 whole tiles of 224 px: the center tile starts at (224, 224).
    image = natural_like_image(size=(3 * _CROP_SIZE, 3 * _CROP_SIZE), seed=2)
    policy = CropPolicy(size=_CROP_SIZE, mode="grid", max_crops=4)

    result = crops_for(image, policy)
    assert len(result) == 4
    assert all(crop.size == (_CROP_SIZE, _CROP_SIZE) for crop in result)

    array = _as_array(image)
    center_tile = array[_CROP_SIZE : 2 * _CROP_SIZE, _CROP_SIZE : 2 * _CROP_SIZE]
    assert np.array_equal(_as_array(result[0]), center_tile)

    again = crops_for(image, policy)
    for first, second in zip(result, again, strict=True):
        assert np.array_equal(_as_array(first), _as_array(second))


def test_grid_mode_returns_fewer_crops_than_requested_on_a_small_image() -> None:
    image = natural_like_image(size=(2 * _CROP_SIZE, _CROP_SIZE), seed=3)
    result = crops_for(image, CropPolicy(size=_CROP_SIZE, mode="grid", max_crops=8))

    assert len(result) == 2


def test_random_mode_is_reproducible_from_the_seed_material() -> None:
    image = natural_like_image(size=(600, 600), seed=4)
    policy = CropPolicy(size=_CROP_SIZE, mode="random", max_crops=3)

    first = crops_for(image, policy, seed_material=b"image-bytes")
    second = crops_for(image, policy, seed_material=b"image-bytes")
    other_image = crops_for(image, policy, seed_material=b"other-image-bytes")
    other_seed = crops_for(
        image,
        CropPolicy(size=_CROP_SIZE, mode="random", max_crops=3, seed=7),
        seed_material=b"image-bytes",
    )

    assert len(first) == 3
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(_as_array(a), _as_array(b))
    assert not np.array_equal(_as_array(first[0]), _as_array(other_image[0]))
    assert not np.array_equal(_as_array(first[0]), _as_array(other_seed[0]))


def test_random_mode_falls_back_to_pixel_content_for_its_seed() -> None:
    image = natural_like_image(size=(600, 600), seed=5)
    policy = CropPolicy(size=_CROP_SIZE, mode="random", max_crops=2)

    first = crops_for(image, policy)
    second = crops_for(image, policy)
    different = crops_for(natural_like_image(size=(600, 600), seed=6), policy)

    for a, b in zip(first, second, strict=True):
        assert np.array_equal(_as_array(a), _as_array(b))
    assert not np.array_equal(_as_array(first[0]), _as_array(different[0]))


@pytest.mark.parametrize("mode", ["center", "grid", "random"])
def test_small_images_are_reflection_padded_not_upscaled(mode: str) -> None:
    image = natural_like_image(size=(64, 48), seed=7)
    result = crops_for(image, CropPolicy(size=_CROP_SIZE, mode=mode, max_crops=2))

    assert result
    assert all(crop.size == (_CROP_SIZE, _CROP_SIZE) for crop in result)
    # Reflection padding repeats real pixels, so the crop cannot be uniform.
    assert _as_array(result[0]).std() > 0


def test_small_image_raises_when_padding_is_disabled() -> None:
    image = natural_like_image(size=(64, 64), seed=8)
    policy = CropPolicy(size=_CROP_SIZE, mode="center", min_side_pad=False)

    with pytest.raises(ValueError, match="smaller than the 224px crop size"):
        crops_for(image, policy)


def test_to_array_normalizes_channels_with_the_given_statistics() -> None:
    crop = Image.new("RGB", (8, 8), color=(255, 128, 0))
    mean = (0.5, 0.25, 0.75)
    std = (0.5, 0.5, 0.25)

    array = to_array([crop, crop], mean, std)

    assert array.shape == (2, 3, 8, 8)
    assert array.dtype == np.float32
    expected = [
        (255 / 255 - mean[0]) / std[0],
        (128 / 255 - mean[1]) / std[1],
        (0 / 255 - mean[2]) / std[2],
    ]
    for channel, value in enumerate(expected):
        assert np.allclose(array[:, channel], value, atol=1e-6)


def test_to_array_rejects_empty_input_and_bad_statistics() -> None:
    crop = Image.new("RGB", (4, 4), color=(10, 20, 30))

    with pytest.raises(ValueError, match="at least one crop"):
        to_array([], (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="must have 3 elements"):
        to_array([crop], (0.5, 0.5), (0.5, 0.5, 0.5))


def test_crop_policy_fingerprint_tracks_field_values() -> None:
    base = CropPolicy()

    assert base.fingerprint() == CropPolicy().fingerprint()
    assert base.fingerprint() != CropPolicy(max_crops=8).fingerprint()
    assert base.fingerprint() != CropPolicy(mode="center").fingerprint()
    assert len(base.fingerprint()) == 12


def test_seed_derivation_matches_the_robustness_suite_pattern() -> None:
    import hashlib

    policy = CropPolicy(seed=3)
    digest = hashlib.sha256(b"material" + b"3").digest()

    assert crops_module._seed_for(policy, b"material") == int.from_bytes(digest[:8], "big")


def test_detectors_package_imports_without_torch_or_timm() -> None:
    """The detectors package and the CLI must import with the ``ml`` extra absent."""
    snippet = textwrap.dedent(
        """
        import sys

        blocked = {"torch", "timm", "torchvision"}

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in blocked:
                    raise ImportError(f"blocked for this test: {name}")
                return None

        sys.meta_path.insert(0, Blocker())

        import imgforensics.cli  # noqa: F401
        import imgforensics.detectors as detectors

        assert detectors.is_ml_available() is False
        assert not blocked & set(sys.modules)
        print("imported cleanly")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "imported cleanly" in completed.stdout


@pytest.mark.parametrize("command", [["features", "info"], ["features", "extract", "MANIFEST"]])
def test_features_cli_exits_with_a_message_when_the_ml_extra_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: list[str]
) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(cli_module.detectors, "is_ml_available", lambda: False)
    argv = [manifest_path.as_posix() if part == "MANIFEST" else part for part in command]

    result = CliRunner().invoke(app, argv)

    assert result.exit_code == 1
    assert "ml" in result.stdout
    assert "not installed" in result.stdout


def test_backbone_registry_exposes_both_frozen_spaces() -> None:
    dinov2 = get_backbone("dinov2_vitb14")
    clip = get_backbone("clip_vitl14")

    assert dinov2.timm_id == "vit_base_patch14_dinov2.lvd142m"
    assert dinov2.input_size == _CROP_SIZE
    assert dinov2.layers == [8, 9, 10, 11]
    assert clip.layers == [20, 21, 22, 23]
    # Normalization is read from the weights' own timm config, not hardcoded.
    assert all(spec.mean is None and spec.std is None for spec in BACKBONES.values())

    with pytest.raises(KeyError, match="clip_vitl14"):
        get_backbone("no_such_backbone")


def test_weights_dir_env_redirects_the_hub_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv(WEIGHTS_DIR_ENV, raising=False)

    assert _apply_weights_dir_env() is None
    assert "HF_HOME" not in os.environ

    weights = tmp_path / "weights"
    monkeypatch.setenv(WEIGHTS_DIR_ENV, str(weights))

    assert _apply_weights_dir_env() == weights
    assert weights.is_dir()
    assert os.environ["HF_HOME"] == str(weights)
    assert os.environ["HF_HUB_CACHE"] == str(weights / "hub")
