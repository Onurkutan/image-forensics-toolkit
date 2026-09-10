"""Tests for the deterministic robustness suite (imgforensics.eval.robustness)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import piexif
import pytest
from conftest import natural_like_image
from PIL import Image
from pydantic import ValidationError

from imgforensics.core.image import ForensicImage
from imgforensics.eval.robustness import Perturbation, RobustnessSuite, preserves_geometry

_SIZE = (96, 96)

_EXPECTED_LEVEL_ORDER = [
    "clean",
    "jpeg_q95",
    "jpeg_q85",
    "jpeg_q75",
    "jpeg_q60",
    "jpeg_q50",
    "webp_q80",
    "resize_0.75",
    "resize_0.5",
    "resize_0.25",
    "roundtrip_0.5",
    "crop_0.8",
    "noise_2",
    "noise_5",
    "social_1080_q80",
]


def _forensic_image(seed: int, size: tuple[int, int] = _SIZE) -> ForensicImage:
    rgb = natural_like_image(size=size, seed=seed)
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=92)
    return ForensicImage.from_bytes(buffer.getvalue())


_EXPECTED_LOCALIZATION_LEVELS = [
    "clean",
    "jpeg_q95",
    "jpeg_q85",
    "jpeg_q75",
    "jpeg_q60",
    "jpeg_q50",
    "webp_q80",
    "noise_2",
    "noise_5",
]


def test_default_suite_has_expected_levels_in_order() -> None:
    suite = RobustnessSuite.default()
    assert [level.name for level in suite.levels] == _EXPECTED_LEVEL_ORDER
    assert suite.version == 1


def test_localization_suite_is_the_geometry_preserving_half_of_the_default() -> None:
    suite = RobustnessSuite.localization()

    assert [level.name for level in suite.levels] == _EXPECTED_LOCALIZATION_LEVELS
    assert suite.version == 1
    assert all(preserves_geometry(level.kind) for level in suite.levels)


def test_localization_levels_share_the_default_suites_params() -> None:
    """A robustness row produced under one suite has to be comparable to the same
    row under the other, so a level of the same name must be the same level.
    """
    default_levels = {level.name: level for level in RobustnessSuite.default().levels}

    for level in RobustnessSuite.localization().levels:
        assert level == default_levels[level.name]


def test_preserves_geometry_splits_the_perturbation_kinds() -> None:
    for kind in ("clean", "jpeg", "webp", "gaussian_noise"):
        assert preserves_geometry(kind), kind
    for kind in ("resize", "resize_roundtrip", "center_crop", "social"):
        assert not preserves_geometry(kind), kind
    # An unrecognized kind is reported as non-preserving, the safe answer.
    assert not preserves_geometry("some_future_kind")


def test_clean_returns_input_unchanged() -> None:
    suite = RobustnessSuite.default()
    image = _forensic_image(seed=1)
    clean = next(level for level in suite.levels if level.name == "clean")
    assert suite.apply(clean, image) is image


def test_noise_perturbation_is_byte_identical_across_calls() -> None:
    suite = RobustnessSuite.default()
    image = _forensic_image(seed=1)
    noise_2 = next(level for level in suite.levels if level.name == "noise_2")

    first = suite.apply(noise_2, image)
    second = suite.apply(noise_2, image)

    assert first.raw is not None and second.raw is not None
    assert first.raw == second.raw


def test_noise_perturbation_differs_across_images() -> None:
    suite = RobustnessSuite.default()
    noise_2 = next(level for level in suite.levels if level.name == "noise_2")

    out_a = suite.apply(noise_2, _forensic_image(seed=1))
    out_b = suite.apply(noise_2, _forensic_image(seed=2))

    assert out_a.raw != out_b.raw


def test_noise_perturbation_differs_across_levels_on_same_image() -> None:
    suite = RobustnessSuite.default()
    image = _forensic_image(seed=1)
    noise_2 = next(level for level in suite.levels if level.name == "noise_2")
    noise_5 = next(level for level in suite.levels if level.name == "noise_5")

    out_2 = suite.apply(noise_2, image)
    out_5 = suite.apply(noise_5, image)

    assert out_2.raw != out_5.raw


def test_jpeg_perturbation_format_and_size() -> None:
    suite = RobustnessSuite.default()
    image = _forensic_image(seed=1)
    jpeg_q75 = next(level for level in suite.levels if level.name == "jpeg_q75")

    out = suite.apply(jpeg_q75, image)

    assert out.format == "JPEG"
    assert out.rgb.size == image.rgb.size


def test_jpeg_perturbation_strips_exif() -> None:
    exif_bytes = piexif.dump({"0th": {piexif.ImageIFD.Make: b"TestCam"}})
    buffer = io.BytesIO()
    natural_like_image(size=_SIZE, seed=1).save(buffer, format="JPEG", exif=exif_bytes)
    image = ForensicImage.from_bytes(buffer.getvalue())

    suite = RobustnessSuite.default()
    jpeg_q85 = next(level for level in suite.levels if level.name == "jpeg_q85")
    out = suite.apply(jpeg_q85, image)

    with Image.open(io.BytesIO(out.raw)) as reopened:
        assert not bool(reopened.getexif())


def test_webp_perturbation_format_and_size() -> None:
    suite = RobustnessSuite.default()
    image = _forensic_image(seed=1)
    webp = next(level for level in suite.levels if level.name == "webp_q80")

    out = suite.apply(webp, image)

    assert out.format == "WEBP"
    assert out.rgb.size == image.rgb.size


@pytest.mark.parametrize(
    ("level_name", "scale"),
    [("resize_0.75", 0.75), ("resize_0.5", 0.5), ("resize_0.25", 0.25)],
)
def test_resize_perturbation_scales_and_is_png(level_name: str, scale: float) -> None:
    suite = RobustnessSuite.default()
    image = _forensic_image(seed=1)
    level = next(level for level in suite.levels if level.name == level_name)

    out = suite.apply(level, image)

    expected = (round(_SIZE[0] * scale), round(_SIZE[1] * scale))
    assert out.rgb.size == expected
    assert out.format == "PNG"


def test_resize_roundtrip_restores_original_size() -> None:
    suite = RobustnessSuite.default()
    image = _forensic_image(seed=1)
    roundtrip = next(level for level in suite.levels if level.name == "roundtrip_0.5")

    out = suite.apply(roundtrip, image)

    assert out.rgb.size == image.rgb.size
    assert out.format == "PNG"


def test_center_crop_fraction_of_each_side() -> None:
    suite = RobustnessSuite.default()
    image = _forensic_image(seed=1)
    crop = next(level for level in suite.levels if level.name == "crop_0.8")

    out = suite.apply(crop, image)

    expected = (round(_SIZE[0] * 0.8), round(_SIZE[1] * 0.8))
    assert out.rgb.size == expected
    assert out.format == "PNG"


def test_gaussian_noise_is_png_and_clipped() -> None:
    suite = RobustnessSuite.default()
    image = _forensic_image(seed=1)
    noise_5 = next(level for level in suite.levels if level.name == "noise_5")

    out = suite.apply(noise_5, image)

    assert out.format == "PNG"
    assert out.rgb.size == image.rgb.size
    array = np.asarray(out.rgb)
    assert array.min() >= 0 and array.max() <= 255


def test_social_resizes_large_image_and_strips_metadata() -> None:
    exif_bytes = piexif.dump({"0th": {piexif.ImageIFD.Make: b"TestCam"}})
    large = natural_like_image(size=(1600, 800), seed=3)
    buffer = io.BytesIO()
    large.save(buffer, format="JPEG", exif=exif_bytes, quality=95)
    image = ForensicImage.from_bytes(buffer.getvalue())

    suite = RobustnessSuite.default()
    social = next(level for level in suite.levels if level.name == "social_1080_q80")
    out = suite.apply(social, image)

    assert out.format == "JPEG"
    assert max(out.rgb.size) <= 1080
    with Image.open(io.BytesIO(out.raw)) as reopened:
        assert not bool(reopened.getexif())


def test_social_does_not_upscale_small_image() -> None:
    image = _forensic_image(seed=1)
    suite = RobustnessSuite.default()
    social = next(level for level in suite.levels if level.name == "social_1080_q80")

    out = suite.apply(social, image)

    assert out.rgb.size == image.rgb.size


def test_from_yaml_round_trips_a_custom_suite(tmp_path: Path) -> None:
    yaml_text = """
    version: 2
    levels:
      - name: clean
        kind: clean
        params: {}
      - name: jpeg_q40
        kind: jpeg
        params: {quality: 40}
    """
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml_text, encoding="utf-8")

    suite = RobustnessSuite.from_yaml(suite_path)

    assert suite.version == 2
    assert [level.name for level in suite.levels] == ["clean", "jpeg_q40"]
    assert suite.levels[1].params == {"quality": 40}


def test_perturbation_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        Perturbation(name="bogus", kind="not-a-kind", params={})  # type: ignore[arg-type]
