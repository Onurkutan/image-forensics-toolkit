"""Tests for imgforensics.views and the way the CLI treats a tool that has no verdict."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from imgforensics.cli import app
from imgforensics.core import registry
from imgforensics.core.base import BaseDetector
from imgforensics.core.image import ForensicImage
from imgforensics.core.parameters import coerce_parameters
from imgforensics.data.manifest import Manifest, ManifestMeta
from imgforensics.views import VIEW_NAMES
from imgforensics.views.bitplane import BitPlanesView
from imgforensics.views.luminance import LuminanceGradientView
from imgforensics.views.noise import NoiseResidualView

runner = CliRunner()

VIEW_CLASSES: tuple[type[BaseDetector], ...] = (
    BitPlanesView,
    LuminanceGradientView,
    NoiseResidualView,
)


def _image(size: tuple[int, int] = (7, 5), seed: int = 0) -> ForensicImage:
    """A ``ForensicImage`` of random pixels at ``(width, height)``."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return ForensicImage.from_pil(Image.fromarray(pixels, mode="RGB"))


def _solid(value: int, size: tuple[int, int] = (9, 9)) -> ForensicImage:
    return ForensicImage.from_pil(Image.new("RGB", size, color=(value, value, value)))


def _jpeg_bytes(size: tuple[int, int] = (48, 32)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(80, 120, 200)).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _empty_manifest(tmp_path: Path) -> Path:
    """A manifest with no entries -- enough for a check that runs before any image is read."""
    manifest = Manifest(
        meta=ManifestMeta(dataset="views", root=str(tmp_path), created="2026-01-01T00:00:00Z")
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest.save(manifest_path)
    return manifest_path


def test_every_view_is_registered_under_its_name_and_declares_the_view_kind() -> None:
    assert set(VIEW_NAMES) == {view.name for view in VIEW_CLASSES}
    for name in VIEW_NAMES:
        assert registry.get(name).kind == "view"


@pytest.mark.parametrize("view_cls", VIEW_CLASSES)
@pytest.mark.parametrize("size", [(7, 5), (1, 1), (64, 40)])
def test_a_view_returns_an_image_shaped_map_in_range(
    view_cls: type[BaseDetector], size: tuple[int, int]
) -> None:
    image = _image(size)

    result = view_cls().run(image)

    assert result.heatmap is not None
    assert result.heatmap.shape == (size[1], size[0])
    assert result.heatmap.dtype == np.float32
    assert result.heatmap.min() >= 0.0
    assert result.heatmap.max() <= 1.0


@pytest.mark.parametrize("view_cls", VIEW_CLASSES)
def test_a_view_claims_no_verdict(view_cls: type[BaseDetector]) -> None:
    result = view_cls().run(_image())

    assert result.score == pytest.approx(0.5)
    assert result.label == "uncertain"
    assert result.details["kind"] == "view"


@pytest.mark.parametrize("view_cls", VIEW_CLASSES)
def test_a_view_is_deterministic(view_cls: type[BaseDetector]) -> None:
    image = _image((32, 24), seed=3)

    first = view_cls().run(image).heatmap
    second = view_cls().run(image).heatmap

    assert first is not None and second is not None
    assert np.array_equal(first, second)


@pytest.mark.parametrize("view_cls", VIEW_CLASSES)
def test_declared_parameters_match_the_constructor_defaults(view_cls: type[BaseDetector]) -> None:
    instance = view_cls()

    specs = view_cls.parameters()
    assert specs, "each view declares at least one parameter"
    for spec in specs:
        assert getattr(instance, spec.name) == spec.default


@pytest.mark.parametrize("view_cls", VIEW_CLASSES)
def test_declared_parameters_are_accepted_as_keyword_arguments(
    view_cls: type[BaseDetector],
) -> None:
    coerced = coerce_parameters(view_cls.parameters(), {})

    result = view_cls(**coerced).run(_image())

    assert result.heatmap is not None


def test_luminance_gradient_reports_its_scale_and_a_blur_radius() -> None:
    result = LuminanceGradientView(radius=2).run(_image((24, 16)))

    assert result.details["radius"] == 2
    assert result.details["p99"] > 0.0
    assert 0.0 <= result.details["mean"] <= 1.0


def test_luminance_gradient_blur_changes_the_map() -> None:
    image = _image((32, 32), seed=7)

    sharp = LuminanceGradientView(radius=1).run(image).heatmap
    blurred = LuminanceGradientView(radius=3).run(image).heatmap

    assert sharp is not None and blurred is not None
    assert not np.array_equal(sharp, blurred)


def test_luminance_gradient_of_a_flat_image_is_all_zero() -> None:
    result = LuminanceGradientView().run(_solid(120))

    assert result.heatmap is not None
    assert np.count_nonzero(result.heatmap) == 0


def test_noise_residual_reports_the_window_it_used() -> None:
    result = NoiseResidualView(window="5").run(_image((24, 16)))

    assert result.details["window"] == 5


def test_noise_residual_window_changes_the_map() -> None:
    image = _image((32, 32), seed=11)

    small = NoiseResidualView(window="3").run(image).heatmap
    large = NoiseResidualView(window="5").run(image).heatmap

    assert small is not None and large is not None
    assert not np.array_equal(small, large)


def test_noise_residual_of_a_flat_image_is_all_zero() -> None:
    result = NoiseResidualView().run(_solid(200))

    assert result.heatmap is not None
    assert np.count_nonzero(result.heatmap) == 0


@pytest.mark.parametrize(("plane", "expected"), [(0, 0.0), (7, 1.0)])
def test_bit_planes_reads_the_selected_bit(plane: int, expected: float) -> None:
    # 128 is 0b1000_0000: only the top bit is set.
    result = BitPlanesView(plane=plane).run(_solid(128))

    assert result.heatmap is not None
    assert np.all(result.heatmap == expected)
    assert result.details["plane"] == plane
    assert result.details["fraction_set"] == pytest.approx(expected)


def test_bit_planes_map_holds_only_zeros_and_ones() -> None:
    result = BitPlanesView(plane=2).run(_image((24, 16), seed=5))

    assert result.heatmap is not None
    assert set(np.unique(result.heatmap)) <= {0.0, 1.0}


@pytest.mark.parametrize("value", [8, -1, "top"])
def test_a_view_parameter_outside_its_spec_is_refused_before_the_view_is_built(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="plane"):
        coerce_parameters(BitPlanesView.parameters(), {"plane": value})


def test_a_coerced_view_parameter_reaches_the_constructor() -> None:
    coerced = coerce_parameters(BitPlanesView.parameters(), {"plane": "3"})

    view = BitPlanesView(**coerced)

    assert view.plane == 3
    assert view.run(_image()).details["plane"] == 3


@pytest.fixture
def no_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the weight-gated tools at an empty directory so they abstain instantly.

    The CLI tests below run every registered detector, which on a machine with
    the ``ml`` extra and downloaded weights would mean loading three models
    onto the GPU to learn something about the views.
    """
    monkeypatch.setenv("IMGFORENSICS_WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("IMGFORENSICS_HEAD_DIR", str(tmp_path / "head"))


@pytest.mark.usefixtures("no_weights")
def test_cli_analyze_skips_views_by_default(tmp_path: Path) -> None:
    jpg_path = tmp_path / "sample.jpg"
    jpg_path.write_bytes(_jpeg_bytes())

    result = runner.invoke(app, ["analyze", str(jpg_path), "--json"])

    assert result.exit_code == 0, result.stdout
    names = {entry["detector"] for entry in json.loads(result.stdout)["results"]}
    assert "ela" in names
    assert not names & set(VIEW_NAMES)


@pytest.mark.usefixtures("no_weights")
def test_cli_analyze_runs_a_view_when_it_is_named(tmp_path: Path) -> None:
    jpg_path = tmp_path / "sample.jpg"
    jpg_path.write_bytes(_jpeg_bytes())
    out_dir = tmp_path / "heatmaps"

    result = runner.invoke(
        app,
        [
            "analyze",
            str(jpg_path),
            "--json",
            "--detector",
            "luminance_gradient",
            "--save-heatmaps",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout
    entries = json.loads(result.stdout)["results"]
    assert len(entries) == 1
    assert entries[0]["detector"] == "luminance_gradient"
    assert entries[0]["score"] == pytest.approx(0.5)
    assert entries[0]["label"] == "uncertain"
    assert entries[0]["details"]["kind"] == "view"
    assert (out_dir / "sample_luminance_gradient.png").exists()


def test_cli_benchmark_refuses_a_view(tmp_path: Path) -> None:
    manifest_path = _empty_manifest(tmp_path)

    result = runner.invoke(
        app, ["benchmark", str(manifest_path), "--detector", "bit_planes", "--robustness", "none"]
    )

    assert result.exit_code != 0
    assert "views carry no score" in result.output
    assert "bit_planes" in result.output
