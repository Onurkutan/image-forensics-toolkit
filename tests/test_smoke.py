"""Smoke tests for package import, core types, image I/O, registry, and CLI."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError
from typer.testing import CliRunner

import imgforensics
from imgforensics.cli import app
from imgforensics.core import registry
from imgforensics.core.base import BaseDetector
from imgforensics.core.image import ForensicImage
from imgforensics.core.types import DetectionResult, label_from_score
from imgforensics.utils.image_io import image_hash, load_image, to_numpy

runner = CliRunner()


def test_package_import_and_version() -> None:
    assert imgforensics.__version__ == "0.1.0"


def test_detection_result_valid() -> None:
    result = DetectionResult(detector="dummy", score=0.5, label="uncertain")
    assert result.score == pytest.approx(0.5)
    assert result.heatmap is None


def test_detection_result_score_out_of_range() -> None:
    with pytest.raises(ValidationError):
        DetectionResult(detector="dummy", score=1.5, label="fake")


def test_detection_result_bad_heatmap() -> None:
    with pytest.raises(ValidationError):
        DetectionResult(
            detector="dummy",
            score=0.5,
            label="uncertain",
            heatmap=np.zeros((4, 4, 3), dtype=np.float32),
        )
    with pytest.raises(ValidationError):
        DetectionResult(
            detector="dummy",
            score=0.5,
            label="uncertain",
            heatmap=np.full((4, 4), 2.0, dtype=np.float32),
        )


def test_detection_result_valid_heatmap() -> None:
    heatmap = np.zeros((4, 4), dtype=np.float32)
    result = DetectionResult(detector="dummy", score=0.5, label="uncertain", heatmap=heatmap)
    assert result.heatmap is not None
    assert result.heatmap.shape == (4, 4)


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.0, "real"), (0.34, "real"), (0.5, "uncertain"), (0.66, "fake"), (1.0, "fake")],
)
def test_label_from_score_thresholds(score: float, expected: str) -> None:
    assert label_from_score(score) == expected


def _make_png_bytes() -> bytes:
    image = Image.new("RGB", (8, 6), color=(120, 40, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_load_image_from_path(tmp_path: Path) -> None:
    png_path = tmp_path / "sample.png"
    png_path.write_bytes(_make_png_bytes())
    image = load_image(png_path)
    assert image.mode == "RGB"
    assert image.size == (8, 6)


def test_load_image_from_bytes() -> None:
    image = load_image(_make_png_bytes())
    assert image.mode == "RGB"
    assert image.size == (8, 6)


def test_load_image_from_pil_image() -> None:
    source = Image.new("RGB", (5, 5), color="white")
    image = load_image(source)
    assert image.mode == "RGB"
    assert image.size == (5, 5)


def test_to_numpy_and_image_hash() -> None:
    image = load_image(_make_png_bytes())
    array = to_numpy(image)
    assert array.dtype == np.uint8
    assert array.shape == (6, 8, 3)
    assert image_hash(image) == image_hash(load_image(_make_png_bytes()))


class _DummyDetector(BaseDetector):
    name = "dummy"

    def predict(self, image: ForensicImage) -> DetectionResult:
        return DetectionResult(detector=self.name, score=0.1, label="real")


def test_registry_register_get_available() -> None:
    registry.register("dummy")(_DummyDetector)
    assert "dummy" in registry.available()
    assert registry.get("dummy") is _DummyDetector
    with pytest.raises(KeyError):
        registry.get("does-not-exist")


def test_base_detector_run_fills_elapsed_ms() -> None:
    detector = _DummyDetector()
    fi = ForensicImage.from_pil(Image.new("RGB", (4, 4)))

    result = detector.run(fi)

    assert result.elapsed_ms is not None
    assert result.elapsed_ms >= 0.0
    assert result.score == pytest.approx(0.1)


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_cli_analyze(tmp_path: Path) -> None:
    png_path = tmp_path / "sample.png"
    png_path.write_bytes(_make_png_bytes())
    result = runner.invoke(app, ["analyze", str(png_path)])
    assert result.exit_code == 0
    assert "sample.png" in result.stdout
    assert "metadata" in result.stdout
    assert "ela" in result.stdout


def _make_jpeg_bytes(quality: int = 90) -> bytes:
    image = Image.new("RGB", (48, 32), color=(80, 120, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def test_cli_analyze_json(tmp_path: Path) -> None:
    jpg_path = tmp_path / "sample.jpg"
    jpg_path.write_bytes(_make_jpeg_bytes())

    # Pin to exactly the two signals under test: the registry is process-wide
    # global state and another test in this module registers a "dummy"
    # detector, so relying on "all registered detectors" here would make this
    # test's result count depend on test order.
    result = runner.invoke(
        app,
        ["analyze", str(jpg_path), "--json", "--detector", "metadata", "--detector", "ela"],
    )

    assert result.exit_code == 0
    document = json.loads(result.stdout)
    assert document["file"] == str(jpg_path)
    assert document["width"] == 48
    assert document["height"] == 32
    assert len(document["results"]) == 2
    names = {entry["detector"] for entry in document["results"]}
    assert names == {"metadata", "ela"}
    for entry in document["results"]:
        assert 0.0 <= entry["score"] <= 1.0
        assert entry["label"] in {"real", "fake", "uncertain"}
        assert entry["elapsed_ms"] is not None
        assert isinstance(entry["details"], dict)


def test_cli_analyze_save_heatmaps(tmp_path: Path) -> None:
    jpg_path = tmp_path / "sample.jpg"
    jpg_path.write_bytes(_make_jpeg_bytes())
    out_dir = tmp_path / "heatmaps"

    result = runner.invoke(
        app, ["analyze", str(jpg_path), "--json", "--save-heatmaps", str(out_dir)]
    )

    assert result.exit_code == 0
    heatmap_path = out_dir / "sample_ela.png"
    assert heatmap_path.exists()
    document = json.loads(result.stdout)
    ela_entry = next(entry for entry in document["results"] if entry["detector"] == "ela")
    assert ela_entry["heatmap"] == str(heatmap_path)
    metadata_entry = next(entry for entry in document["results"] if entry["detector"] == "metadata")
    assert metadata_entry["heatmap"] is None


def test_cli_analyze_selects_single_detector(tmp_path: Path) -> None:
    jpg_path = tmp_path / "sample.jpg"
    jpg_path.write_bytes(_make_jpeg_bytes())

    result = runner.invoke(app, ["analyze", str(jpg_path), "--json", "--detector", "metadata"])

    assert result.exit_code == 0
    document = json.loads(result.stdout)
    assert len(document["results"]) == 1
    assert document["results"][0]["detector"] == "metadata"


def test_cli_analyze_unknown_detector_errors(tmp_path: Path) -> None:
    jpg_path = tmp_path / "sample.jpg"
    jpg_path.write_bytes(_make_jpeg_bytes())

    result = runner.invoke(app, ["analyze", str(jpg_path), "--detector", "does-not-exist"])

    assert result.exit_code != 0


def test_cli_analyze_missing_path_gives_clean_usage_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "does-not-exist.jpg"

    result = runner.invoke(app, ["analyze", str(missing_path)])

    assert result.exit_code != 0
    assert "Traceback" not in result.output
