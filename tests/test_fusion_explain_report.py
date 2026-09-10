"""Tests for imgforensics.fusion.explain_report: the explanation-report builder.

Signals only (`metadata`, `ela`) and a hand-built or CLI-fitted fuser -- no
learned detector, no torch -- per the same "signals only" scope as
`tests/test_fusion_cli.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from conftest import natural_like_image, synthetic_fusion_records
from PIL import Image
from typer.testing import CliRunner

from imgforensics.cli import app
from imgforensics.core.image import ForensicImage
from imgforensics.core.types import DetectionResult
from imgforensics.eval.runner import BenchmarkResult
from imgforensics.fusion.explain_report import (
    OVERLAY_MAX_SIDE,
    ReportPaths,
    apply_colormap,
    blend_overlay,
    build_report,
)
from imgforensics.fusion.stacking import Band, FitInfo, Fuser, FuserMetrics

runner = CliRunner()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _manual_fuser(detectors: list[str]) -> Fuser:
    """A hand-built fuser (bypassing fit_fuser), matching test_fusion_report.py's helper."""
    n = len(detectors)
    return Fuser(
        detectors=tuple(detectors),
        logit_weights=np.linspace(1.0, float(n), n),
        presence_weights=np.zeros(n),
        bias=0.0,
        temperature=1.0,
        band=Band(low=0.35, high=0.65),
        fit_info=FitInfo(n_images=10, n_fake=5, n_real=5, sources=["s"], levels=["clean"]),
        metrics=FuserMetrics(
            train_auc=0.9,
            holdout_auc=0.9,
            ece_before=0.1,
            ece_after=0.05,
            abstain_rate=0.1,
            outside_band_balanced_accuracy=0.9,
        ),
    )


def _gradient_heatmap(width: int, height: int) -> np.ndarray:
    """A deterministic float32 HxW heatmap in [0, 1], varying left-to-right."""
    return np.tile(np.linspace(0.0, 1.0, width, dtype=np.float32), (height, 1))


def _save_records(records, tmp_path: Path, name: str = "records.json") -> Path:
    result = BenchmarkResult(
        manifest_name="synthetic",
        entry_count=len({r.entry_path for r in records}),
        levels=sorted({r.level for r in records}),
        fixed_threshold=0.5,
        tuned_thresholds={},
        threshold_caveat=None,
        records=records,
    )
    path = tmp_path / name
    result.save_json(path)
    return path


# --------------------------------------------------------------------------
# Colormap / blend helpers
# --------------------------------------------------------------------------


def test_apply_colormap_shape_and_dtype() -> None:
    heatmap = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    colors = apply_colormap(heatmap)
    assert colors.shape == (4, 4, 3)
    assert colors.dtype == np.uint8


def test_apply_colormap_endpoints_are_blue_and_red() -> None:
    heatmap = np.array([[0.0, 1.0]], dtype=np.float32)
    colors = apply_colormap(heatmap)
    assert tuple(colors[0, 0]) == (0, 0, 255)  # cold -> blue
    assert tuple(colors[0, 1]) == (255, 0, 0)  # hot -> red


def test_apply_colormap_clips_out_of_range_values() -> None:
    heatmap = np.array([[-1.0, 2.0]], dtype=np.float32)
    colors = apply_colormap(heatmap)
    assert tuple(colors[0, 0]) == (0, 0, 255)
    assert tuple(colors[0, 1]) == (255, 0, 0)


def test_blend_overlay_shape_and_dtype() -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    heatmap = np.full((8, 8), 0.8, dtype=np.float32)
    blended = blend_overlay(rgb, heatmap)
    assert blended.shape == (8, 8, 3)
    assert blended.dtype == np.uint8


def test_blend_overlay_zero_heatmap_leaves_pixels_unchanged() -> None:
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(10, 10, 3), dtype=np.uint8)
    heatmap = np.zeros((10, 10), dtype=np.float32)
    blended = blend_overlay(rgb, heatmap)
    assert np.array_equal(blended, rgb)


def test_blend_overlay_shape_mismatch_raises() -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    heatmap = np.zeros((4, 4), dtype=np.float32)
    try:
        blend_overlay(rgb, heatmap)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError on shape mismatch")


# --------------------------------------------------------------------------
# build_report: JSON structure
# --------------------------------------------------------------------------


def _metadata_result(**overrides: object) -> DetectionResult:
    defaults: dict[str, object] = dict(
        detector="metadata",
        score=0.3,
        label="real",
        elapsed_ms=1.2,
        details={"camera_make": "TestCam", "editor_markers": [], "ai_markers": []},
    )
    defaults.update(overrides)
    return DetectionResult(**defaults)  # type: ignore[arg-type]


def _ela_result(image: ForensicImage, **overrides: object) -> DetectionResult:
    heatmap = _gradient_heatmap(image.width, image.height)
    defaults: dict[str, object] = dict(
        detector="ela",
        score=0.55,
        label="uncertain",
        elapsed_ms=3.4,
        heatmap=heatmap,
        details={"mean_ela": 0.1, "p99_ela": 0.9, "high_region_fraction": 0.05},
    )
    defaults.update(overrides)
    return DetectionResult(**defaults)  # type: ignore[arg-type]


def test_build_report_json_keys_present_and_serialisable(tmp_path: Path) -> None:
    image = ForensicImage.from_pil(natural_like_image(size=(64, 64)))
    results = [_metadata_result(), _ela_result(image)]

    paths = build_report(
        image, results, fuser=None, out_dir=tmp_path / "report", source_name="test.jpg"
    )
    assert isinstance(paths, ReportPaths)
    assert paths.json_path.is_file()

    document = json.loads(paths.json_path.read_text(encoding="utf-8"))
    # round-trips through json.dumps again without raising -> fully JSON-safe
    json.dumps(document)

    assert set(document.keys()) >= {"image", "detectors", "package_version", "created"}
    assert set(document["image"].keys()) == {"name", "width", "height", "format", "sha256"}
    assert document["image"]["name"] == "test.jpg"
    assert document["image"]["width"] == 64
    assert document["image"]["height"] == 64

    by_detector = {entry["detector"]: entry for entry in document["detectors"]}
    assert set(by_detector) == {"metadata", "ela"}
    for entry in by_detector.values():
        assert set(entry.keys()) >= {
            "detector",
            "score",
            "label",
            "elapsed_ms",
            "details",
            "heatmap",
            "overlay",
        }
    assert by_detector["metadata"]["heatmap"] is None
    assert by_detector["metadata"]["overlay"] is None
    assert by_detector["ela"]["heatmap"] == "ela_heatmap.png"
    assert by_detector["ela"]["overlay"]["file"] == "ela_overlay.png"
    assert "fusion" not in document


def test_build_report_no_fuser_omits_fusion_key_and_card(tmp_path: Path) -> None:
    image = ForensicImage.from_pil(natural_like_image(size=(48, 48)))
    results = [_metadata_result()]

    paths = build_report(
        image, results, fuser=None, out_dir=tmp_path / "report", source_name="x.jpg"
    )
    document = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert "fusion" not in document

    markdown = paths.markdown_path.read_text(encoding="utf-8")
    assert "Fused verdict" not in markdown
    assert "metadata" in markdown


def test_build_report_with_fuser_includes_fusion_and_card(tmp_path: Path) -> None:
    image = ForensicImage.from_pil(natural_like_image(size=(48, 48)))
    results = [_metadata_result(), _ela_result(image)]
    fuser = _manual_fuser(["metadata", "ela"])

    paths = build_report(
        image, results, fuser=fuser, out_dir=tmp_path / "report", source_name="x.jpg"
    )
    document = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert set(document["fusion"].keys()) == {"probability", "label", "band", "contributions"}
    assert {c["detector"] for c in document["fusion"]["contributions"]} == {"metadata", "ela"}

    markdown = paths.markdown_path.read_text(encoding="utf-8")
    assert "## Fused verdict" in markdown
    assert "abstain band" in markdown


# --------------------------------------------------------------------------
# build_report: heatmap / overlay PNGs
# --------------------------------------------------------------------------


def test_build_report_overlay_and_heatmap_files_for_ela(tmp_path: Path) -> None:
    image = ForensicImage.from_pil(natural_like_image(size=(96, 64)))
    results = [_metadata_result(), _ela_result(image)]

    out_dir = tmp_path / "report"
    build_report(image, results, fuser=None, out_dir=out_dir, source_name="x.jpg")

    heatmap_path = out_dir / "ela_heatmap.png"
    overlay_path = out_dir / "ela_overlay.png"
    assert heatmap_path.is_file()
    assert overlay_path.is_file()

    with Image.open(heatmap_path) as heatmap_img:
        assert heatmap_img.mode == "L"
        assert heatmap_img.size == (96, 64)  # full resolution, no downscale

    with Image.open(overlay_path) as overlay_img:
        assert overlay_img.mode == "RGB"
        assert overlay_img.size == (96, 64)  # small image: no downscale needed

    # metadata has no heatmap: no files written for it
    assert not (out_dir / "metadata_heatmap.png").exists()
    assert not (out_dir / "metadata_overlay.png").exists()


def test_build_report_downscales_large_overlay_but_keeps_full_size_heatmap(
    tmp_path: Path,
) -> None:
    width, height = 1600, 1000
    image = ForensicImage.from_pil(Image.new("RGB", (width, height), color=(120, 130, 140)))
    heatmap = _gradient_heatmap(width, height)
    result = DetectionResult(
        detector="ela", score=0.5, label="uncertain", heatmap=heatmap, details={}
    )

    out_dir = tmp_path / "report"
    paths = build_report(image, [result], fuser=None, out_dir=out_dir, source_name="big.jpg")

    with Image.open(paths.heatmap_paths["ela"]) as heatmap_img:
        assert heatmap_img.size == (width, height)

    with Image.open(paths.overlay_paths["ela"]) as overlay_img:
        assert max(overlay_img.size) == OVERLAY_MAX_SIDE
        assert overlay_img.size[0] <= OVERLAY_MAX_SIDE
        assert overlay_img.size[1] <= OVERLAY_MAX_SIDE

    document = json.loads(paths.json_path.read_text(encoding="utf-8"))
    overlay_info = next(d for d in document["detectors"] if d["detector"] == "ela")["overlay"]
    assert overlay_info["downscaled"] is True
    assert max(overlay_info["width"], overlay_info["height"]) == OVERLAY_MAX_SIDE
    assert document["overlay_max_side"] == OVERLAY_MAX_SIDE


# --------------------------------------------------------------------------
# build_report: abstention caution
# --------------------------------------------------------------------------


def test_build_report_shows_caution_for_abstaining_detector(tmp_path: Path) -> None:
    image = ForensicImage.from_pil(natural_like_image(size=(48, 48)))
    result = DetectionResult(
        detector="metadata",
        score=0.5,
        label="uncertain",
        details={"reason": "no encoded file available"},
    )

    paths = build_report(
        image, [result], fuser=None, out_dir=tmp_path / "report", source_name="x.jpg"
    )
    markdown = paths.markdown_path.read_text(encoding="utf-8")
    assert "Caution" in markdown
    assert "no encoded file available" in markdown


def test_build_report_no_caution_for_a_confident_detector(tmp_path: Path) -> None:
    image = ForensicImage.from_pil(natural_like_image(size=(48, 48)))
    result = _metadata_result(score=0.3, label="real")

    paths = build_report(
        image, [result], fuser=None, out_dir=tmp_path / "report", source_name="x.jpg"
    )
    markdown = paths.markdown_path.read_text(encoding="utf-8")
    assert "Caution" not in markdown


# --------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------


def test_cli_analyze_report_dir_without_fuser(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    natural_like_image().save(image_path, format="JPEG", quality=90)
    report_dir = tmp_path / "report"

    result = runner.invoke(
        app,
        [
            "analyze",
            str(image_path),
            "--detector",
            "metadata",
            "--detector",
            "ela",
            "--report-dir",
            str(report_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout
    # Rich wraps long lines to the console width, so check the pieces rather
    # than the exact "Report written to <path>" substring.
    assert "Report written to" in result.stdout
    assert (report_dir / "report.json").is_file()
    assert (report_dir / "report.md").is_file()
    assert (report_dir / "ela_heatmap.png").is_file()
    assert (report_dir / "ela_overlay.png").is_file()

    markdown = (report_dir / "report.md").read_text(encoding="utf-8")
    assert "Fused verdict" not in markdown

    document = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert "fusion" not in document


def test_cli_analyze_report_dir_with_fuser(tmp_path: Path) -> None:
    detector_scores = {
        "metadata": lambda y, rng: float(
            np.clip(y * 0.9 + 0.05 + rng.normal(0, 0.05), 1e-3, 1 - 1e-3)
        ),
        "ela": lambda y, rng: float(
            np.clip((1 - y) * 0.9 + 0.05 + rng.normal(0, 0.05), 1e-3, 1 - 1e-3)
        ),
    }
    records = synthetic_fusion_records(detector_scores, n_per_class=150, seed=0)
    records_path = _save_records(records, tmp_path)
    fuser_path = tmp_path / "fuser.json"

    fit_result = runner.invoke(app, ["fusion", "fit", str(records_path), "--out", str(fuser_path)])
    assert fit_result.exit_code == 0, fit_result.stdout

    image_path = tmp_path / "image.jpg"
    natural_like_image().save(image_path, format="JPEG", quality=90)
    report_dir = tmp_path / "report"

    result = runner.invoke(
        app,
        [
            "analyze",
            str(image_path),
            "--fuser",
            str(fuser_path),
            "--detector",
            "metadata",
            "--detector",
            "ela",
            "--report-dir",
            str(report_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout

    markdown = (report_dir / "report.md").read_text(encoding="utf-8")
    assert "## Fused verdict" in markdown

    document = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert "fusion" in document
    assert document["fusion"]["label"] in ("real", "fake", "uncertain")


def test_cli_analyze_report_dir_json_output_includes_report_dir(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    natural_like_image().save(image_path, format="JPEG", quality=90)
    report_dir = tmp_path / "report"

    result = runner.invoke(
        app,
        [
            "analyze",
            str(image_path),
            "--detector",
            "metadata",
            "--json",
            "--report-dir",
            str(report_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout
    document = json.loads(result.stdout)
    assert document["report_dir"] == str(report_dir)
    assert (report_dir / "report.json").is_file()


def test_cli_analyze_without_report_dir_writes_nothing(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    natural_like_image().save(image_path, format="JPEG", quality=90)

    result = runner.invoke(app, ["analyze", str(image_path), "--detector", "metadata"])
    assert result.exit_code == 0, result.stdout
    assert "Report written to" not in result.stdout
