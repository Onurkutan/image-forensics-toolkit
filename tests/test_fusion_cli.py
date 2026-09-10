"""CLI tests for `imgforensics fusion fit`/`fusion info` and `analyze --fuser`.

The `analyze --fuser` test uses only classical signals (`metadata`, `ela`) --
no learned detector, no torch -- per the "signals only" scope for this
slice, and fits its own fuser on synthetic records naming those two
detectors, so no real dataset or manifest is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from conftest import natural_like_image, synthetic_fusion_records
from typer.testing import CliRunner

from imgforensics.cli import app
from imgforensics.eval.runner import BenchmarkResult

runner = CliRunner()


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


def test_cli_fusion_fit_and_info(tmp_path: Path) -> None:
    records = synthetic_fusion_records(n_per_class=200, seed=0)
    records_path = _save_records(records, tmp_path)
    fuser_path = tmp_path / "fuser.json"

    fit_result = runner.invoke(app, ["fusion", "fit", str(records_path), "--out", str(fuser_path)])
    assert fit_result.exit_code == 0, fit_result.stdout
    assert fuser_path.is_file()
    assert "held-out AUC" in fit_result.stdout
    assert "informative" in fit_result.stdout

    fuser_data = json.loads(fuser_path.read_text(encoding="utf-8"))
    assert set(fuser_data["detectors"]) == {"informative", "inverted", "abstaining"}
    assert str(records_path) in fuser_data["records_sha256"]

    info_result = runner.invoke(app, ["fusion", "info", str(fuser_path)])
    assert info_result.exit_code == 0, info_result.stdout
    assert "informative" in info_result.stdout
    assert "held-out AUC" in info_result.stdout


def test_cli_fusion_fit_respects_level_and_detector_filters(tmp_path: Path) -> None:
    records = synthetic_fusion_records(n_per_class=200, seed=0)
    records_path = _save_records(records, tmp_path)
    fuser_path = tmp_path / "fuser.json"

    fit_result = runner.invoke(
        app,
        [
            "fusion",
            "fit",
            str(records_path),
            "--out",
            str(fuser_path),
            "--level",
            "clean",
            "--detector",
            "informative",
            "--detector",
            "inverted",
        ],
    )
    assert fit_result.exit_code == 0, fit_result.stdout
    fuser_data = json.loads(fuser_path.read_text(encoding="utf-8"))
    assert set(fuser_data["detectors"]) == {"informative", "inverted"}
    assert fuser_data["fit"]["levels"] == ["clean"]


def test_cli_fusion_fit_multiple_record_files(tmp_path: Path) -> None:
    records_a = synthetic_fusion_records(n_per_class=100, seed=0)
    records_b = synthetic_fusion_records(n_per_class=100, seed=1)
    for record in records_b:  # keep entry paths distinct across the two files
        record.entry_path = "b_" + record.entry_path

    path_a = _save_records(records_a, tmp_path, "records_a.json")
    path_b = _save_records(records_b, tmp_path, "records_b.json")
    fuser_path = tmp_path / "fuser.json"

    fit_result = runner.invoke(
        app, ["fusion", "fit", str(path_a), str(path_b), "--out", str(fuser_path)]
    )
    assert fit_result.exit_code == 0, fit_result.stdout
    fuser_data = json.loads(fuser_path.read_text(encoding="utf-8"))
    assert fuser_data["fit"]["n_images"] == 400
    assert set(fuser_data["records_sha256"].keys()) == {str(path_a), str(path_b)}


def test_cli_fusion_fit_rejects_unknown_detector(tmp_path: Path) -> None:
    records = synthetic_fusion_records(n_per_class=50, seed=0)
    records_path = _save_records(records, tmp_path)
    fuser_path = tmp_path / "fuser.json"

    result = runner.invoke(
        app,
        [
            "fusion",
            "fit",
            str(records_path),
            "--out",
            str(fuser_path),
            "--detector",
            "not-a-real-detector",
        ],
    )
    assert result.exit_code != 0


def test_cli_analyze_with_fuser_prints_panel_and_json(tmp_path: Path) -> None:
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

    text_result = runner.invoke(
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
        ],
    )
    assert text_result.exit_code == 0, text_result.stdout
    assert "Fused verdict" in text_result.stdout

    json_result = runner.invoke(
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
            "--json",
        ],
    )
    assert json_result.exit_code == 0, json_result.stdout
    document = json.loads(json_result.stdout)
    assert "fusion" in document
    fusion = document["fusion"]
    assert set(fusion.keys()) == {"probability", "label", "band", "contributions"}
    assert fusion["label"] in ("real", "fake", "uncertain")
    assert 0.0 <= fusion["probability"] <= 1.0
    assert {c["detector"] for c in fusion["contributions"]} == {"metadata", "ela"}


def test_cli_analyze_without_fuser_option_is_unchanged(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    natural_like_image().save(image_path, format="JPEG", quality=90)

    result = runner.invoke(app, ["analyze", str(image_path), "--detector", "metadata"])
    assert result.exit_code == 0, result.stdout
    assert "Fused verdict" not in result.stdout

    json_result = runner.invoke(
        app, ["analyze", str(image_path), "--detector", "metadata", "--json"]
    )
    assert json_result.exit_code == 0, json_result.stdout
    document = json.loads(json_result.stdout)
    assert "fusion" not in document


def test_cli_fusion_eval_writes_report_and_json(tmp_path: Path) -> None:
    records = synthetic_fusion_records(n_per_class=100, seed=0)
    records_path = _save_records(records, tmp_path)
    fuser_path = tmp_path / "fuser.json"

    fit_result = runner.invoke(app, ["fusion", "fit", str(records_path), "--out", str(fuser_path)])
    assert fit_result.exit_code == 0, fit_result.stdout

    report_path = tmp_path / "eval.md"
    json_path = tmp_path / "eval.json"
    eval_result = runner.invoke(
        app,
        [
            "fusion",
            "eval",
            str(records_path),
            "--fuser",
            str(fuser_path),
            "--report",
            str(report_path),
            "--json",
            str(json_path),
        ],
    )
    assert eval_result.exit_code == 0, eval_result.stdout
    assert report_path.is_file()
    report_text = report_path.read_text(encoding="utf-8")
    assert "## Level: clean" in report_text
    assert "fused_outside_band" in report_text

    assert json_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {"detectors", "levels", "rows"}
    assert set(payload["detectors"]) == {"informative", "inverted", "abstaining"}


def test_cli_fusion_eval_rejects_unknown_level(tmp_path: Path) -> None:
    records = synthetic_fusion_records(n_per_class=50, seed=0)
    records_path = _save_records(records, tmp_path)
    fuser_path = tmp_path / "fuser.json"

    fit_result = runner.invoke(app, ["fusion", "fit", str(records_path), "--out", str(fuser_path)])
    assert fit_result.exit_code == 0, fit_result.stdout

    result = runner.invoke(
        app,
        [
            "fusion",
            "eval",
            str(records_path),
            "--fuser",
            str(fuser_path),
            "--level",
            "does-not-exist",
        ],
    )
    assert result.exit_code != 0


def test_cli_fusion_eval_rejects_missing_fuser_file(tmp_path: Path) -> None:
    records = synthetic_fusion_records(n_per_class=50, seed=0)
    records_path = _save_records(records, tmp_path)

    result = runner.invoke(
        app,
        [
            "fusion",
            "eval",
            str(records_path),
            "--fuser",
            str(tmp_path / "does-not-exist.json"),
        ],
    )
    assert result.exit_code != 0


def test_cli_analyze_rejects_missing_fuser_file(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    natural_like_image().save(image_path, format="JPEG", quality=90)

    result = runner.invoke(
        app,
        [
            "analyze",
            str(image_path),
            "--fuser",
            str(tmp_path / "does-not-exist.json"),
            "--detector",
            "metadata",
        ],
    )
    assert result.exit_code != 0
