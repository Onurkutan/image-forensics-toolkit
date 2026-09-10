"""CLI test for `imgforensics benchmark` (imgforensics.eval.runner via the CLI)."""

from __future__ import annotations

from pathlib import Path

from test_eval_runner import _build_manifest
from typer.testing import CliRunner

from imgforensics.cli import app

runner = CliRunner()


def test_cli_benchmark_runs_end_to_end(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest.save(manifest_path)

    out_json = tmp_path / "results.json"
    report_path = tmp_path / "report.md"

    result = runner.invoke(
        app,
        [
            "benchmark",
            str(manifest_path),
            "--robustness",
            "none",
            "--limit",
            "4",
            "--detector",
            "metadata",
            "--baselines",
            "--out",
            str(out_json),
            "--report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert out_json.exists()
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Benchmark report" in report_text
    assert "metadata" in report_text
    assert "constant_real" in report_text


def test_cli_benchmark_prints_report_to_stdout_without_report_flag(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest.save(manifest_path)

    result = runner.invoke(
        app,
        [
            "benchmark",
            str(manifest_path),
            "--robustness",
            "none",
            "--limit",
            "4",
            "--detector",
            "metadata",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Benchmark report" in result.stdout


def test_cli_benchmark_rejects_unknown_detector(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest.save(manifest_path)

    result = runner.invoke(app, ["benchmark", str(manifest_path), "--detector", "does-not-exist"])

    assert result.exit_code != 0


def test_cli_benchmark_requires_at_least_one_detector(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest.save(manifest_path)

    result = runner.invoke(app, ["benchmark", str(manifest_path)])

    assert result.exit_code != 0


def test_cli_benchmark_runs_with_workers(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest.save(manifest_path)

    result = runner.invoke(
        app,
        [
            "benchmark",
            str(manifest_path),
            "--robustness",
            "none",
            "--limit",
            "4",
            "--all-signals",
            "--workers",
            "2",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Benchmark report" in result.stdout
