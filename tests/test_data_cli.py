"""CLI tests for the `datasets`, `manifest build`, and `audit` commands."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from imgforensics.cli import app
from imgforensics.data.manifest import Manifest

runner = CliRunner()


def test_cli_datasets_list() -> None:
    result = runner.invoke(app, ["datasets", "list"])
    assert result.exit_code == 0
    assert "TGIF" in result.stdout
    assert "Community Forensics" in result.stdout


def test_cli_datasets_show() -> None:
    result = runner.invoke(app, ["datasets", "show", "TGIF"])
    assert result.exit_code == 0
    assert "localization" in result.stdout
    assert "CC BY-SA 4.0" in result.stdout


def test_cli_datasets_show_unknown_name_errors() -> None:
    result = runner.invoke(app, ["datasets", "show", "does-not-exist"])
    assert result.exit_code != 0


def _make_tree(root: Path) -> None:
    (root / "real").mkdir(parents=True)
    (root / "fake").mkdir(parents=True)
    for i in range(4):
        Image.new("RGB", (48, 32)).save(root / "real" / f"r{i}.jpg", format="JPEG", quality=85)
    for i in range(4):
        Image.new("RGB", (48, 32)).save(root / "fake" / f"f{i}.jpg", format="JPEG", quality=85)


def test_cli_manifest_build_and_load(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    out_path = tmp_path / "manifest.jsonl"

    result = runner.invoke(
        app,
        [
            "manifest",
            "build",
            str(tmp_path),
            "--dataset",
            "cli-test",
            "--out",
            str(out_path),
            "--license",
            "MIT",
            "--commercial-ok",
        ],
    )

    assert result.exit_code == 0
    assert out_path.exists()
    manifest = Manifest.load(out_path)
    assert len(manifest.entries) == 8
    assert manifest.meta.dataset == "cli-test"
    assert manifest.meta.commercial_ok is True


def test_cli_manifest_build_no_commercial_ok_flag(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    out_path = tmp_path / "manifest.jsonl"

    result = runner.invoke(
        app,
        ["manifest", "build", str(tmp_path), "--dataset", "cli-test", "--out", str(out_path)],
    )

    assert result.exit_code == 0
    manifest = Manifest.load(out_path)
    assert manifest.meta.commercial_ok is None


def test_cli_audit_passes_on_balanced_manifest(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    manifest_path = tmp_path / "manifest.jsonl"
    runner.invoke(
        app,
        ["manifest", "build", str(tmp_path), "--dataset", "cli-test", "--out", str(manifest_path)],
    )

    result = runner.invoke(app, ["audit", str(manifest_path)])
    assert result.exit_code == 0
    assert "Bias audit report" in result.stdout


def test_cli_audit_strict_exits_nonzero_on_bias(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "fake").mkdir()
    for i in range(10):
        Image.new("RGB", (48, 32)).save(tmp_path / "real" / f"r{i}.jpg", format="JPEG", quality=85)
    for i in range(10):
        Image.new("RGB", (48, 32)).save(tmp_path / "fake" / f"f{i}.png", format="PNG")

    manifest_path = tmp_path / "manifest.jsonl"
    runner.invoke(
        app,
        ["manifest", "build", str(tmp_path), "--dataset", "cli-test", "--out", str(manifest_path)],
    )

    result = runner.invoke(app, ["audit", str(manifest_path), "--strict"])
    assert result.exit_code == 1

    lenient_result = runner.invoke(app, ["audit", str(manifest_path)])
    assert lenient_result.exit_code == 0
