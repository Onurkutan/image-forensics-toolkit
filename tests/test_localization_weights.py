"""Tests for the license-gated weights fetcher.

Torch-free by design (so is the module under test), and offline: the two
download helpers borrowed from :mod:`imgforensics.data.acquire` are
monkeypatched to write a known byte string, so nothing here touches the
network or Google Drive.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from imgforensics.cli import app
from imgforensics.data.acquire import ChecksumMismatchError, LicenseNotAcceptedError
from imgforensics.detectors.backbones import WEIGHTS_DIR_ENV
from imgforensics.localization import weights as weights_module
from imgforensics.localization.weights import (
    DEFAULT_WEIGHTS_DIR,
    WEIGHTS,
    WeightSpec,
    fetch_weights,
    get_weight_spec,
    resolve_weights_dir,
    weights_file,
)

runner = CliRunner()

_PAYLOAD = b"not really a checkpoint, but it hashes the same way"
_PAYLOAD_SHA256 = hashlib.sha256(_PAYLOAD).hexdigest()


@pytest.fixture
def fake_spec(monkeypatch: pytest.MonkeyPatch) -> WeightSpec:
    """Register a throwaway ``gdrive`` entry whose download writes ``_PAYLOAD``."""
    spec = WeightSpec(
        name="test_weights",
        model="A stand-in",
        method="gdrive",
        url_or_gdrive_id="test-drive-id",
        filename="stand_in.pth",
        sha256=_PAYLOAD_SHA256,
        size_mb=0.1,
        license="MIT",
        commercial_ok=True,
        source="tests",
    )
    monkeypatch.setitem(WEIGHTS, spec.name, spec)

    def _write(*, file_id: str, output: Path) -> str:
        assert file_id == spec.url_or_gdrive_id
        output.write_bytes(_PAYLOAD)
        return str(output)

    monkeypatch.setattr(weights_module, "_gdown_download", _write)
    return spec


def test_the_iml_vit_entry_records_its_license_and_source() -> None:
    spec = get_weight_spec("iml_vit")
    assert spec.license == "MIT"
    assert spec.commercial_ok is True
    assert spec.method == "gdrive"
    assert spec.filename == "iml-vit_checkpoint.pth"
    assert len(spec.sha256 or "") == 64
    assert "IML-ViT" in spec.source
    assert spec.size_mb is not None and spec.size_mb > 100


def test_an_unknown_name_lists_the_known_ones() -> None:
    with pytest.raises(KeyError, match="iml_vit"):
        get_weight_spec("nope")


def test_the_directory_resolution_prefers_argument_then_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(WEIGHTS_DIR_ENV, raising=False)
    assert resolve_weights_dir() == DEFAULT_WEIGHTS_DIR
    assert weights_file("iml_vit") == DEFAULT_WEIGHTS_DIR / "iml_vit" / "iml-vit_checkpoint.pth"

    monkeypatch.setenv(WEIGHTS_DIR_ENV, str(tmp_path / "from-env"))
    assert resolve_weights_dir() == tmp_path / "from-env"
    assert resolve_weights_dir(tmp_path / "explicit") == tmp_path / "explicit"


def test_fetching_without_accepting_the_license_downloads_nothing(
    fake_spec: WeightSpec, tmp_path: Path
) -> None:
    with pytest.raises(LicenseNotAcceptedError):
        fetch_weights(fake_spec.name, tmp_path)
    assert not (tmp_path / fake_spec.name).exists()


def test_fetching_writes_the_file_and_is_idempotent(
    fake_spec: WeightSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = fetch_weights(fake_spec.name, tmp_path, accept_license=True)
    assert first.status == "downloaded"
    assert first.path == tmp_path / fake_spec.name / fake_spec.filename
    assert first.path.read_bytes() == _PAYLOAD
    assert first.sha256 == _PAYLOAD_SHA256

    # A second call must not download again -- proven by making the download
    # helper fail if it is reached at all.
    def _explode(**_: object) -> str:
        raise AssertionError("the download helper was called for an already-present file")

    monkeypatch.setattr(weights_module, "_gdown_download", _explode)
    second = fetch_weights(fake_spec.name, tmp_path, accept_license=True)
    assert second.status == "cached"
    assert second.sha256 == _PAYLOAD_SHA256


def test_a_corrupted_existing_file_is_reported_rather_than_silently_reused(
    fake_spec: WeightSpec, tmp_path: Path
) -> None:
    target = tmp_path / fake_spec.name / fake_spec.filename
    target.parent.mkdir(parents=True)
    target.write_bytes(b"truncated")

    with pytest.raises(ChecksumMismatchError, match="sha256 mismatch"):
        fetch_weights(fake_spec.name, tmp_path, accept_license=True)
    # Left in place: deleting a file the caller may have put there deliberately
    # is not the fetcher's call.
    assert target.is_file()


def test_a_download_whose_digest_is_wrong_is_deleted(
    fake_spec: WeightSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _write_wrong(*, file_id: str, output: Path) -> str:  # noqa: ARG001
        output.write_bytes(b"something else entirely")
        return str(output)

    monkeypatch.setattr(weights_module, "_gdown_download", _write_wrong)
    with pytest.raises(ChecksumMismatchError, match="file deleted"):
        fetch_weights(fake_spec.name, tmp_path, accept_license=True)
    assert not (tmp_path / fake_spec.name / fake_spec.filename).exists()


def test_a_gdrive_download_that_returns_nothing_is_reported_not_raised(
    fake_spec: WeightSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(weights_module, "_gdown_download", lambda **_: None)
    report = fetch_weights(fake_spec.name, tmp_path, accept_license=True)
    assert report.status == "failed"


def test_cli_weights_list_shows_every_entry_and_whether_it_is_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(WEIGHTS_DIR_ENV, str(tmp_path))
    result = runner.invoke(app, ["weights", "list"])

    assert result.exit_code == 0, result.stdout
    assert "iml_vit" in result.stdout
    assert "MIT" in result.stdout


def test_cli_weights_fetch_refuses_without_the_license_flag(
    fake_spec: WeightSpec, tmp_path: Path
) -> None:
    result = runner.invoke(app, ["weights", "fetch", fake_spec.name, "--dest", str(tmp_path)])

    assert result.exit_code == 1
    assert "license not accepted" in result.stdout
    assert not (tmp_path / fake_spec.name).exists()


def test_cli_weights_fetch_downloads_with_the_license_flag(
    fake_spec: WeightSpec, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        ["weights", "fetch", fake_spec.name, "--dest", str(tmp_path), "--accept-license"],
    )

    assert result.exit_code == 0, result.stdout
    assert (tmp_path / fake_spec.name / fake_spec.filename).read_bytes() == _PAYLOAD


def test_localization_package_imports_without_torch_and_registers_nothing() -> None:
    """Fetching weights must work where the model cannot yet be run.

    Also pins the registration rule: with the ``ml`` extra absent, importing
    :mod:`imgforensics.localization` must not add ``iml_vit`` to the registry,
    so ``imgforensics analyze`` offers exactly what it did before.
    """
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

        import imgforensics.localization as localization
        from imgforensics.core import registry

        assert not blocked & set(sys.modules)
        assert "iml_vit" not in registry.available()
        assert "iml_vit" in localization.WEIGHTS
        assert localization.get_weight_spec("iml_vit").license == "MIT"
        print("imported cleanly")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", snippet], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert "imported cleanly" in completed.stdout
