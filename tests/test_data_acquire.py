"""Tests for imgforensics.data.acquire: license gate, http/hf/gdrive/manual steps.

HTTP behaviour (streaming download, ``.part``-file resume, sha256
verification) is tested against a real local server (stdlib
``http.server``, run in a background thread) rather than mocked, per this
project's network-free-tests policy: no test here reaches the public
internet. ``hf``/``gdrive`` calls are monkeypatched at the wrapper-function
level so the tests do not require (or exercise) the optional
``huggingface_hub``/``gdown`` packages being installed.
"""

from __future__ import annotations

import http.server
import io
import threading
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from imgforensics.cli import app
from imgforensics.data import acquire
from imgforensics.data.acquire import (
    AcquireRecipe,
    AcquireStep,
    ChecksumMismatchError,
    LicenseNotAcceptedError,
    _http_download,
    _unpack,
    fetch,
    get_recipe,
    load_acquire_recipes,
)
from imgforensics.data.registry import DatasetInfo, load_registry

runner = CliRunner()


# --- local HTTP server with Range support (stdlib http.server does not implement it) ---


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    payload: bytes = b""

    def do_GET(self) -> None:  # noqa: N802 -- http.server's naming convention
        data = self.payload
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            start = int(range_header.removeprefix("bytes=").split("-")[0])
            chunk = data[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 -- stdlib signature
        pass  # silence per-request logging in test output


@pytest.fixture
def http_server() -> Iterator[Callable[[bytes], tuple[str, int]]]:
    servers: list[http.server.HTTPServer] = []

    def start(payload: bytes) -> tuple[str, int]:
        handler_cls = type("_BoundRangeHandler", (_RangeHandler,), {"payload": payload})
        server = http.server.HTTPServer(("127.0.0.1", 0), handler_cls)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        return "127.0.0.1", server.server_address[1]

    yield start

    for server in servers:
        server.shutdown()
        server.server_close()


# --- _http_download ---------------------------------------------------------


def test_http_download_fetches_full_file(tmp_path: Path, http_server: Callable) -> None:
    payload = b"hello world " * 500
    host, port = http_server(payload)
    target = tmp_path / "out.bin"

    written = _http_download(f"http://{host}:{port}/f.bin", target, sha256=None, progress=False)

    assert written == len(payload)
    assert target.read_bytes() == payload
    assert not target.with_name(target.name + ".part").exists()


def test_http_download_resumes_from_part_file(tmp_path: Path, http_server: Callable) -> None:
    payload = bytes(range(256)) * 100  # deterministic, not repetitive of a single byte
    host, port = http_server(payload)
    target = tmp_path / "out.bin"
    part = target.with_name(target.name + ".part")
    part.write_bytes(payload[:10_000])

    written = _http_download(f"http://{host}:{port}/f.bin", target, sha256=None, progress=False)

    assert written == len(payload)
    assert target.read_bytes() == payload


def test_http_download_sha256_match_succeeds(tmp_path: Path, http_server: Callable) -> None:
    import hashlib

    payload = b"checksum me"
    host, port = http_server(payload)
    target = tmp_path / "out.bin"
    digest = hashlib.sha256(payload).hexdigest()

    written = _http_download(f"http://{host}:{port}/f.bin", target, sha256=digest, progress=False)

    assert written == len(payload)
    assert target.exists()


def test_http_download_sha256_mismatch_deletes_file(tmp_path: Path, http_server: Callable) -> None:
    payload = b"real bytes"
    host, port = http_server(payload)
    target = tmp_path / "out.bin"

    with pytest.raises(ChecksumMismatchError):
        _http_download(f"http://{host}:{port}/f.bin", target, sha256="0" * 64, progress=False)

    assert not target.exists()
    assert not target.with_name(target.name + ".part").exists()


# --- _unpack ------------------------------------------------------------------


def test_unpack_zip_extracts_into_dest(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("sub/hello.txt", "hi there")
        zf.writestr("top.txt", "top level")

    dest = tmp_path / "out"
    _unpack(archive, dest)

    assert (dest / "sub" / "hello.txt").read_text() == "hi there"
    assert (dest / "top.txt").read_text() == "top level"


def test_unpack_unknown_extension_raises(tmp_path: Path) -> None:
    archive = tmp_path / "a.rar"
    archive.write_bytes(b"not really a rar")
    with pytest.raises(ValueError, match="unpack"):
        _unpack(archive, tmp_path / "out")


def test_unpack_zip_rejects_path_traversal_member(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.txt", "should never land here")

    dest = tmp_path / "out"
    with pytest.raises(ValueError, match=r"\.\./evil\.txt"):
        _unpack(archive, dest)

    assert not (tmp_path / "evil.txt").exists()
    assert not dest.exists() or not any(dest.iterdir())


# --- fetch(): license gate ------------------------------------------------


def _fake_info(
    name: str = "TestDS", *, license_: str = "MIT", commercial_ok: bool | None = True
) -> DatasetInfo:
    return DatasetInfo(
        name=name,
        task="detection",
        homepage="https://example.org/testds",
        download="https://example.org/testds/download",
        access="open",
        license=license_,
        commercial_ok=commercial_ok,
        verified=True,
        verified_on="2026-01-01",
    )


def _fake_recipe(name: str, url: str, filename: str, *, unpack: bool = False) -> AcquireRecipe:
    return AcquireRecipe(
        dataset=name,
        steps=[AcquireStep(method="http", url=url, filename=filename, unpack=unpack)],
    )


def test_fetch_refuses_without_accept_license(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(acquire, "get_dataset", lambda name: _fake_info())
    monkeypatch.setattr(
        acquire, "get_recipe", lambda name: _fake_recipe("TestDS", "http://x/y", "y.bin")
    )

    with pytest.raises(LicenseNotAcceptedError):
        fetch("TestDS", tmp_path, accept_license=False, dry_run=False)

    assert not (tmp_path / "TestDS").exists()


def test_fetch_dry_run_prints_plan_and_downloads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(acquire, "get_dataset", lambda name: _fake_info())
    monkeypatch.setattr(
        acquire, "get_recipe", lambda name: _fake_recipe("TestDS", "http://x/y", "y.bin")
    )

    report = fetch("TestDS", tmp_path, accept_license=False, dry_run=True)

    assert report.dry_run is True
    assert report.accepted_license is False
    assert len(report.steps) == 1
    assert report.steps[0].status == "pending"
    assert not (tmp_path / "TestDS").exists()
    out = capsys.readouterr().out
    assert "Dry run" in out
    assert "MIT" in out


def test_fetch_writes_license_acceptance_file_and_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, http_server: Callable
) -> None:
    payload = b"payload bytes" * 10
    host, port = http_server(payload)
    monkeypatch.setattr(acquire, "get_dataset", lambda name: _fake_info(commercial_ok=False))
    monkeypatch.setattr(
        acquire,
        "get_recipe",
        lambda name: _fake_recipe("TestDS", f"http://{host}:{port}/f.bin", "f.bin"),
    )

    report = fetch("TestDS", tmp_path, accept_license=True, dry_run=False, progress=False)

    assert report.accepted_license is True
    license_path = tmp_path / "TestDS" / "LICENSE_ACCEPTED.txt"
    assert license_path.exists()
    text = license_path.read_text(encoding="utf-8")
    assert "TestDS" in text
    assert "MIT" in text
    assert "Commercial OK: False" in text
    assert report.steps[0].status == "done"
    assert (tmp_path / "TestDS" / "f.bin").read_bytes() == payload


def test_fetch_unpacks_when_step_requests_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, http_server: Callable
) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("inner/file.txt", "contents")
    payload = buffer.getvalue()
    host, port = http_server(payload)

    monkeypatch.setattr(acquire, "get_dataset", lambda name: _fake_info())
    monkeypatch.setattr(
        acquire,
        "get_recipe",
        lambda name: _fake_recipe("TestDS", f"http://{host}:{port}/a.zip", "a.zip", unpack=True),
    )

    fetch("TestDS", tmp_path, accept_license=True, dry_run=False, progress=False)

    assert (tmp_path / "TestDS" / "inner" / "file.txt").read_text() == "contents"


def test_fetch_manual_step_is_pending_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(acquire, "get_dataset", lambda name: _fake_info())
    monkeypatch.setattr(
        acquire,
        "get_recipe",
        lambda name: AcquireRecipe(
            dataset="TestDS",
            steps=[
                AcquireStep(
                    method="manual", instructions="Visit https://example.org and click download."
                )
            ],
        ),
    )

    report = fetch("TestDS", tmp_path, accept_license=True, dry_run=False)

    assert report.steps[0].status == "pending"
    assert report.steps[0].method == "manual"


def test_fetch_unknown_variant_raises_key_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(acquire, "get_dataset", lambda name: _fake_info())
    monkeypatch.setattr(
        acquire, "get_recipe", lambda name: _fake_recipe("TestDS", "http://x/y", "y.bin")
    )
    with pytest.raises(KeyError):
        fetch("TestDS", tmp_path, variant="does-not-exist", dry_run=True)


# --- hf / gdrive: monkeypatched wrapper functions --------------------------


def test_fetch_hf_step_calls_wrapper_with_expected_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []

    def fake_snapshot(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(kwargs["local_dir"])

    monkeypatch.setattr(acquire, "_hf_snapshot_download", fake_snapshot)
    monkeypatch.setattr(acquire, "get_dataset", lambda name: _fake_info())
    monkeypatch.setattr(
        acquire,
        "get_recipe",
        lambda name: AcquireRecipe(
            dataset="TestDS",
            steps=[
                AcquireStep(
                    method="hf",
                    repo_id="org/repo",
                    repo_type="dataset",
                    allow_patterns=["data/*.parquet"],
                )
            ],
        ),
    )

    report = fetch("TestDS", tmp_path, accept_license=True, dry_run=False)

    assert len(calls) == 1
    assert calls[0]["repo_id"] == "org/repo"
    assert calls[0]["allow_patterns"] == ["data/*.parquet"]
    assert report.steps[0].status == "done"


def test_fetch_hf_step_with_max_files_bounds_sorts_and_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    list_calls: list[dict] = []
    download_calls: list[dict] = []

    def fake_list_files(**kwargs: object) -> list[str]:
        list_calls.append(kwargs)
        return [
            "data/003.parquet",
            "data/001.parquet",
            "README.md",
            "data/002.parquet",
            "data/000.parquet",
        ]

    def fake_hub_download(**kwargs: object) -> str:
        download_calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * 10)
        return str(target)

    monkeypatch.setattr(acquire, "_hf_list_repo_files", fake_list_files)
    monkeypatch.setattr(acquire, "_hf_hub_download", fake_hub_download)
    monkeypatch.setattr(acquire, "get_dataset", lambda name: _fake_info())
    monkeypatch.setattr(
        acquire,
        "get_recipe",
        lambda name: AcquireRecipe(
            dataset="TestDS",
            steps=[
                AcquireStep(
                    method="hf",
                    repo_id="org/repo",
                    repo_type="dataset",
                    allow_patterns=["data/*.parquet"],
                    max_files=2,
                )
            ],
        ),
    )

    report = fetch("TestDS", tmp_path, accept_license=True, dry_run=False)

    assert len(list_calls) == 1
    requested = [call["filename"] for call in download_calls]
    # First two sorted matches of "data/*.parquet"; "README.md" and the
    # remaining shards are excluded.
    assert requested == ["data/000.parquet", "data/001.parquet"]
    assert report.steps[0].status == "done"
    assert report.steps[0].bytes_downloaded == 20
    assert "2 file(s)" in report.steps[0].description


def test_fetch_dry_run_community_forensics_shows_bounded_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = fetch("Community Forensics", tmp_path, dry_run=True)

    assert report.dry_run is True
    assert len(report.steps) == 1
    assert "max_files=8" in report.steps[0].description
    out = capsys.readouterr().out
    assert "max_files=8" in out


def test_fetch_gdrive_step_calls_wrapper_and_unpacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("real/a.jpg", "fake-image-bytes")
    zip_bytes = buffer.getvalue()

    def fake_gdown(*, file_id: str, output: Path) -> str:
        Path(output).write_bytes(zip_bytes)
        return str(output)

    monkeypatch.setattr(acquire, "_gdown_download", fake_gdown)
    monkeypatch.setattr(acquire, "get_dataset", lambda name: _fake_info())
    monkeypatch.setattr(
        acquire,
        "get_recipe",
        lambda name: AcquireRecipe(
            dataset="TestDS",
            steps=[AcquireStep(method="gdrive", file_id="abc123", filename="d.zip", unpack=True)],
        ),
    )

    report = fetch("TestDS", tmp_path, accept_license=True, dry_run=False)

    assert report.steps[0].status == "done"
    assert (tmp_path / "TestDS" / "real" / "a.jpg").exists()


def test_fetch_gdrive_step_reports_skipped_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(acquire, "_gdown_download", lambda **kwargs: None)
    monkeypatch.setattr(acquire, "get_dataset", lambda name: _fake_info())
    monkeypatch.setattr(
        acquire,
        "get_recipe",
        lambda name: AcquireRecipe(
            dataset="TestDS",
            steps=[AcquireStep(method="gdrive", file_id="abc123", filename="d.zip", unpack=True)],
        ),
    )

    report = fetch("TestDS", tmp_path, accept_license=True, dry_run=False)

    assert report.steps[0].status == "skipped"


# --- packaged acquire.yaml validation --------------------------------------


def test_every_recipe_references_a_registry_dataset() -> None:
    registry_names = {entry.name for entry in load_registry()}
    for recipe in load_acquire_recipes():
        assert recipe.dataset in registry_names, (
            f"{recipe.dataset!r} has no matching registry entry"
        )


def test_every_recipe_step_has_its_required_fields() -> None:
    for recipe in load_acquire_recipes():
        all_steps = list(recipe.steps)
        for variant_steps in (recipe.variants or {}).values():
            all_steps.extend(variant_steps)
        assert all_steps, f"{recipe.dataset!r} has no steps at all"
        for step in all_steps:
            if step.method == "http":
                assert step.url and step.filename
            elif step.method == "hf":
                assert step.repo_id
            elif step.method == "gdrive":
                assert step.file_id and step.filename
            elif step.method == "manual":
                assert step.instructions


def test_get_recipe_unknown_dataset_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_recipe("does-not-exist")


# --- CLI ---------------------------------------------------------------------


def test_cli_datasets_recipe_prints_steps() -> None:
    result = runner.invoke(app, ["datasets", "recipe", "Synthbuster"])
    assert result.exit_code == 0
    assert "zenodo.org" in result.stdout


def test_cli_datasets_recipe_unknown_name_errors() -> None:
    result = runner.invoke(app, ["datasets", "recipe", "does-not-exist"])
    assert result.exit_code != 0


def test_cli_datasets_fetch_without_accept_license_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["datasets", "fetch", "Synthbuster", "--dest", str(tmp_path)])
    assert result.exit_code == 1
    assert not (tmp_path / "Synthbuster").exists()


def test_cli_datasets_fetch_dry_run_downloads_nothing(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["datasets", "fetch", "Synthbuster", "--dest", str(tmp_path), "--dry-run"]
    )
    assert result.exit_code == 0
    assert not (tmp_path / "Synthbuster").exists()
    assert "Fetch report" in result.stdout


def test_cli_datasets_fetch_dry_run_community_forensics_shows_bounded_plan(
    tmp_path: Path,
) -> None:
    result = runner.invoke(
        app, ["datasets", "fetch", "Community Forensics", "--dest", str(tmp_path), "--dry-run"]
    )
    assert result.exit_code == 0
    assert not (tmp_path / "Community Forensics").exists()
    assert "max_files=8" in result.stdout


def test_cli_datasets_fetch_hf_recipe_with_monkeypatched_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []

    def fake_snapshot(**kwargs: object) -> str:
        calls.append(kwargs)
        Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)
        return str(kwargs["local_dir"])

    monkeypatch.setattr(acquire, "_hf_snapshot_download", fake_snapshot)

    result = runner.invoke(
        app, ["datasets", "fetch", "ITW-SM", "--dest", str(tmp_path), "--accept-license"]
    )

    assert result.exit_code == 0
    assert calls and calls[0]["repo_id"] == "dkarageo/itw-sm"
    assert (tmp_path / "ITW-SM" / "LICENSE_ACCEPTED.txt").exists()


def test_cli_datasets_fetch_gdrive_recipe_with_monkeypatched_gdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("train/0_real/a.jpg", "x")
    zip_bytes = buffer.getvalue()

    def fake_gdown(*, file_id: str, output: Path) -> str:
        Path(output).write_bytes(zip_bytes)
        return str(output)

    monkeypatch.setattr(acquire, "_gdown_download", fake_gdown)

    result = runner.invoke(
        app, ["datasets", "fetch", "WildRF", "--dest", str(tmp_path), "--accept-license"]
    )

    assert result.exit_code == 0
    assert (tmp_path / "WildRF" / "train" / "0_real" / "a.jpg").exists()
