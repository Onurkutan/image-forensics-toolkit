"""Tests for imgforensics.api: the JSON contract a workbench client talks to.

Everything runs through FastAPI's ``TestClient`` against an application built
by :func:`~imgforensics.api.create_app`, over two tools registered in this
module -- a signal that reports both maps and one parameter, and a view -- so
no real model runs, no weights are needed and no torch is imported. The weight
and head directories are pointed at an empty temporary directory as well, so
an ``ml`` tool that somehow got run would abstain instead of downloading
anything, and the working directory is moved there too, so a fuser sitting in
the checkout's ``weights/`` is never picked up by accident.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import synthetic_fusion_records
from PIL import Image

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from imgforensics import __version__  # noqa: E402
from imgforensics.api import create_app  # noqa: E402
from imgforensics.cli import app as cli_app  # noqa: E402
from imgforensics.core import registry  # noqa: E402
from imgforensics.core.base import BaseDetector  # noqa: E402
from imgforensics.core.image import ForensicImage  # noqa: E402
from imgforensics.core.parameters import ParameterSpec  # noqa: E402
from imgforensics.core.types import DetectionResult, label_from_score  # noqa: E402
from imgforensics.fusion.stacking import Fuser, fit_fuser  # noqa: E402
from imgforensics.service import AnalysisSession, SessionStore, to_png  # noqa: E402
from imgforensics.views.base import ViewTool  # noqa: E402

runner = CliRunner()

#: The two tools these tests run. Registered here rather than borrowed from
#: the real catalogue so that a change in what ELA scores never fails an API
#: test, and so that both maps and a parameter are always present.
_SIGNAL_NAME = "api_probe"
_VIEW_NAME = "api_probe_view"

#: Test image size, wide enough that level 0 is more than one tile across.
_IMAGE_SIZE = (400, 300)


class _ProbeSignal(BaseDetector):
    """A signal with one parameter, both maps, and a numpy value in its details."""

    name = _SIGNAL_NAME

    def __init__(self, level: int = 1) -> None:
        self.level = level

    @classmethod
    def parameters(cls) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                name="level",
                kind="int",
                default=1,
                minimum=0,
                maximum=3,
                description="How suspicious the probe pretends to be.",
            )
        ]

    def predict(self, image: ForensicImage) -> DetectionResult:
        ramp = np.linspace(0.0, 1.0, image.height * image.width, dtype=np.float32).reshape(
            image.height, image.width
        )
        score = self.level / 4.0
        return DetectionResult(
            detector=self.name,
            score=score,
            label=label_from_score(score),
            heatmap=ramp * (self.level / 3.0),
            attribution=np.ascontiguousarray(ramp[::-1]),
            details={"level": self.level, "not_json_native": np.float32(0.25)},
        )


class _ProbeView(ViewTool):
    """A view: one map, no verdict, and no attribution to go with it."""

    name = _VIEW_NAME

    def predict(self, image: ForensicImage) -> DetectionResult:
        return self.view_result(np.zeros((image.height, image.width), dtype=np.float32))


registry.register(_SIGNAL_NAME)(_ProbeSignal)
registry.register(_VIEW_NAME)(_ProbeView)


class _FakeClock:
    """A monotonic clock a test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(autouse=True)
def _isolated_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No weights, no head checkpoint, and no fuser found by accident."""
    monkeypatch.setenv("IMGFORENSICS_WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("IMGFORENSICS_HEAD_DIR", str(tmp_path / "head"))
    monkeypatch.delenv("IMGFORENSICS_FUSER", raising=False)
    monkeypatch.chdir(tmp_path)


def _png_bytes(size: tuple[int, int] = _IMAGE_SIZE, seed: int = 0) -> bytes:
    """A small PNG with real pixel variation, as an upload would arrive."""
    width, height = size
    pixels = np.random.default_rng(seed).integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _client(**kwargs: Any) -> TestClient:
    return TestClient(create_app(**kwargs))


def _upload(client: TestClient, data: bytes | None = None, name: str = "sample.png") -> str:
    response = client.post(
        "/sessions", files={"file": (name, data if data is not None else _png_bytes(), "image/png")}
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def _fitted_fuser() -> Fuser:
    """A fuser fitted on synthetic records naming the two probes."""
    records = synthetic_fusion_records(
        {
            _SIGNAL_NAME: lambda y, rng: float(
                np.clip(y * 0.9 + 0.05 + rng.normal(0, 0.05), 1e-3, 1 - 1e-3)
            ),
            _VIEW_NAME: lambda _y, _rng: 0.5,
        },
        n_per_class=150,
        seed=0,
    )
    return fit_fuser(records)


# --------------------------------------------------------------------------
# Health and catalogue
# --------------------------------------------------------------------------


def test_health_reports_the_version_whose_contract_this_is() -> None:
    response = _client().get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_the_catalogue_lists_every_registered_tool_with_its_parameters() -> None:
    body = _client().get("/tools").json()

    by_name = {entry["name"]: entry for entry in body}
    assert set(by_name) == set(registry.available())
    probe = by_name[_SIGNAL_NAME]
    assert probe["kind"] == "signal"
    assert probe["installed"] is True
    assert probe["needs_ml"] is False
    assert [parameter["name"] for parameter in probe["parameters"]] == ["level"]
    assert probe["parameters"][0]["default"] == 1
    assert (probe["parameters"][0]["minimum"], probe["parameters"][0]["maximum"]) == (0, 3)
    assert by_name[_VIEW_NAME]["kind"] == "view"


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------


def test_uploading_an_image_starts_a_session() -> None:
    client = _client()

    response = client.post("/sessions", files={"file": ("holiday.png", _png_bytes(), "image/png")})

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "holiday.png"
    assert (body["width"], body["height"]) == _IMAGE_SIZE
    assert body["format"] == "PNG"
    assert client.get(f"/sessions/{body['id']}").json() == {**body, "tools_run": []}


def test_an_upload_over_the_limit_is_refused() -> None:
    client = _client(max_upload_bytes=1024)

    response = client.post("/sessions", files={"file": ("big.png", _png_bytes(), "image/png")})

    assert response.status_code == 413
    assert "1024" in response.json()["detail"]


def test_an_upload_no_decoder_understands_is_refused() -> None:
    client = _client()

    response = client.post(
        "/sessions", files={"file": ("notes.png", b"this is not an image", "image/png")}
    )

    assert response.status_code == 422
    assert "notes.png" in response.json()["detail"]


def test_an_unknown_session_is_a_404() -> None:
    client = _client()

    assert client.get("/sessions/nosuchid").status_code == 404
    assert client.post(f"/sessions/nosuchid/tools/{_SIGNAL_NAME}").status_code == 404
    assert "nosuchid" in client.get("/sessions/nosuchid").json()["detail"]


def test_an_expired_session_is_a_404() -> None:
    clock = _FakeClock()
    client = _client(store=SessionStore(ttl_seconds=60, clock=clock))
    session_id = _upload(client)

    assert client.get(f"/sessions/{session_id}").status_code == 200
    clock.advance(61)

    assert client.get(f"/sessions/{session_id}").status_code == 404


# --------------------------------------------------------------------------
# Running tools
# --------------------------------------------------------------------------


def test_running_a_tool_returns_its_result_and_where_to_fetch_its_maps() -> None:
    client = _client()
    session_id = _upload(client)

    response = client.post(f"/sessions/{session_id}/tools/{_SIGNAL_NAME}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["detector"] == _SIGNAL_NAME
    assert payload["score"] == pytest.approx(0.25)
    assert payload["label"] == "real"
    assert payload["elapsed_ms"] >= 0.0
    assert payload["details"] == {"level": 1, "not_json_native": pytest.approx(0.25)}
    assert set(payload["maps"]) == {"heatmap", "attribution"}
    heatmap = payload["maps"]["heatmap"]
    assert heatmap["levels"] == 2
    assert heatmap["shapes"] == [[300, 400], [150, 200]]
    assert heatmap["tile_size"] == 256
    assert heatmap["tile_url"] == (
        f"/sessions/{session_id}/maps/{_SIGNAL_NAME}/heatmap/{{z}}/{{x}}/{{y}}.png"
    )
    assert client.get(heatmap["tile_url"].format(z=0, x=0, y=0)).status_code == 200
    assert client.get(f"/sessions/{session_id}").json()["tools_run"] == [_SIGNAL_NAME]


def test_parameters_are_applied_and_saying_nothing_means_the_defaults() -> None:
    client = _client()
    session_id = _upload(client)

    with_default = client.post(
        f"/sessions/{session_id}/tools/{_SIGNAL_NAME}", json={"parameters": {}}
    ).json()
    without_body = client.post(f"/sessions/{session_id}/tools/{_SIGNAL_NAME}").json()
    raised = client.post(
        f"/sessions/{session_id}/tools/{_SIGNAL_NAME}", json={"parameters": {"level": 3}}
    ).json()

    assert without_body["details"] == with_default["details"]
    assert raised["details"]["level"] == 3
    assert raised["score"] == pytest.approx(0.75)


def test_a_parameter_the_tool_refuses_is_a_422() -> None:
    client = _client()
    session_id = _upload(client)

    out_of_range = client.post(
        f"/sessions/{session_id}/tools/{_SIGNAL_NAME}", json={"parameters": {"level": 9}}
    )
    undeclared = client.post(
        f"/sessions/{session_id}/tools/{_SIGNAL_NAME}", json={"parameters": {"lvl": 1}}
    )

    assert out_of_range.status_code == 422
    assert "at most 3" in out_of_range.json()["detail"]
    assert undeclared.status_code == 422
    assert "Unknown parameter" in undeclared.json()["detail"]


def test_an_unknown_tool_is_a_404_listing_the_known_ones() -> None:
    client = _client()
    session_id = _upload(client)

    response = client.post(f"/sessions/{session_id}/tools/no_such_tool")

    assert response.status_code == 404
    assert "No detector registered" in response.json()["detail"]


# --------------------------------------------------------------------------
# Map tiles
# --------------------------------------------------------------------------


def test_a_tile_is_the_png_the_service_layer_would_have_produced() -> None:
    data = _png_bytes()
    client = _client()
    session_id = _upload(client, data)
    client.post(f"/sessions/{session_id}/tools/{_SIGNAL_NAME}", json={"parameters": {"level": 2}})

    response = client.get(f"/sessions/{session_id}/maps/{_SIGNAL_NAME}/heatmap/0/1/0.png")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    with Image.open(io.BytesIO(response.content)) as tile:
        assert tile.mode == "L"
        assert tile.size == (256, 256)
    offline = AnalysisSession(
        ForensicImage.from_bytes(data, path=Path("sample.png")), name="sample.png"
    )
    pyramid = offline.maps(_SIGNAL_NAME, {"level": 2})["heatmap"]
    assert response.content == to_png(pyramid.tile(0, 1, 0, size=256))


def test_a_tile_follows_the_parameters_the_tool_last_ran_with() -> None:
    client = _client()
    session_id = _upload(client)
    url = f"/sessions/{session_id}/maps/{_SIGNAL_NAME}/heatmap/0/0/0.png"

    client.post(f"/sessions/{session_id}/tools/{_SIGNAL_NAME}", json={"parameters": {"level": 1}})
    dim = client.get(url).content
    client.post(f"/sessions/{session_id}/tools/{_SIGNAL_NAME}", json={"parameters": {"level": 3}})
    bright = client.get(url).content

    assert dim != bright


def test_a_tile_outside_the_level_is_a_422() -> None:
    client = _client()
    session_id = _upload(client)
    client.post(f"/sessions/{session_id}/tools/{_SIGNAL_NAME}")

    response = client.get(f"/sessions/{session_id}/maps/{_SIGNAL_NAME}/heatmap/0/9/0.png")

    assert response.status_code == 422
    assert "outside level 0" in response.json()["detail"]


def test_a_tile_for_a_map_that_is_not_there_is_a_404() -> None:
    client = _client()
    session_id = _upload(client)
    client.post(f"/sessions/{session_id}/tools/{_VIEW_NAME}")

    not_run = client.get(f"/sessions/{session_id}/maps/{_SIGNAL_NAME}/heatmap/0/0/0.png")
    no_such_map = client.get(f"/sessions/{session_id}/maps/{_VIEW_NAME}/attribution/0/0/0.png")

    assert not_run.status_code == 404
    assert "has not run" in not_run.json()["detail"]
    assert no_such_map.status_code == 404
    assert "no 'attribution' map" in no_such_map.json()["detail"]


# --------------------------------------------------------------------------
# Fusion and report
# --------------------------------------------------------------------------


def test_the_fused_verdict_comes_from_the_configured_fuser() -> None:
    client = _client(fuser=_fitted_fuser())
    session_id = _upload(client)
    client.post(f"/sessions/{session_id}/tools/{_SIGNAL_NAME}", json={"parameters": {"level": 3}})

    response = client.get(f"/sessions/{session_id}/fusion")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"probability", "label", "band", "contributions"}
    assert 0.0 <= payload["probability"] <= 1.0
    assert payload["label"] in {"real", "fake", "uncertain"}
    by_detector = {item["detector"]: item for item in payload["contributions"]}
    assert by_detector[_SIGNAL_NAME]["present"] is True
    assert by_detector[_VIEW_NAME]["present"] is False


def test_there_is_no_fused_verdict_without_a_fuser() -> None:
    client = _client()
    session_id = _upload(client)

    response = client.get(f"/sessions/{session_id}/fusion")

    assert response.status_code == 404
    assert "no fuser configured" in response.json()["detail"]


def test_a_fuser_file_is_loaded_when_one_is_named(tmp_path: Path) -> None:
    fuser_path = tmp_path / "fuser.json"
    _fitted_fuser().save(fuser_path)
    client = _client(fuser_path=fuser_path)
    session_id = _upload(client)
    client.post(f"/sessions/{session_id}/tools/{_SIGNAL_NAME}")

    assert client.get(f"/sessions/{session_id}/fusion").status_code == 200


def test_the_report_zip_carries_the_report_the_cli_writes() -> None:
    client = _client()
    session_id = _upload(client)
    client.post(f"/sessions/{session_id}/tools/{_SIGNAL_NAME}")

    response = client.get(f"/sessions/{session_id}/report.zip")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["content-disposition"] == 'attachment; filename="sample_report.zip"'
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = set(archive.namelist())
        assert {"report.json", "report.md"} <= names
        assert f"{_SIGNAL_NAME}_heatmap.png" in names
        assert all("/" not in name and "\\" not in name for name in names)


# --------------------------------------------------------------------------
# The serve command
# --------------------------------------------------------------------------


def test_serve_without_the_api_extra_prints_an_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "uvicorn", None)

    result = runner.invoke(cli_app, ["serve"])

    assert result.exit_code == 1
    assert 'pip install "imgforensics[api]"' in result.stdout


def test_serve_hands_the_configured_application_to_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uvicorn = pytest.importorskip("uvicorn")
    recorded: dict[str, Any] = {}

    def _record(application: Any, **kwargs: Any) -> None:
        recorded["application"] = application
        recorded.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", _record)

    result = runner.invoke(
        cli_app,
        [
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            "9123",
            "--ttl-seconds",
            "5",
            "--max-sessions",
            "2",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert isinstance(recorded["application"], fastapi.FastAPI)
    assert (recorded["host"], recorded["port"]) == ("127.0.0.1", 9123)
    store = recorded["application"].state.store
    assert (store.ttl_seconds, store.max_sessions) == (5, 2)
    assert recorded["application"].state.fuser is None
    assert "9123" in result.stdout
