"""Tests for the workbench client the API serves at ``/``.

What can be checked from Python is that the page is *reachable and complete*:
that ``/`` returns the HTML, that the files it names are served under
``/static`` with media types a browser will execute, that the JSON routes
still answer with the mount in place, and that the files are located the way
an installed wheel would locate them rather than the way a checkout can get
away with.

What cannot be checked here is behaviour: there is no JavaScript engine in
this test environment and adding one would mean a Node toolchain, which is the
dependency the no-build decision exists to avoid. So the JavaScript is held to
the two properties that survive without an engine -- it is text, and it is
offline: no ``<script`` inside a module, no absolute URL anywhere, and an
``index.html`` that names nothing but ``/static`` assets.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from imgforensics.api import create_app, static_dir  # noqa: E402

#: Where the checkout keeps its packaging metadata, or ``None`` when these
#: tests run against an installed wheel with no source tree around it.
_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

#: Every file the client is made of.
_CLIENT_FILES = ("index.html", "style.css", "api.js", "app.js", "ui.js", "viewer.js")

#: An absolute URL, which the client must not contain: the page has to work
#: with no network beyond the server it came from, and a CDN reference is the
#: usual way that stops being true.
_ABSOLUTE_URL = re.compile(r"https?://")

#: What ``index.html`` may load: its own origin's static files, and nothing
#: else. ``href``/``src`` only -- ``http-equiv`` is an attribute name, not a URL.
_ASSET_REFERENCE = re.compile(r"(?:href|src)=\"([^\"]+)\"")


def _client() -> TestClient:
    return TestClient(create_app())


def _read(name: str) -> str:
    return (static_dir() / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Serving
# --------------------------------------------------------------------------


def test_the_root_serves_the_workbench_page() -> None:
    response = _client().get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].split(";")[0] == "text/html"
    assert "Content-Security-Policy" in response.text
    assert "/static/style.css" in response.text
    assert "/static/app.js" in response.text


def test_the_client_files_are_served_as_something_a_browser_will_run() -> None:
    client = _client()

    script = client.get("/static/app.js")
    style = client.get("/static/style.css")

    assert script.status_code == 200
    assert script.headers["content-type"].split(";")[0] in {
        "text/javascript",
        "application/javascript",
    }
    assert style.status_code == 200
    assert style.headers["content-type"].split(";")[0] == "text/css"


def test_a_file_the_client_does_not_have_is_a_404() -> None:
    assert _client().get("/static/missing.js").status_code == 404


def test_the_json_routes_still_answer_with_the_static_mount_in_place() -> None:
    client = _client()

    assert client.get("/health").json()["status"] == "ok"
    assert isinstance(client.get("/tools").json(), list)
    assert client.get("/sessions/nosuchid").status_code == 404


def test_the_page_is_not_part_of_the_json_schema() -> None:
    schema = _client().get("/openapi.json").json()

    assert "/" not in schema["paths"]
    assert "/tools" in schema["paths"]


# --------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------


def test_the_static_directory_is_found_through_the_package() -> None:
    directory = static_dir()

    assert directory.is_dir()
    for name in _CLIENT_FILES:
        assert (directory / name).is_file(), name


@pytest.mark.skipif(not _PYPROJECT.is_file(), reason="no source tree to read packaging from")
def test_the_client_files_are_declared_as_package_data() -> None:
    text = _PYPROJECT.read_text(encoding="utf-8")

    section = text.split("[tool.setuptools.package-data]", 1)[1].split("\n[", 1)[0]
    assert "api/static/*" in section


# --------------------------------------------------------------------------
# The JavaScript, as far as Python can read it
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", [name for name in _CLIENT_FILES if name.endswith(".js")])
def test_every_module_is_text_and_names_no_remote_host(name: str) -> None:
    source = _read(name)

    assert source.strip()
    assert "<script" not in source
    assert not _ABSOLUTE_URL.search(source), f"{name} names a remote host"


def test_the_stylesheet_names_no_remote_host() -> None:
    assert not _ABSOLUTE_URL.search(_read("style.css"))


def test_the_page_loads_nothing_but_its_own_static_files() -> None:
    references = _ASSET_REFERENCE.findall(_read("index.html"))

    assert references, "the page loads nothing at all"
    for reference in references:
        assert reference.startswith("/static/"), reference


def test_the_page_declares_the_policy_that_keeps_it_offline() -> None:
    page = _read("index.html")

    assert "default-src 'self'" in page
    assert "img-src 'self' blob: data:" in page
    assert "style-src 'self'" in page
