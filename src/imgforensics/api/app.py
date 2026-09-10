"""The routes themselves: a session, a tool, a tile, a verdict, a report.

Every route here is a few lines of translation over
:mod:`imgforensics.service`, and that is the whole design. The service layer
already decides what a tool is, what its parameters mean, when a result is
cached and how a map is cut into tiles; repeating any of that here would give
the workbench a second opinion on questions the CLI and the benchmark runner
have only one of.

Three translations are worth naming, because they are the ones a client
depends on:

- **Errors are the service's, mapped once.** The registry raises ``KeyError``
  for a tool nobody registered and :class:`~imgforensics.service.SessionStore`
  raises it for a session that never existed or has expired; both become 404.
  :func:`~imgforensics.core.parameters.coerce_parameters` raises ``ValueError``
  for a parameter the caller got wrong, and so does a tile outside its level;
  both become 422 carrying the message the service wrote, which was written to
  be shown to whoever sent the request. Nothing else leaks: no traceback, and
  no path from this machine's filesystem.
- **Maps are never JSON.** A run returns the numbers plus, per map, its
  levels, their shapes and a tile URL template; the pixels arrive as 8-bit
  grayscale PNG tiles from :func:`~imgforensics.service.to_png`, the same
  quantization ``analyze --save-heatmaps`` writes.
- **A tile URL names a tool, not a parameter set.** It serves the map of the
  run that tool last made in this session, so moving a slider re-runs the tool
  and the very same tile URLs then return the new map -- see
  :meth:`~imgforensics.service.AnalysisSession.latest_maps`.

One route translates nothing: ``GET /`` returns the workbench client, the
plain HTML, CSS and ES modules under ``static/``, and ``/static`` serves the
rest of them. Shipping the client inside the package is what makes
``imgforensics serve`` one process and one URL rather than a server plus a
Node toolchain, and it costs this module a mount and a file response.
"""

from __future__ import annotations

import io
import mimetypes
import re
import tempfile
import zipfile
from importlib import resources
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

# The four packages are imported for their registration side effects, the same
# set the CLI imports: a catalogue that depended on what a caller happened to
# import first would answer /tools differently from one run to the next.
from imgforensics import __version__, detectors, localization, signals, views  # noqa: F401
from imgforensics.core.image import ForensicImage
from imgforensics.fusion.stacking import Fuser
from imgforensics.service import AnalysisSession, MapPyramid, SessionStore, catalogue, to_png
from imgforensics.service.session import MAX_LEVEL_SIDE
from imgforensics.utils.jsonsafe import to_jsonable

#: The status codes this API answers with, spelled out rather than imported
#: from ``fastapi.status``: two of these names have been renamed between
#: Starlette releases, and a deprecation warning is not worth the import.
_CREATED = 201
_NOT_FOUND = 404
_TOO_LARGE = 413
_UNPROCESSABLE = 422

#: Largest upload accepted by default. A 64 MiB ceiling clears any camera JPEG
#: and any reasonable PNG while keeping one careless client from filling this
#: process's memory; it is an argument because a deployment behind a proxy
#: usually wants a smaller one.
DEFAULT_MAX_UPLOAD_BYTES = 64 * 1024 * 1024

#: How much of an upload is read per step, so the size limit is enforced while
#: the bytes arrive rather than after all of them have been buffered.
_UPLOAD_CHUNK_BYTES = 1024 * 1024

#: Tile side served by the tile route, matching the pyramid's smallest level
#: so that the last level is always exactly one tile.
TILE_SIZE = MAX_LEVEL_SIDE

#: Everything that may not appear in a downloaded file's name. That name is
#: built from what somebody uploaded, and it goes back out in a header.
_UNSAFE_IN_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")

#: What the client's own files are served as. Registered explicitly because
#: :mod:`mimetypes` answers from the registry on Windows, where ``.js`` is
#: mapped by whatever last installed a script host -- and a browser refuses to
#: execute a module it was told is ``text/plain``, which would leave the page
#: blank on one machine and fine on the next.
_WEB_MEDIA_TYPES = {".js": "text/javascript", ".css": "text/css", ".html": "text/html"}


def _register_web_media_types() -> None:
    """Pin the client's media types, whatever this machine's registry says."""
    for suffix, media_type in _WEB_MEDIA_TYPES.items():
        mimetypes.add_type(media_type, suffix)


_register_web_media_types()


def static_dir() -> Path:
    """The directory the workbench client's files live in.

    Located through :mod:`importlib.resources` rather than relative to
    ``__file__``, because "where is this package's data" is exactly the
    question, and it is one that has the same answer for a checkout and for an
    installed wheel -- provided ``pyproject.toml`` keeps ``api/static/*`` in
    its package data, which is what puts the files in the wheel at all.
    """
    return Path(str(resources.files("imgforensics.api").joinpath("static")))


class RunRequest(BaseModel):
    """The body of a tool run: what the user set, and nothing else.

    Attributes:
        parameters: ``{name: value}`` for the tool's declared parameters.
            Anything left out takes that parameter's default, so an empty body
            runs the tool exactly as its catalogue entry describes it. Strings
            are accepted for every kind (a form field carries ``"80"``, not
            ``80``), and a name the tool does not declare is refused with 422
            rather than ignored.
    """

    parameters: dict[str, Any] = Field(default_factory=dict)


def create_app(
    *,
    store: SessionStore | None = None,
    fuser: Fuser | None = None,
    fuser_path: Path | None = None,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> FastAPI:
    """Build the API application.

    Everything the routes need is resolved here and kept on ``app.state``, so
    that no request ever reads an environment variable or touches the
    filesystem to find out how this server was configured.

    Args:
        store: The session store to serve from. Defaults to a fresh one with
            the service layer's TTL and population cap.
        fuser: An already-loaded fuser, which wins over ``fuser_path``.
        fuser_path: A fuser file to load. When both are ``None`` the CLI's
            resolution applies -- ``$IMGFORENSICS_FUSER``, then
            ``weights/fuser.json`` if it exists -- and a server that finds
            neither simply has no fused verdict to offer.
        max_upload_bytes: Largest upload accepted, in bytes.

    Returns:
        A FastAPI application, ready for uvicorn or for a ``TestClient``.

    Raises:
        FileNotFoundError: if a fuser path was given, or found in the
            environment, but no file is there. A server that was asked for a
            fuser and cannot load one should say so at startup rather than
            once per request.
    """
    app = FastAPI(
        title="imgforensics",
        version=__version__,
        summary="Run image forensics tools on one uploaded image and read their maps.",
        description=(
            "The JSON contract of the forensic workbench. Sessions live in this "
            "process's memory behind a TTL, there is no authentication, and every "
            "map is served as 8-bit grayscale PNG tiles."
        ),
    )
    app.state.store = store if store is not None else SessionStore()
    app.state.fuser = _resolve_fuser(fuser, fuser_path)
    app.state.max_upload_bytes = max_upload_bytes

    @app.exception_handler(KeyError)
    def _handle_missing(_request: Request, exc: Exception) -> JSONResponse:
        """A session or a tool that is not there: the service's message, as a 404."""
        return JSONResponse(status_code=_NOT_FOUND, content=_detail(exc))

    @app.exception_handler(ValueError)
    def _handle_rejected(_request: Request, exc: Exception) -> JSONResponse:
        """A parameter or a tile the service refused: its message, as a 422."""
        return JSONResponse(status_code=_UNPROCESSABLE, content=_detail(exc))

    static = static_dir()
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/", include_in_schema=False, response_class=FileResponse)
    def workbench() -> FileResponse:
        """The workbench client: the one page this server is meant to be used from.

        Kept out of the schema on purpose. ``/docs`` describes the JSON
        contract, and an HTML page is not part of it -- a client that reads
        the schema to find out what this server can do should find the routes
        below and nothing else.
        """
        return FileResponse(static / "index.html", media_type="text/html")

    @app.get("/health")
    def health() -> dict[str, str]:
        """Say that the server is up, and whose version's contract it speaks."""
        return {"status": "ok", "version": __version__}

    @app.get("/tools")
    def tools() -> list[dict[str, Any]]:
        """The tool catalogue: what a client can draw before it runs anything.

        One entry per registered tool, ordered by category and then by name,
        with its parameters and with ``installed`` saying whether its weights
        are on this machine. A tool that is not installed is still listed and
        still runnable -- it abstains with a reason -- so a client greys it out
        and offers the fetch command instead of hiding it.
        """
        return [spec.model_dump() for spec in catalogue()]

    @app.post("/sessions", status_code=_CREATED)
    def create_session(
        file: Annotated[UploadFile, File(description="The image to analyze.")],
    ) -> dict[str, Any]:
        """Upload one image and start a session over it.

        The upload is decoded once, here, so a file no decoder understands is
        refused now rather than by the first tool that touches it.

        Raises:
            HTTPException: 413 if the upload is larger than this server
                accepts, 422 if it cannot be decoded as an image.
        """
        data = _read_upload(file, app.state.max_upload_bytes)
        name = file.filename or "upload"
        try:
            image = ForensicImage.from_bytes(data, path=Path(name))
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise HTTPException(
                status_code=_UNPROCESSABLE,
                detail=f"{name!r} could not be decoded as an image",
            ) from exc
        session_id = app.state.store.create(image, name=name)
        return _session_info(session_id, app.state.store.get(session_id))

    @app.get("/sessions/{session_id}")
    def read_session(session_id: str) -> dict[str, Any]:
        """The image this session holds, and the tools that have run on it."""
        session = app.state.store.get(session_id)
        return {**_session_info(session_id, session), "tools_run": session.tools_run}

    @app.post("/sessions/{session_id}/tools/{name}")
    def run_tool(
        session_id: str,
        name: str,
        body: Annotated[RunRequest | None, Body()] = None,
    ) -> dict[str, Any]:
        """Run one tool on this session's image and return its result.

        Running the same tool with the same parameters again is free: the
        session hands back what it already computed. Running it with different
        parameters computes again, and the tile URLs below then serve the maps
        of that newer run.

        Returns:
            The result's numbers and details, plus one entry per map it
            carries: how many pyramid levels the map has, their shapes, the
            tile side, and a ``tile_url`` template to fill ``{z}``, ``{x}``
            and ``{y}`` into.
        """
        session = app.state.store.get(session_id)
        parameters = body.parameters if body is not None else {}
        result = session.run(name, parameters)
        pyramids = session.maps(name, parameters)
        return {
            "detector": result.detector,
            "score": result.score,
            "label": result.label,
            "elapsed_ms": result.elapsed_ms,
            "details": to_jsonable(result.details),
            "maps": {
                map_name: _map_info(session_id, name, map_name, pyramid)
                for map_name, pyramid in pyramids.items()
            },
        }

    @app.get(
        "/sessions/{session_id}/maps/{name}/{map_name}/{z}/{x}/{y}.png",
        response_class=Response,
    )
    def map_tile(session_id: str, name: str, map_name: str, z: int, x: int, y: int) -> Response:
        """One square tile of a map, as an 8-bit grayscale PNG.

        ``z`` is the pyramid level (0 is full resolution, the last level is a
        single tile), ``x`` and ``y`` are tile indices across and down. A tile
        that runs off the edge of its level is zero-padded, so every tile a
        viewer receives is the same size.

        Raises:
            HTTPException: 404 if the tool has not run in this session or made
                no such map, 422 if the tile lies outside the level.
        """
        session = app.state.store.get(session_id)
        pyramid = session.latest_maps(name).get(map_name)
        if pyramid is None:
            raise HTTPException(
                status_code=_NOT_FOUND,
                detail=f"Tool {name!r} produced no {map_name!r} map in this session",
            )
        tile = pyramid.tile(z, x, y, size=TILE_SIZE)
        return Response(content=to_png(tile), media_type="image/png")

    @app.get("/sessions/{session_id}/fusion")
    def read_fusion(session_id: str) -> dict[str, Any]:
        """The fused verdict over this session's latest results.

        The payload ``analyze --fuser`` prints: a calibrated probability, the
        abstain band, and what each detector contributed. A detector the fuser
        expects but that has not run here is imputed as abstaining.

        Raises:
            HTTPException: 404 if this server was started without a fuser.
        """
        session = app.state.store.get(session_id)
        return to_jsonable(session.fusion(_require_fuser(app)))

    @app.get("/sessions/{session_id}/report.zip", response_class=Response)
    def read_report(session_id: str) -> Response:
        """The explanation report folder for this session, zipped.

        The folder ``analyze --report-dir`` writes -- ``report.json``,
        ``report.md`` and a heatmap/overlay pair per map -- built into a
        temporary directory that is gone before the response leaves, so the
        server keeps no files and the archive carries names only.
        """
        session = app.state.store.get(session_id)
        with tempfile.TemporaryDirectory() as directory:
            paths = session.report(Path(directory) / "report", app.state.fuser)
            archive = _zip_directory(paths.out_dir)
        return Response(
            content=archive,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{_download_name(session.name)}"'
            },
        )

    return app


def _resolve_fuser(fuser: Fuser | None, fuser_path: Path | None) -> Fuser | None:
    """The fuser a server fuses with: the given object, a given path, or the CLI's default.

    The CLI's resolver is imported inside the function on purpose. The rule --
    an explicit path, then ``$IMGFORENSICS_FUSER``, then ``weights/fuser.json``
    when it exists -- must not drift from the one ``analyze`` follows, and it
    lives there; but an API server has no reason to import typer and rich in
    order to start, and the CLI imports this package lazily in the other
    direction.
    """
    if fuser is not None:
        return fuser
    from imgforensics.cli import _resolve_fuser_path

    path = _resolve_fuser_path(fuser_path)
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Fuser file not found: {path}")
    return Fuser.load(path)


def _require_fuser(app: FastAPI) -> Fuser:
    """The configured fuser, or a 404 saying that there is none.

    Not a 501: whether this server can fuse is a fact about how it was
    started, and the honest answer to "give me the fused verdict" is that
    there is nothing at that address.
    """
    fuser = app.state.fuser
    if fuser is None:
        raise HTTPException(
            status_code=_NOT_FOUND,
            detail=(
                "This server has no fuser configured, so there is no fused verdict. "
                "Start it with 'imgforensics serve --fuser <path>' or set $IMGFORENSICS_FUSER."
            ),
        )
    return fuser


def _read_upload(upload: UploadFile, limit: int) -> bytes:
    """Read an upload, refusing it as soon as it passes ``limit`` bytes.

    Measured while reading rather than taken from the request's declared
    length: that length is the client's claim, and what the limit is about is
    how much this process actually holds.

    Raises:
        HTTPException: 413 once more than ``limit`` bytes have arrived.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := upload.file.read(_UPLOAD_CHUNK_BYTES):
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=_TOO_LARGE,
                detail=f"The upload is larger than this server accepts ({limit} bytes)",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _session_info(session_id: str, session: AnalysisSession) -> dict[str, Any]:
    """The facts about the image that a client shows next to the canvas."""
    return {
        "id": session_id,
        "name": session.name,
        "width": session.image.width,
        "height": session.image.height,
        "format": session.image.format,
    }


def _map_info(session_id: str, tool: str, map_name: str, pyramid: MapPyramid) -> dict[str, Any]:
    """One map, described well enough for a viewer to lay it out and fetch it."""
    return {
        "levels": pyramid.levels,
        "shapes": [list(pyramid.shape(level)) for level in range(pyramid.levels)],
        "tile_url": f"/sessions/{session_id}/maps/{tool}/{map_name}/{{z}}/{{x}}/{{y}}.png",
        "tile_size": TILE_SIZE,
    }


def _zip_directory(directory: Path) -> bytes:
    """Every file under ``directory``, zipped in memory under relative names."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(directory).as_posix())
    return buffer.getvalue()


def _download_name(session_name: str) -> str:
    """A report file name derived from the image's, safe to put in a header.

    The session name came from an upload, so it is reduced to its own stem and
    to the characters a file name may have before it goes back out.
    """
    stem = _UNSAFE_IN_FILENAME.sub("_", Path(session_name).stem).strip("._")
    return f"{stem or 'imgforensics'}_report.zip"


def _detail(exc: BaseException) -> dict[str, str]:
    """An error body carrying the exception's own message and nothing else.

    ``str()`` on a ``KeyError`` repr-quotes it, which would show a client its
    own session id wrapped in quotes, so the first argument is preferred.
    """
    message = str(exc.args[0]) if exc.args else str(exc)
    return {"detail": message}
