"""One image, many tools: the state a workbench keeps between requests.

The CLI's model is "load the image, run everything, print, exit". A workbench's
is the opposite: one image is loaded once and then poked at for as long as the
user is curious -- a tool here, the same tool with a different parameter there,
a zoom into one corner of its map, a report at the end
(``docs/design/01_toolbox_architecture.md``, section 3.1). Nothing in that
sequence should recompute what it already has, and none of it should depend on
a web framework, so it lives here as plain objects.

Three of them:

- :class:`AnalysisSession` holds the image and the results, and is the single
  place a tool is built and run. Its cache is keyed by tool *and parameters*,
  so moving a slider back to where it was is free.
- :class:`MapPyramid` turns one full-resolution map into the levels a viewer
  actually fetches. A 12-megapixel heatmap is 48 MB of float32 and no client
  wants it whole; it wants a thumbnail now and a 256-pixel tile of the region
  the user zoomed into. The report builder's 1,024 px overlay cap is the same
  idea, one step less general.
- :class:`SessionStore` keeps sessions alive for a while and then does not.
  A public demo must bound its memory without a database, so: a TTL, a cap on
  how many sessions exist at once, and an injectable clock so both are
  testable in microseconds.
"""

from __future__ import annotations

import io
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from imgforensics.core import registry
from imgforensics.core.image import ForensicImage
from imgforensics.core.parameters import coerce_parameters
from imgforensics.core.types import DetectionResult
from imgforensics.fusion.explain_report import ReportPaths, build_report
from imgforensics.fusion.report import fusion_payload
from imgforensics.fusion.stacking import Fuser

#: Longest side of the smallest pyramid level, and the default tile size. One
#: level therefore always fits in a single tile, which is the thumbnail a
#: client opens a map with.
MAX_LEVEL_SIDE = 256

#: The map keys a :class:`~imgforensics.core.types.DetectionResult` can carry,
#: in the order a client should offer them: what the tool found, then where it
#: looked.
MAP_NAMES: tuple[str, ...] = ("heatmap", "attribution")

#: Default session lifetime and population cap. Half an hour is longer than
#: anyone stares at one image and short enough that an abandoned tab does not
#: hold a decoded 12-megapixel image plus its maps forever.
DEFAULT_TTL_SECONDS = 1800.0
DEFAULT_MAX_SESSIONS = 32

#: A tool call, identified by name plus the parameters it ran with, sorted so
#: that two callers who spelled the same request in a different dict order
#: share one cache entry.
_CallKey = tuple[str, tuple[tuple[str, object], ...]]


def to_png(array: np.ndarray) -> bytes:
    """Encode a float map in [0, 1] as an 8-bit grayscale PNG.

    The same quantization ``analyze --save-heatmaps`` writes, so a tile
    fetched from a session and a PNG saved from the CLI show the same pixels.

    Args:
        array: A 2-D float array; values outside [0, 1] are clipped.

    Returns:
        The encoded PNG bytes.
    """
    quantized = np.clip(np.asarray(array, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(quantized, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def _halve(level: np.ndarray) -> np.ndarray:
    """One 2x box-average downscale, replicating a final odd row/column first."""
    height, width = level.shape
    if height % 2:
        level = np.concatenate([level, level[-1:, :]], axis=0)
    if width % 2:
        level = np.concatenate([level, level[:, -1:]], axis=1)
    halved = level.reshape(level.shape[0] // 2, 2, level.shape[1] // 2, 2)
    return halved.mean(axis=(1, 3)).astype(np.float32)


class MapPyramid:
    """One map at several resolutions, sliced into tiles on request.

    Level 0 is the map itself; each further level is the previous one box-
    averaged 2x, down to the first level whose longest side is at most
    :data:`MAX_LEVEL_SIDE`. Averaging rather than sampling is what makes a
    zoomed-out level honest: a single bright pixel in a heatmap survives as a
    dimmer pixel instead of disappearing between two sample points, which
    matters when the bright pixel is the finding.

    Building every level costs about a third of the full map again, and the
    levels are built once in the constructor: a session that serves tiles is
    going to ask for them.
    """

    def __init__(self, array: np.ndarray, *, max_side: int = MAX_LEVEL_SIDE) -> None:
        """Build the pyramid over ``array``.

        Args:
            array: A 2-D float map (a heatmap or an attribution map).
            max_side: Longest side of the smallest level.

        Raises:
            ValueError: if ``array`` is not 2-D, or ``max_side`` is below 1.
        """
        if array.ndim != 2:
            raise ValueError(f"a map pyramid needs a 2-D array, got shape {array.shape}")
        if max_side < 1:
            raise ValueError(f"max_side must be at least 1, got {max_side}")
        levels = [np.asarray(array, dtype=np.float32)]
        while max(levels[-1].shape) > max_side:
            levels.append(_halve(levels[-1]))
        self._levels = levels

    @property
    def levels(self) -> int:
        """How many levels there are; valid indices are ``0`` to ``levels - 1``."""
        return len(self._levels)

    def shape(self, level: int = 0) -> tuple[int, int]:
        """``(height, width)`` of one level.

        Raises:
            ValueError: if ``level`` is not one of this pyramid's levels.
        """
        height, width = self.level(level).shape
        return height, width

    def level(self, level: int) -> np.ndarray:
        """One whole level as a float32 array.

        Raises:
            ValueError: if ``level`` is not one of this pyramid's levels.
        """
        if not 0 <= level < len(self._levels):
            raise ValueError(f"no level {level} in a pyramid with {len(self._levels)} level(s)")
        return self._levels[level]

    def tile(self, level: int, x: int, y: int, size: int = MAX_LEVEL_SIDE) -> np.ndarray:
        """One ``size`` x ``size`` tile of a level, addressed by tile index.

        Tile ``(x, y)`` covers the pixels ``[y * size, (y + 1) * size)`` down
        and ``[x * size, (x + 1) * size)`` across. The last tile in a row or
        column is zero-padded to the full square, so every tile a client
        receives has the same shape and a viewer needs no special case at the
        edge -- zero being "nothing found here", which is what lies beyond the
        map anyway.

        Args:
            level: Pyramid level, 0 being full resolution.
            x: Tile column index.
            y: Tile row index.
            size: Tile side in pixels.

        Returns:
            A float32 ``(size, size)`` array.

        Raises:
            ValueError: if ``level`` does not exist, ``size`` is below 1, or
                the tile lies entirely outside the level.
        """
        array = self.level(level)
        if size < 1:
            raise ValueError(f"tile size must be at least 1, got {size}")
        height, width = array.shape
        if x < 0 or y < 0 or x * size >= width or y * size >= height:
            raise ValueError(
                f"tile ({x}, {y}) at size {size} is outside level {level}, "
                f"which is {width}x{height}"
            )
        patch = array[y * size : y * size + size, x * size : x * size + size]
        if patch.shape == (size, size):
            return patch
        tile = np.zeros((size, size), dtype=np.float32)
        tile[: patch.shape[0], : patch.shape[1]] = patch
        return tile


class AnalysisSession:
    """One loaded image, the tools run on it, and the maps they produced.

    A tool is built, loaded and run by :meth:`run`, and the result is kept
    under the parameters it was produced with. Two consequences worth knowing:

    - Running the same tool twice with the same parameters returns the same
      object without recomputing. Running it with *different* parameters
      computes again and keeps both, so a client can compare two settings side
      by side -- but :meth:`results`, the fusion payload and the report use
      only the latest parameters per tool, because that is what the user is
      looking at.
    - Every tool instance is discarded after its run, except in as much as the
      result holds on to what it needs. That keeps a session's memory
      proportional to the maps it is showing rather than to the models it has
      touched, at the cost of reloading a model when its parameters change --
      the trade the other way round would pin a 900 MB localizer per session.
    """

    def __init__(self, image: ForensicImage, *, name: str) -> None:
        """Start a session over one image.

        Args:
            image: The image every tool in this session runs on.
            name: Display name for the image (a file name, typically),
                recorded in the report.
        """
        self.image = image
        self.name = name
        self._results: dict[_CallKey, DetectionResult] = {}
        self._maps: dict[_CallKey, dict[str, MapPyramid]] = {}
        self._order: list[str] = []
        self._latest: dict[str, _CallKey] = {}

    @property
    def tools_run(self) -> list[str]:
        """The tools this session has run, in the order they were first run."""
        return list(self._order)

    def run(self, tool: str, parameters: Mapping[str, object] | None = None) -> DetectionResult:
        """Run one registered tool on this session's image, or return the cached result.

        Args:
            tool: A registry name.
            parameters: What the caller chose, validated against the tool's
                declared specs; anything left out takes its default.

        Returns:
            The tool's :class:`~imgforensics.core.types.DetectionResult`.

        Raises:
            KeyError: if ``tool`` is not registered (the registry's message,
                which lists what is).
            ValueError: if ``parameters`` names something the tool does not
                declare, or carries a value its ParameterSpec rejects (see
                :func:`~imgforensics.core.parameters.coerce_parameters`).
        """
        tool_cls = registry.get(tool)
        resolved = coerce_parameters(tool_cls.parameters(), parameters or {})
        key = self._key(tool, resolved)

        cached = self._results.get(key)
        if cached is None:
            instance = tool_cls(**resolved)
            instance.load()
            cached = instance.run(self.image)
            self._results[key] = cached
            if tool not in self._latest:
                self._order.append(tool)
        self._latest[tool] = key
        return cached

    def results(self) -> list[DetectionResult]:
        """Every tool's latest result, in the order the tools were first run."""
        return [self._results[self._latest[tool]] for tool in self._order]

    def maps(
        self, tool: str, parameters: Mapping[str, object] | None = None
    ) -> dict[str, MapPyramid]:
        """The pyramids for one tool's maps, running it first if needed.

        Args:
            tool: A registry name.
            parameters: As for :meth:`run`.

        Returns:
            ``{"heatmap": ..., "attribution": ...}``, holding only the maps
            this result actually carries -- an empty dict for a tool that
            produced neither.
        """
        self.run(tool, parameters)
        return self._pyramids(self._latest[tool])

    def latest_maps(self, tool: str) -> dict[str, MapPyramid]:
        """The maps of the parameters ``tool`` last ran with, without running anything.

        What a tile request needs. A tile is addressed by tool and map name
        only -- putting the parameters in a tile URL would make every slider
        move invalidate a viewer's whole tile cache -- so the map it serves is
        the one belonging to the run the client was last shown. A tool that
        has not run here has no such map, and saying so is more use to the
        caller than quietly running it with its defaults.

        Raises:
            KeyError: if ``tool`` has not run in this session.
        """
        key = self._latest.get(tool)
        if key is None:
            raise KeyError(f"Tool {tool!r} has not run in this session")
        return self._pyramids(key)

    def _pyramids(self, key: _CallKey) -> dict[str, MapPyramid]:
        """One call's maps as pyramids, built on first request and then kept."""
        pyramids = self._maps.get(key)
        if pyramids is None:
            result = self._results[key]
            pyramids = {
                map_name: MapPyramid(array)
                for map_name in MAP_NAMES
                if (array := getattr(result, map_name)) is not None
            }
            self._maps[key] = pyramids
        return pyramids

    def fusion(self, fuser: Fuser) -> dict[str, Any]:
        """The fused verdict over this session's latest results.

        Identical to what ``analyze --fuser`` prints, because it is the same
        function (:func:`~imgforensics.fusion.report.fusion_payload`): a tool
        the fuser expects but that has not been run in this session is imputed
        as abstaining, exactly as a detector that did not run is on the CLI.
        """
        return fusion_payload(fuser, self.results())

    def report(self, out_dir: Path, fuser: Fuser | None = None) -> ReportPaths:
        """Build the explanation report folder for this session's latest results.

        The same folder ``analyze --report-dir`` writes -- ``report.json``,
        ``report.md``, and a heatmap/overlay pair per map.
        """
        return build_report(
            self.image,
            self.results(),
            fuser=fuser,
            out_dir=out_dir,
            source_name=self.name,
        )

    @staticmethod
    def _key(tool: str, parameters: Mapping[str, object]) -> _CallKey:
        """A cache key that depends on the parameter values, not on their order."""
        return tool, tuple(sorted(parameters.items()))


@dataclass
class _Entry:
    """One stored session and when it was last touched."""

    session: AnalysisSession
    last_used: float


class SessionStore:
    """In-memory sessions with a lifetime and a population cap.

    Two limits, because they fail differently: the TTL drops a session nobody
    came back to, and the cap drops the least recently used one when a burst of
    new uploads would otherwise outrun the TTL. Both are enforced when a
    session is created, so a store that is never written to never sweeps.

    Every method takes a lock: an API server touches this from several request
    threads at once, and the interesting operations are read-modify-write.
    The lock is held only around the dictionary, never around a tool run --
    :meth:`get` hands back the session and gets out of the way, so one long
    localizer run does not block another request from finding its session.
    Two threads that run tools in the *same* session still need to coordinate
    themselves; the session is not itself thread-safe, and a client with one
    session per user does not need it to be.
    """

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Configure the store.

        Args:
            ttl_seconds: How long a session survives after its last use.
            max_sessions: How many sessions may exist at once.
            clock: Source of monotonic time, injectable so a test can move
                time by an hour without waiting one.

        Raises:
            ValueError: if ``ttl_seconds`` or ``max_sessions`` is not positive.
        """
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds}")
        if max_sessions < 1:
            raise ValueError(f"max_sessions must be at least 1, got {max_sessions}")
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._clock = clock
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        """How many sessions are currently stored, expired ones included."""
        with self._lock:
            return len(self._entries)

    def create(self, image: ForensicImage, *, name: str) -> str:
        """Store a new session over ``image`` and return its id.

        Expired sessions are dropped first, and the least recently used one is
        dropped if the store is still full afterwards.

        Returns:
            A uuid4 hex id, the handle every later call uses.
        """
        session = AnalysisSession(image, name=name)
        session_id = uuid.uuid4().hex
        with self._lock:
            self._evict_expired_locked()
            while len(self._entries) >= self.max_sessions:
                self._entries.popitem(last=False)
            self._entries[session_id] = _Entry(session=session, last_used=self._clock())
        return session_id

    def get(self, session_id: str) -> AnalysisSession:
        """Return a stored session and mark it as used just now.

        Raises:
            KeyError: if no such session exists, or it has expired -- the two
                are one case on purpose, since a caller can do nothing
                different about them.
        """
        with self._lock:
            self._evict_expired_locked()
            entry = self._entries.get(session_id)
            if entry is None:
                raise KeyError(f"No session {session_id!r}: it never existed or has expired")
            entry.last_used = self._clock()
            self._entries.move_to_end(session_id)
            return entry.session

    def evict_expired(self) -> list[str]:
        """Drop every session past its TTL and return the dropped ids."""
        with self._lock:
            return self._evict_expired_locked()

    def _evict_expired_locked(self) -> list[str]:
        deadline = self._clock() - self.ttl_seconds
        expired = [
            session_id for session_id, entry in self._entries.items() if entry.last_used <= deadline
        ]
        for session_id in expired:
            del self._entries[session_id]
        return expired
