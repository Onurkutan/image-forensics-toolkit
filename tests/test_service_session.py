"""Tests for imgforensics.service.session: map pyramids, result caching, and session expiry."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from imgforensics.cli import app
from imgforensics.core import registry
from imgforensics.core.base import BaseDetector
from imgforensics.core.image import ForensicImage
from imgforensics.core.parameters import ParameterSpec
from imgforensics.core.types import DetectionResult
from imgforensics.fusion.stacking import Band, FitInfo, Fuser, FuserMetrics
from imgforensics.service import AnalysisSession, MapPyramid, SessionStore, to_png
from imgforensics.utils.jsonsafe import to_jsonable

runner = CliRunner()

#: A tool that counts its own predictions, so a test can see the cache work.
_COUNTER_NAME = "session_counter"


class _CountingDetector(BaseDetector):
    """Records how often it actually ran, and puts its parameter in its maps."""

    name = _COUNTER_NAME
    runs = 0
    loads = 0

    def __init__(self, level: int = 1, flag: bool = False) -> None:
        self.level = level
        self.flag = flag

    @classmethod
    def parameters(cls) -> list[ParameterSpec]:
        return [
            ParameterSpec(name="level", kind="int", default=1, minimum=0, maximum=3),
            ParameterSpec(name="flag", kind="bool", default=False),
        ]

    @classmethod
    def reset(cls) -> None:
        cls.runs = 0
        cls.loads = 0

    def load(self, device: str = "cpu") -> None:
        type(self).loads += 1

    def predict(self, image: ForensicImage) -> DetectionResult:
        type(self).runs += 1
        heatmap = np.full((image.height, image.width), self.level / 4.0, dtype=np.float32)
        return DetectionResult(
            detector=self.name,
            score=0.5,
            label="uncertain",
            heatmap=heatmap,
            attribution=heatmap / 2.0,
            details={"level": self.level, "flag": self.flag},
        )


registry.register(_COUNTER_NAME)(_CountingDetector)


@pytest.fixture(autouse=True)
def _reset_counter() -> None:
    _CountingDetector.reset()


def _image(size: tuple[int, int] = (24, 16), seed: int = 0) -> ForensicImage:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return ForensicImage.from_pil(Image.fromarray(pixels, mode="RGB"))


def _session(size: tuple[int, int] = (24, 16)) -> AnalysisSession:
    return AnalysisSession(_image(size), name="sample.png")


def _manual_fuser(detectors: list[str]) -> Fuser:
    """A hand-built fuser, so these tests do not depend on fitting one."""
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


# --------------------------------------------------------------------------
# MapPyramid
# --------------------------------------------------------------------------


def test_pyramid_halves_until_the_longest_side_fits() -> None:
    pyramid = MapPyramid(np.zeros((1000, 600), dtype=np.float32))

    assert pyramid.levels == 3
    assert [pyramid.shape(level) for level in range(pyramid.levels)] == [
        (1000, 600),
        (500, 300),
        (250, 150),
    ]


def test_a_small_map_has_one_level() -> None:
    pyramid = MapPyramid(np.zeros((10, 7), dtype=np.float32))

    assert pyramid.levels == 1
    assert pyramid.shape(0) == (10, 7)


def test_an_odd_side_rounds_up_when_it_is_halved() -> None:
    pyramid = MapPyramid(np.zeros((513, 257), dtype=np.float32))

    assert pyramid.shape(1) == (257, 129)
    assert pyramid.shape(2) == (129, 65)


def test_downscaling_averages_rather_than_samples() -> None:
    array = np.zeros((512, 512), dtype=np.float32)
    array[0, 0] = 1.0

    pyramid = MapPyramid(array)

    # A lone bright pixel dims instead of vanishing: 1/4 per halving.
    assert pyramid.level(1)[0, 0] == pytest.approx(0.25)
    assert pyramid.level(1).max() == pytest.approx(0.25)


def test_a_level_outside_the_pyramid_is_refused() -> None:
    pyramid = MapPyramid(np.zeros((8, 8), dtype=np.float32))

    with pytest.raises(ValueError, match="no level 1"):
        pyramid.shape(1)
    with pytest.raises(ValueError, match="no level -1"):
        pyramid.level(-1)


def test_a_full_tile_is_the_region_it_covers() -> None:
    array = np.arange(512 * 512, dtype=np.float32).reshape(512, 512) / (512 * 512)

    tile = MapPyramid(array).tile(0, 1, 0, size=256)

    assert tile.shape == (256, 256)
    assert np.array_equal(tile, array[0:256, 256:512])


def test_an_edge_tile_is_zero_padded() -> None:
    array = np.ones((300, 300), dtype=np.float32)

    tile = MapPyramid(array).tile(0, 1, 1, size=256)

    assert tile.shape == (256, 256)
    assert np.all(tile[:44, :44] == 1.0)
    assert np.all(tile[44:, :] == 0.0)
    assert np.all(tile[:, 44:] == 0.0)


@pytest.mark.parametrize(("x", "y"), [(2, 0), (0, 2), (-1, 0), (0, -1)])
def test_a_tile_outside_the_level_is_refused(x: int, y: int) -> None:
    pyramid = MapPyramid(np.zeros((300, 300), dtype=np.float32))

    with pytest.raises(ValueError, match="outside level 0"):
        pyramid.tile(0, x, y, size=256)


def test_to_png_writes_an_8_bit_grayscale_image() -> None:
    array = np.array([[0.0, 0.5], [1.0, 2.0]], dtype=np.float32)

    with Image.open(io.BytesIO(to_png(array))) as decoded:
        assert decoded.mode == "L"
        assert decoded.size == (2, 2)
        assert np.asarray(decoded).tolist() == [[0, 127], [255, 255]]


# --------------------------------------------------------------------------
# AnalysisSession
# --------------------------------------------------------------------------


def test_running_the_same_tool_twice_reuses_the_result() -> None:
    session = _session()

    first = session.run(_COUNTER_NAME)
    second = session.run(_COUNTER_NAME)

    assert first is second
    assert _CountingDetector.runs == 1
    assert _CountingDetector.loads == 1


def test_the_cache_key_does_not_depend_on_the_order_the_parameters_were_written_in() -> None:
    session = _session()

    session.run(_COUNTER_NAME, {"level": 2, "flag": True})
    session.run(_COUNTER_NAME, {"flag": True, "level": 2})

    assert _CountingDetector.runs == 1


def test_a_call_is_keyed_by_the_coerced_value_not_by_what_was_typed() -> None:
    session = _session()

    session.run(_COUNTER_NAME, {"level": 2})
    session.run(_COUNTER_NAME, {"level": "2", "flag": "false"})

    assert _CountingDetector.runs == 1


def test_different_parameters_are_computed_and_kept_apart() -> None:
    session = _session()

    low = session.run(_COUNTER_NAME, {"level": 1})
    high = session.run(_COUNTER_NAME, {"level": 3})

    assert _CountingDetector.runs == 2
    assert low.details["level"] == 1
    assert high.details["level"] == 3
    assert session.run(_COUNTER_NAME, {"level": 1}) is low


def test_the_declared_default_and_saying_nothing_are_the_same_call() -> None:
    session = _session()

    session.run(_COUNTER_NAME)
    session.run(_COUNTER_NAME, {"level": 1})

    assert _CountingDetector.runs == 1


def test_an_unknown_tool_raises_with_the_registry_message() -> None:
    session = _session()

    with pytest.raises(KeyError, match="No detector registered"):
        session.run("no_such_tool")


def test_a_rejected_parameter_surfaces_before_the_tool_is_built() -> None:
    session = _session()

    with pytest.raises(ValueError, match="at most 3"):
        session.run(_COUNTER_NAME, {"level": 9})
    with pytest.raises(ValueError, match="Unknown parameter"):
        session.run(_COUNTER_NAME, {"lvl": 1})

    assert _CountingDetector.runs == 0


def test_results_are_in_first_run_order_with_the_latest_parameters() -> None:
    session = _session()

    session.run("ela")
    session.run(_COUNTER_NAME, {"level": 1})
    session.run("metadata")
    session.run(_COUNTER_NAME, {"level": 3})

    results = session.results()
    assert [result.detector for result in results] == ["ela", _COUNTER_NAME, "metadata"]
    assert results[1].details["level"] == 3
    assert session.tools_run == ["ela", _COUNTER_NAME, "metadata"]


def test_maps_are_offered_for_the_maps_a_result_carries() -> None:
    session = _session()

    pyramids = session.maps(_COUNTER_NAME)

    assert set(pyramids) == {"heatmap", "attribution"}
    assert pyramids["heatmap"].shape(0) == (16, 24)
    assert session.maps(_COUNTER_NAME)["heatmap"] is pyramids["heatmap"]
    assert _CountingDetector.runs == 1


def test_a_tool_without_a_map_offers_none() -> None:
    session = _session()

    assert session.maps("metadata") == {}


def test_maps_follow_the_parameters_they_were_asked_for() -> None:
    session = _session()

    low = session.maps(_COUNTER_NAME, {"level": 1})["heatmap"]
    high = session.maps(_COUNTER_NAME, {"level": 2})["heatmap"]

    assert low.level(0).max() == pytest.approx(0.25)
    assert high.level(0).max() == pytest.approx(0.5)


def test_fusion_matches_what_analyze_prints_for_the_same_image(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.fromarray(
        np.random.default_rng(1).integers(0, 256, size=(16, 24, 3), dtype=np.uint8), mode="RGB"
    ).save(image_path)
    fuser = _manual_fuser(["ela", "metadata", "copy_move"])
    fuser_path = tmp_path / "fuser.json"
    fuser.save(fuser_path)

    session = AnalysisSession(ForensicImage.from_path(image_path), name=image_path.name)
    session.run("ela")
    session.run("metadata")

    result = runner.invoke(
        app,
        [
            "analyze",
            str(image_path),
            "--json",
            "--detector",
            "ela",
            "--detector",
            "metadata",
            "--fuser",
            str(fuser_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["fusion"] == to_jsonable(session.fusion(fuser))


def test_report_writes_the_same_folder_analyze_does(tmp_path: Path) -> None:
    session = _session()
    session.run("ela")
    session.run(_COUNTER_NAME)

    paths = session.report(tmp_path / "report")

    assert paths.json_path.exists()
    assert paths.markdown_path.exists()
    assert (tmp_path / "report" / "ela_heatmap.png").exists()
    document = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert document["image"]["name"] == "sample.png"
    assert {entry["detector"] for entry in document["detectors"]} == {"ela", _COUNTER_NAME}
    assert "fusion" not in document


def test_report_carries_the_fused_verdict_when_a_fuser_is_given(tmp_path: Path) -> None:
    session = _session()
    session.run("ela")

    paths = session.report(tmp_path / "report", _manual_fuser(["ela", "metadata"]))

    document = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert document["fusion"]["label"] in {"real", "fake", "uncertain"}


# --------------------------------------------------------------------------
# SessionStore
# --------------------------------------------------------------------------


class _FakeClock:
    """A monotonic clock a test can move by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_a_stored_session_comes_back_by_its_id() -> None:
    store = SessionStore()
    image = _image()

    session_id = store.create(image, name="sample.png")

    session = store.get(session_id)
    assert session.image is image
    assert session.name == "sample.png"
    assert len(session_id) == 32


def test_each_session_gets_its_own_id() -> None:
    store = SessionStore()

    ids = {store.create(_image(), name="a.png") for _ in range(3)}

    assert len(ids) == 3


def test_an_unknown_id_is_a_key_error() -> None:
    store = SessionStore()

    with pytest.raises(KeyError, match="never existed or has expired"):
        store.get("0" * 32)


def test_a_session_expires_once_its_ttl_has_passed() -> None:
    clock = _FakeClock()
    store = SessionStore(ttl_seconds=60, clock=clock)
    session_id = store.create(_image(), name="a.png")

    clock.advance(59)
    assert store.get(session_id) is not None

    clock.advance(61)
    with pytest.raises(KeyError):
        store.get(session_id)
    assert len(store) == 0


def test_using_a_session_keeps_it_alive() -> None:
    clock = _FakeClock()
    store = SessionStore(ttl_seconds=60, clock=clock)
    session_id = store.create(_image(), name="a.png")

    for _ in range(4):
        clock.advance(50)
        store.get(session_id)

    assert store.get(session_id) is not None


def test_evict_expired_reports_what_it_dropped() -> None:
    clock = _FakeClock()
    store = SessionStore(ttl_seconds=60, clock=clock)
    old = store.create(_image(), name="old.png")
    clock.advance(30)
    fresh = store.create(_image(), name="fresh.png")
    clock.advance(31)  # the first session has now been idle for 61 seconds

    dropped = store.evict_expired()

    assert dropped == [old]
    assert store.evict_expired() == []
    assert store.get(fresh) is not None


def test_the_least_recently_used_session_is_dropped_at_the_cap() -> None:
    clock = _FakeClock()
    store = SessionStore(max_sessions=2, clock=clock)
    first = store.create(_image(), name="a.png")
    second = store.create(_image(), name="b.png")

    store.get(first)  # first is now the most recently used
    third = store.create(_image(), name="c.png")

    assert len(store) == 2
    assert store.get(first) is not None
    assert store.get(third) is not None
    with pytest.raises(KeyError):
        store.get(second)


@pytest.mark.parametrize("kwargs", [{"ttl_seconds": 0}, {"max_sessions": 0}])
def test_a_store_that_could_hold_nothing_is_refused(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SessionStore(**kwargs)
