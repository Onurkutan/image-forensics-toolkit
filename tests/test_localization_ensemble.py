"""Tests for the registered localizer ensemble (optional ``ml`` extra).

Offline and CPU-only. The ensemble itself never touches torch -- it only
calls its members -- so every test here registers trivial stand-in localizers
that return hand-chosen heatmaps and combines *those*: two constant maps, and
a ``dino_inpaint``-shaped one that is confident inside one region and near
zero everywhere else, which is what a specialist member contributes. What is
under test is the wrapper: the three combination modes' arithmetic with two
and three members, the resampling of a member map that does not match the
image, the abstention contract, the details, and the mode's configuration
precedence. Whether the real members agree on a given image is a benchmark
question, not a unit-test one; the one test that builds the real default
members points every weight directory at an empty folder, so nothing is ever
downloaded or loaded.

The module still carries the ``ml`` marker and skips without torch, because
``localizer_ensemble`` is only registered where its real members are.
CUDA is hidden so that a machine with a GPU runs these on the CPU like any
other.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import natural_like_image
from typer.testing import CliRunner

torch = pytest.importorskip("torch")

from imgforensics.cli import app  # noqa: E402
from imgforensics.core import registry  # noqa: E402
from imgforensics.core.base import BaseDetector  # noqa: E402
from imgforensics.core.image import ForensicImage  # noqa: E402
from imgforensics.core.registry import register  # noqa: E402
from imgforensics.core.types import DetectionResult, label_from_score  # noqa: E402
from imgforensics.localization._scoring import top_fraction_score  # noqa: E402
from imgforensics.localization.ensemble import (  # noqa: E402
    DEFAULT_MEMBERS,
    MODE_ENV,
    LocalizerEnsemble,
    combine,
    percentile_ranks,
    resize_heatmap,
    resolve_mode,
)

pytestmark = pytest.mark.ml

runner = CliRunner()

_SIZE = (8, 6)  # (width, height): small enough to write expected maps out by hand


class _FixedLocalizer(BaseDetector):
    """A localizer whose heatmap is a fixed value, or ``None`` to abstain.

    ``scale`` shrinks the returned map relative to the image, which is how the
    ensemble's resampling path is reached without a model that predicts at a
    reduced resolution.
    """

    value: float = 0.2
    scale: int = 1
    abstains: bool = False
    is_loaded: bool = True

    def predict(self, image: ForensicImage) -> DetectionResult:
        if self.abstains:
            return DetectionResult(
                detector=self.name, score=0.5, label="uncertain", details={"reason": "no weights"}
            )
        shape = (image.height // self.scale, image.width // self.scale)
        heatmap = np.full(shape, self.value, dtype=np.float32)
        score = top_fraction_score(heatmap)
        return DetectionResult(
            detector=self.name,
            score=score,
            label=label_from_score(score),
            heatmap=heatmap,
            details={"value": self.value},
        )


@register("fake_localizer_low")
class _LowLocalizer(_FixedLocalizer):
    name = "fake_localizer_low"
    value = 0.2


@register("fake_localizer_high")
class _HighLocalizer(_FixedLocalizer):
    name = "fake_localizer_high"
    value = 0.8


@register("fake_localizer_half")
class _HalfResolutionLocalizer(_FixedLocalizer):
    name = "fake_localizer_half"
    value = 0.6
    scale = 2


@register("fake_localizer_abstaining")
class _AbstainingLocalizer(_FixedLocalizer):
    name = "fake_localizer_abstaining"
    abstains = True


@register("fake_localizer_unweighted")
class _UnweightedLocalizer(_FixedLocalizer):
    """Stands in for a member whose weights were never downloaded."""

    name = "fake_localizer_unweighted"
    is_loaded = False


#: The region the dino_inpaint-shaped stand-in is confident about, as
#: (top, bottom, left, right) on the 6 x 8 test image: 3 x 4 = 12 of 48 pixels.
_REGION = (1, 4, 2, 6)
_REGION_VALUE = 0.9
_BACKGROUND_VALUE = 0.05


class _RegionLocalizer(_FixedLocalizer):
    """Shaped like ``dino_inpaint``: confident about one region, near zero elsewhere.

    A splicing localizer on a fully regenerated image returns a flat, low map;
    the inpainting localizer returns a high patch where the regeneration is.
    That asymmetry is what the combination modes treat differently, so it is
    what the three-member tests are built on.
    """

    def predict(self, image: ForensicImage) -> DetectionResult:
        if self.abstains:
            return super().predict(image)
        top, bottom, left, right = _REGION
        heatmap = np.full((image.height, image.width), _BACKGROUND_VALUE, dtype=np.float32)
        heatmap[top:bottom, left:right] = _REGION_VALUE
        score = top_fraction_score(heatmap)
        return DetectionResult(
            detector=self.name,
            score=score,
            label=label_from_score(score),
            heatmap=heatmap,
            details={},
        )


@register("fake_localizer_inpaint")
class _InpaintLocalizer(_RegionLocalizer):
    name = "fake_localizer_inpaint"


@register("fake_localizer_inpaint_no_checkpoint")
class _InpaintWithoutCheckpoint(_RegionLocalizer):
    """``dino_inpaint`` with no trained checkpoint installed: dropped at load time."""

    name = "fake_localizer_inpaint_no_checkpoint"
    is_loaded = False


@register("fake_localizer_inpaint_abstaining")
class _InpaintAbstaining(_RegionLocalizer):
    """``dino_inpaint`` loaded, but returning no heatmap for this image."""

    name = "fake_localizer_inpaint_abstaining"
    abstains = True


@register("fake_localizer_blind")
class _BlindLocalizer(_FixedLocalizer):
    """A splicing localizer looking at an edit with no seam: flat and low."""

    name = "fake_localizer_blind"
    value = _BACKGROUND_VALUE


_FAKE_MEMBERS = ("fake_localizer_low", "fake_localizer_high")
_THREE_MEMBERS = ("fake_localizer_low", "fake_localizer_high", "fake_localizer_inpaint")


def _region_mask() -> np.ndarray:
    top, bottom, left, right = _REGION
    mask = np.zeros((_SIZE[1], _SIZE[0]), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


@pytest.fixture(autouse=True)
def hide_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve ``device="auto"`` to the CPU even on a machine that has a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture(autouse=True)
def clear_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a mode exported in the developer's own shell out of these tests."""
    monkeypatch.delenv(MODE_ENV, raising=False)


def _forensic_image(size: tuple[int, int] = _SIZE, seed: int = 3) -> ForensicImage:
    buffer = io.BytesIO()
    natural_like_image(size=size, seed=seed).save(buffer, format="PNG")
    return ForensicImage.from_bytes(buffer.getvalue())


def _predict(**kwargs: Any) -> DetectionResult:
    kwargs.setdefault("members", _FAKE_MEMBERS)
    ensemble = LocalizerEnsemble(**kwargs)
    ensemble.load("cpu")
    return ensemble.predict(_forensic_image())


# ---------------------------------------------------------------------------
# Registration and configuration
# ---------------------------------------------------------------------------


def test_registry_exposes_the_ensemble_when_torch_is_installed() -> None:
    assert "localizer_ensemble" in registry.available()
    assert registry.get("localizer_ensemble") is LocalizerEnsemble


def test_the_default_members_are_the_three_real_localizers_in_order() -> None:
    assert DEFAULT_MEMBERS == ("catnet_v2", "iml_vit", "dino_inpaint")
    assert LocalizerEnsemble().members == DEFAULT_MEMBERS
    assert all(name in registry.available() for name in DEFAULT_MEMBERS)


def test_the_real_default_members_are_dropped_cleanly_without_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real three-member default, with every weight directory empty.

    Each member's own ``load()`` only checks for its file, so this builds the
    real ``catnet_v2``, ``iml_vit`` and ``dino_inpaint`` wrappers without
    downloading or loading a single weight -- and proves that the new member
    fails the same way the other two do: dropped at load time and named in the
    abstention, never an exception.
    """
    monkeypatch.setenv("IMGFORENSICS_WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("IMGFORENSICS_INPAINT_DIR", str(tmp_path / "no_checkpoint"))

    ensemble = LocalizerEnsemble()
    ensemble.load("cpu")

    assert ensemble.loaded_members == []
    result = ensemble.predict(_forensic_image())
    assert result.score == 0.5
    assert result.label == "uncertain"
    assert result.heatmap is None
    assert result.details["members"] == list(DEFAULT_MEMBERS)
    for name in DEFAULT_MEMBERS:
        assert name in result.details["reason"]


def test_the_default_mode_is_max_and_the_env_var_overrides_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert LocalizerEnsemble(members=_FAKE_MEMBERS).mode == "max"

    monkeypatch.setenv(MODE_ENV, "rank_mean")
    assert LocalizerEnsemble(members=_FAKE_MEMBERS).mode == "rank_mean"
    # An explicit argument still wins over the environment.
    assert LocalizerEnsemble(members=_FAKE_MEMBERS, mode="mean").mode == "mean"


def test_an_unknown_mode_is_rejected_from_either_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="median"):
        LocalizerEnsemble(members=_FAKE_MEMBERS, mode="median")

    monkeypatch.setenv(MODE_ENV, "median")
    with pytest.raises(ValueError, match=MODE_ENV):
        LocalizerEnsemble(members=_FAKE_MEMBERS)

    assert resolve_mode("mean") == "mean"


# ---------------------------------------------------------------------------
# Combination arithmetic
# ---------------------------------------------------------------------------


def test_mean_mode_averages_the_member_maps() -> None:
    result = _predict(mode="mean")

    heatmap = result.heatmap
    assert heatmap is not None
    assert heatmap.shape == (_SIZE[1], _SIZE[0])
    assert heatmap.dtype == np.float32
    np.testing.assert_allclose(heatmap, 0.5, atol=1e-6)
    assert result.score == pytest.approx(0.5, abs=1e-6)


def test_max_mode_takes_the_more_confident_member() -> None:
    result = _predict(mode="max")

    heatmap = result.heatmap
    assert heatmap is not None
    np.testing.assert_allclose(heatmap, 0.8, atol=1e-6)
    assert result.label == "fake"


def test_combine_computes_mean_and_max_pixelwise() -> None:
    a = np.array([[0.0, 0.25], [1.0, 0.5]], dtype=np.float32)
    b = np.array([[1.0, 0.75], [0.0, 0.5]], dtype=np.float32)

    np.testing.assert_allclose(
        combine([a, b], "mean"), np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        combine([a, b], "max"), np.array([[1.0, 0.75], [1.0, 0.5]], dtype=np.float32)
    )


def test_combine_rejects_an_empty_list_and_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="at least one heatmap"):
        combine([], "mean")
    with pytest.raises(ValueError, match="median"):
        combine([np.zeros((2, 2), dtype=np.float32)], "median")


def test_percentile_ranks_spread_distinct_values_evenly() -> None:
    ranks = percentile_ranks(np.array([[0.1, 0.9], [0.4, 0.7]], dtype=np.float32))

    # Four distinct values -> sorted positions 0..3, divided by 3.
    np.testing.assert_allclose(
        ranks, np.array([[0.0, 1.0], [1 / 3, 2 / 3]], dtype=np.float64), atol=1e-12
    )


def test_percentile_ranks_average_the_ranks_of_tied_values() -> None:
    # Positions 0, 1, 2 are tied -> each gets (0 + 1 + 2) / 3 = 1; positions 3
    # and 4 are tied -> each gets 3.5. Divided by (5 - 1).
    ranks = percentile_ranks(np.array([0.2, 0.9, 0.2, 0.9, 0.2], dtype=np.float32))

    np.testing.assert_allclose(ranks, [0.25, 0.875, 0.25, 0.875, 0.25], atol=1e-12)


def test_percentile_ranks_of_a_constant_map_are_one_half() -> None:
    np.testing.assert_allclose(percentile_ranks(np.full((3, 4), 0.7)), 0.5)
    np.testing.assert_allclose(percentile_ranks(np.array([[0.42]])), 0.5)


def test_rank_mean_discards_the_members_absolute_scale() -> None:
    # One member's map is the other's, shifted down by 0.5: the two rank maps
    # are therefore identical, and so is their mean.
    low = np.array([[0.00, 0.10], [0.20, 0.30]], dtype=np.float32)
    high = low + 0.5

    combined = combine([low, high], "rank_mean")

    np.testing.assert_allclose(
        combined, np.array([[0.0, 1 / 3], [2 / 3, 1.0]], dtype=np.float32), atol=1e-6
    )
    # The same two maps under "mean" keep the offset the ranks threw away.
    np.testing.assert_allclose(combine([low, high], "mean"), low + 0.25, atol=1e-6)


def test_rank_mean_marks_half_the_image_above_the_fixed_threshold() -> None:
    """The caveat the docstring warns about, asserted rather than only stated."""
    rng = np.random.default_rng(0)
    faint = rng.uniform(0.0, 0.05, size=(40, 40)).astype(np.float32)

    combined = combine([faint], "rank_mean")

    assert float((faint > 0.5).mean()) == 0.0  # nothing is manipulated
    assert float((combined > 0.5).mean()) == pytest.approx(0.5, abs=0.02)


def test_the_ensemble_runs_rank_mean_end_to_end() -> None:
    result = _predict(mode="rank_mean")

    heatmap = result.heatmap
    assert heatmap is not None
    # Both members are constant maps, so every pixel ties: rank 0.5 throughout.
    np.testing.assert_allclose(heatmap, 0.5, atol=1e-6)
    assert result.details["mode"] == "rank_mean"


# ---------------------------------------------------------------------------
# Three members: two constant maps and a dino_inpaint-shaped one
# ---------------------------------------------------------------------------


def test_three_members_under_max_take_the_specialist_inside_its_region() -> None:
    result = _predict(members=_THREE_MEMBERS, mode="max")

    heatmap = result.heatmap
    assert heatmap is not None
    region = _region_mask()
    # Inside the region the inpainting member (0.9) beats the high one (0.8);
    # outside it the high member wins over the specialist's near-zero map.
    np.testing.assert_allclose(heatmap[region], _REGION_VALUE, atol=1e-6)
    np.testing.assert_allclose(heatmap[~region], 0.8, atol=1e-6)
    assert result.details["mode"] == "max"
    assert result.details["members"] == list(_THREE_MEMBERS)
    assert set(result.details["member_scores"]) == set(_THREE_MEMBERS)


def test_three_members_under_mean_average_all_three() -> None:
    result = _predict(members=_THREE_MEMBERS, mode="mean")

    heatmap = result.heatmap
    assert heatmap is not None
    region = _region_mask()
    np.testing.assert_allclose(heatmap[region], (0.2 + 0.8 + _REGION_VALUE) / 3, atol=1e-6)
    np.testing.assert_allclose(heatmap[~region], (0.2 + 0.8 + _BACKGROUND_VALUE) / 3, atol=1e-6)
    assert result.details["members"] == list(_THREE_MEMBERS)


def test_three_members_under_rank_mean_average_the_members_ranks() -> None:
    result = _predict(members=_THREE_MEMBERS, mode="rank_mean")

    heatmap = result.heatmap
    assert heatmap is not None
    region = _region_mask()
    # The two constant members tie everywhere -> rank 0.5. The region map has
    # two values: its 36 background pixels share positions 0..35 (mean 17.5)
    # and its 12 region pixels positions 36..47 (mean 41.5), over 48 - 1.
    background_rank, region_rank = 17.5 / 47, 41.5 / 47
    np.testing.assert_allclose(heatmap[region], (0.5 + 0.5 + region_rank) / 3, atol=1e-6)
    np.testing.assert_allclose(heatmap[~region], (0.5 + 0.5 + background_rank) / 3, atol=1e-6)
    assert result.details["mode"] == "rank_mean"


def test_max_keeps_a_region_only_the_specialist_sees_and_mean_dilutes_it() -> None:
    """Why ``max`` is the rule a specialist member needs.

    On a regenerated edit the two splicing localizers are flat and low; only
    the inpainting localizer sees the region. The maximum keeps it above the
    0.5 threshold exactly where it is, while the mean divides it by three and
    drops every pixel of it below the threshold.
    """
    members = ("fake_localizer_blind", "fake_localizer_blind", "fake_localizer_inpaint")
    region = _region_mask()

    under_max = _predict(members=members, mode="max").heatmap
    under_mean = _predict(members=members, mode="mean").heatmap

    assert under_max is not None and under_mean is not None
    assert np.array_equal(under_max > 0.5, region)
    assert not (under_mean > 0.5).any()


def test_a_specialist_without_a_checkpoint_leaves_the_two_member_result() -> None:
    three = _predict(members=[*_FAKE_MEMBERS, "fake_localizer_inpaint_no_checkpoint"], mode="mean")
    two = _predict(members=_FAKE_MEMBERS, mode="mean")

    assert three.heatmap is not None and two.heatmap is not None
    np.testing.assert_allclose(three.heatmap, two.heatmap, atol=1e-6)
    # Dropped at load time: it never ran, so it has no score on the record.
    assert three.details["members"] == list(_FAKE_MEMBERS)
    assert "fake_localizer_inpaint_no_checkpoint" not in three.details["member_scores"]


def test_a_specialist_that_abstains_is_on_the_record_but_not_in_the_map() -> None:
    result = _predict(members=[*_FAKE_MEMBERS, "fake_localizer_inpaint_abstaining"], mode="max")

    assert result.heatmap is not None
    np.testing.assert_allclose(result.heatmap, 0.8, atol=1e-6)
    assert result.details["members"] == list(_FAKE_MEMBERS)
    # It ran, so its abstaining score and its cost stay visible.
    assert result.details["member_scores"]["fake_localizer_inpaint_abstaining"] == 0.5
    assert "fake_localizer_inpaint_abstaining" in result.details["member_elapsed_ms"]


def test_the_ensemble_abstains_when_every_member_including_the_specialist_is_out() -> None:
    result = _predict(
        members=[
            "fake_localizer_unweighted",
            "fake_localizer_abstaining",
            "fake_localizer_inpaint_no_checkpoint",
        ]
    )

    assert result.heatmap is None
    assert result.score == 0.5
    for name in (
        "fake_localizer_unweighted",
        "fake_localizer_abstaining",
        "fake_localizer_inpaint_no_checkpoint",
    ):
        assert name in result.details["reason"]


# ---------------------------------------------------------------------------
# Resampling, abstention and details
# ---------------------------------------------------------------------------


def test_resize_heatmap_is_a_no_op_at_the_right_shape() -> None:
    heatmap = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)

    resized = resize_heatmap(heatmap, (3, 4))

    assert resized is heatmap


def test_resize_heatmap_upsamples_bilinearly() -> None:
    heatmap = np.array([[0.0, 1.0]], dtype=np.float32)

    resized = resize_heatmap(heatmap, (2, 4))

    assert resized.shape == (2, 4)
    assert resized.dtype == np.float32
    # Bilinear upsampling keeps the endpoints and stays monotone between them.
    assert float(resized.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(resized.max()) == pytest.approx(1.0, abs=1e-6)
    assert list(resized[0]) == sorted(resized[0])


def test_a_smaller_member_map_is_resampled_to_the_image_shape() -> None:
    result = _predict(members=["fake_localizer_low", "fake_localizer_half"], mode="mean")

    heatmap = result.heatmap
    assert heatmap is not None
    assert heatmap.shape == (_SIZE[1], _SIZE[0])
    # The half-resolution member is a constant 0.6, so resampling leaves it at
    # 0.6 and the mean with the 0.2 member is 0.4 everywhere.
    np.testing.assert_allclose(heatmap, 0.4, atol=1e-6)
    assert result.details["members"] == ["fake_localizer_low", "fake_localizer_half"]


def test_an_abstaining_member_is_dropped_from_the_combination() -> None:
    result = _predict(members=["fake_localizer_abstaining", "fake_localizer_high"])

    heatmap = result.heatmap
    assert heatmap is not None
    np.testing.assert_allclose(heatmap, 0.8, atol=1e-6)
    # It ran, so its cost and its abstaining score stay on the record even
    # though it contributed no pixels.
    assert result.details["members"] == ["fake_localizer_high"]
    assert result.details["member_scores"]["fake_localizer_abstaining"] == 0.5
    assert set(result.details["member_elapsed_ms"]) == {
        "fake_localizer_abstaining",
        "fake_localizer_high",
    }


def test_a_member_without_weights_is_dropped_at_load_time() -> None:
    ensemble = LocalizerEnsemble(members=["fake_localizer_unweighted", "fake_localizer_low"])
    ensemble.load("cpu")

    assert ensemble.loaded_members == ["fake_localizer_low"]
    result = ensemble.predict(_forensic_image())
    assert result.details["members"] == ["fake_localizer_low"]
    assert "fake_localizer_unweighted" not in result.details["member_scores"]


def test_the_ensemble_abstains_when_no_member_produces_a_heatmap() -> None:
    result = _predict(members=["fake_localizer_unweighted", "fake_localizer_abstaining"])

    assert result.score == 0.5
    assert result.label == "uncertain"
    assert result.heatmap is None
    assert result.details["members"] == [
        "fake_localizer_unweighted",
        "fake_localizer_abstaining",
    ]
    assert "fake_localizer_unweighted" in result.details["reason"]
    assert "fake_localizer_abstaining" in result.details["reason"]


def test_predict_loads_lazily_when_load_was_never_called() -> None:
    result = LocalizerEnsemble(members=_FAKE_MEMBERS).predict(_forensic_image())

    assert result.heatmap is not None
    assert result.details["members"] == list(_FAKE_MEMBERS)


def test_details_summarize_the_combined_map() -> None:
    result = _predict(mode="max")

    heatmap = result.heatmap
    assert heatmap is not None
    details = result.details
    assert details["mode"] == "max"
    assert details["members"] == list(_FAKE_MEMBERS)
    assert details["member_scores"] == {"fake_localizer_low": 0.2, "fake_localizer_high": 0.8}
    assert all(ms >= 0.0 for ms in details["member_elapsed_ms"].values())
    assert details["max_prob"] == pytest.approx(float(heatmap.max()), abs=1e-4)
    assert details["mean_prob"] == pytest.approx(float(heatmap.mean()), abs=1e-4)
    assert details["area_fraction_above_0.5"] == pytest.approx(1.0, abs=1e-4)
    assert details["device"] == "cpu"


# ---------------------------------------------------------------------------
# The shared score rule
# ---------------------------------------------------------------------------


def test_top_fraction_score_matches_the_members_own_rule() -> None:
    from imgforensics.localization.catnet import CATNetLocalizer
    from imgforensics.localization.iml_vit import IMLViTLocalizer

    heatmap = np.random.default_rng(7).uniform(0.0, 1.0, size=(120, 90)).astype(np.float32)

    flat = np.sort(heatmap.reshape(-1))
    keep = max(1, int(round(flat.size * 0.01)))
    expected = float(flat[-keep:].mean())

    assert top_fraction_score(heatmap) == pytest.approx(expected, abs=1e-6)
    assert IMLViTLocalizer._score_from(heatmap) == pytest.approx(expected, abs=1e-6)
    assert CATNetLocalizer._score_from(heatmap) == pytest.approx(expected, abs=1e-6)


def test_top_fraction_score_keeps_at_least_one_pixel() -> None:
    heatmap = np.array([[0.0, 0.0], [0.0, 0.9]], dtype=np.float32)

    # 1% of four pixels rounds to zero, so the floor of one pixel decides.
    assert top_fraction_score(heatmap) == pytest.approx(0.9, abs=1e-6)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_analyze_runs_the_ensemble_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "imgforensics.localization.ensemble.DEFAULT_MEMBERS", _FAKE_MEMBERS, raising=True
    )
    image_path = tmp_path / "sample.png"
    natural_like_image(size=_SIZE, seed=12).save(image_path, format="PNG")
    heatmap_dir = tmp_path / "heatmaps"

    result = runner.invoke(
        app,
        [
            "analyze",
            str(image_path),
            "--json",
            "--detector",
            "localizer_ensemble",
            "--save-heatmaps",
            str(heatmap_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout
    entry: dict[str, Any] = json.loads(result.stdout)["results"][0]
    assert entry["detector"] == "localizer_ensemble"
    assert entry["details"]["mode"] == "max"
    assert entry["details"]["members"] == list(_FAKE_MEMBERS)
    # The default mode is max: the higher of the two constant member maps wins.
    assert entry["score"] == pytest.approx(0.8, abs=1e-6)
    assert entry["heatmap"] == str(heatmap_dir / "sample_localizer_ensemble.png")
