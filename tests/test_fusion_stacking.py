"""Tests for imgforensics.fusion.stacking: FusionFeatures and fit_fuser.

Uses ``conftest.synthetic_fusion_records`` -- 400 synthetic images (200 real,
200 fake) scored by three detectors that exercise the three cases a
stacking fuser has to handle: an informative detector (correlates with the
true label), an inverted one (anti-correlates), and one that always
abstains at 0.5 (carries no information). No image, model, or file I/O is
needed to build these records, so this whole module runs fast and without
the ``ml`` extra.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import synthetic_fusion_records

from imgforensics.eval.records import ScoreRecord
from imgforensics.fusion.stacking import (
    Band,
    BandFit,
    FitInfo,
    Fuser,
    FuserMetrics,
    FusionFeatures,
    _fit_band,
    _min_outside_floor,
    fit_fuser,
    logit_from_score,
)


def test_logit_from_score_maps_abstain_to_zero() -> None:
    assert logit_from_score(0.5) == 0.0
    assert logit_from_score(0.9) > 0.0
    assert logit_from_score(0.1) < 0.0
    # Clipped, not raising, at the extremes.
    assert np.isfinite(logit_from_score(0.0))
    assert np.isfinite(logit_from_score(1.0))


def test_fusion_features_imputes_missing_detector() -> None:
    records = [
        ScoreRecord(
            entry_path="a.png",
            label="fake",
            source="s",
            generator=None,
            split=None,
            level="clean",
            detector="d1",
            score=0.9,
            elapsed_ms=None,
        ),
        ScoreRecord(
            entry_path="a.png",
            label="fake",
            source="s",
            generator=None,
            split=None,
            level="clean",
            detector="d2",
            score=0.8,
            elapsed_ms=None,
        ),
        ScoreRecord(
            entry_path="b.png",
            label="real",
            source="s",
            generator=None,
            split=None,
            level="clean",
            detector="d1",
            score=0.1,
            elapsed_ms=None,
        ),
        # "d2" is missing entirely for b.png.
    ]
    features = FusionFeatures.build(records, detectors=["d1", "d2"])

    row_a = features.entry_paths.index("a.png")
    row_b = features.entry_paths.index("b.png")
    n = features.n_detectors
    assert n == 2

    # Missing detector: imputed logit(0.5) == 0, presence indicator 0.
    assert features.X[row_b, 1] == 0.0
    assert features.X[row_b, n + 1] == 0.0
    # Present detector: real logit, presence indicator 1.
    assert features.X[row_a, 1] == pytest.approx(logit_from_score(0.8))
    assert features.X[row_a, n + 1] == 1.0
    assert features.y[row_a] == 1.0
    assert features.y[row_b] == 0.0


def test_fusion_features_default_levels_include_every_level_present() -> None:
    records = synthetic_fusion_records(n_per_class=5)
    perturbed = [
        ScoreRecord(
            entry_path=r.entry_path,
            label=r.label,
            source=r.source,
            generator=r.generator,
            split=r.split,
            level="jpeg_q50",
            detector=r.detector,
            score=r.score,
            elapsed_ms=None,
        )
        for r in records
    ]
    features = FusionFeatures.build(records + perturbed)
    assert set(features.levels_used) == {"clean", "jpeg_q50"}
    # Each image contributes one row per level it appears at.
    assert len(features.entry_paths) == 2 * len({r.entry_path for r in records})


def test_fusion_features_rejects_unknown_level() -> None:
    records = synthetic_fusion_records(n_per_class=5)
    with pytest.raises(ValueError, match="not present in records"):
        FusionFeatures.build(records, levels=["does-not-exist"])


def test_fusion_features_rejects_empty_records() -> None:
    with pytest.raises(ValueError, match="no records"):
        FusionFeatures.build([])


def test_default_detectors_excludes_trivial_baselines() -> None:
    records = synthetic_fusion_records(n_per_class=5)
    records += [
        ScoreRecord(
            entry_path="synthetic/real/0000.png",
            label="real",
            source="synthetic",
            generator=None,
            split=None,
            level="clean",
            detector=name,
            score=0.5,
            elapsed_ms=None,
        )
        for name in ("constant_real", "constant_fake", "random", "signals_mean")
    ]
    features = FusionFeatures.build(records)
    assert set(features.detectors) == {"informative", "inverted", "abstaining"}


def test_fit_fuser_learns_correct_signs_and_generalizes() -> None:
    records = synthetic_fusion_records(n_per_class=200, seed=0)
    fuser = fit_fuser(records, seed=0)

    assert set(fuser.detectors) == {"informative", "inverted", "abstaining"}
    index = {name: i for i, name in enumerate(fuser.detectors)}

    assert fuser.logit_weights[index["informative"]] > 0
    assert fuser.logit_weights[index["inverted"]] < 0
    # The abstaining detector's logit column is identically zero (logit(0.5)
    # == 0 for every row), so gradient descent starting from zero never
    # moves its weight at all.
    assert fuser.logit_weights[index["abstaining"]] == pytest.approx(0.0, abs=1e-9)

    assert fuser.metrics.holdout_auc > 0.95
    assert fuser.metrics.train_auc > 0.95
    assert 0.0 <= fuser.metrics.abstain_rate <= 1.0
    assert 0.0 <= fuser.metrics.outside_band_balanced_accuracy <= 1.0
    assert fuser.fit_info.n_images == 400
    assert fuser.fit_info.n_fake == 200
    assert fuser.fit_info.n_real == 200
    assert fuser.fit_info.sources == ["synthetic"]
    assert fuser.fit_info.levels == ["clean"]


def test_fit_fuser_predicts_high_probability_for_clear_fake() -> None:
    records = synthetic_fusion_records(n_per_class=200, seed=0)
    fuser = fit_fuser(records, seed=0)

    fake_probability = fuser.predict({"informative": 0.98, "inverted": 0.02, "abstaining": 0.5})
    real_probability = fuser.predict({"informative": 0.02, "inverted": 0.98, "abstaining": 0.5})
    assert fake_probability > 0.9
    assert real_probability < 0.1
    assert fuser.predict_label({"informative": 0.98, "inverted": 0.02, "abstaining": 0.5}) == "fake"
    assert fuser.predict_label({"informative": 0.02, "inverted": 0.98, "abstaining": 0.5}) == "real"


def test_fit_fuser_rejects_too_little_data() -> None:
    records = synthetic_fusion_records(n_per_class=2, seed=0)
    with pytest.raises(ValueError):
        fit_fuser(records, seed=0)


def test_save_load_round_trip_reproduces_predictions_exactly(tmp_path: Path) -> None:
    records = synthetic_fusion_records(n_per_class=200, seed=1)
    fuser = fit_fuser(records, seed=1)
    out_path = tmp_path / "fuser.json"
    fuser.save(out_path)
    loaded = Fuser.load(out_path)

    assert loaded.detectors == fuser.detectors
    assert np.array_equal(loaded.logit_weights, fuser.logit_weights)
    assert np.array_equal(loaded.presence_weights, fuser.presence_weights)
    assert loaded.bias == fuser.bias
    assert loaded.temperature == fuser.temperature
    assert loaded.band == fuser.band

    for scores in (
        {"informative": 0.9, "inverted": 0.1, "abstaining": 0.5},
        {"informative": 0.1, "inverted": 0.9},  # "abstaining" missing entirely
        {},
    ):
        assert fuser.predict(scores) == loaded.predict(scores)
        assert fuser.predict_label(scores) == loaded.predict_label(scores)


def test_missing_detector_score_matches_explicit_abstain(tmp_path: Path) -> None:
    records = synthetic_fusion_records(n_per_class=200, seed=2)
    fuser = fit_fuser(records, seed=2)

    with_explicit_abstain = {"informative": 0.7, "inverted": 0.3, "abstaining": 0.5}
    without_key_at_all = {"informative": 0.7, "inverted": 0.3}

    # A detector that ran and reported 0.5 has presence=1, logit=0; a
    # detector that never ran has presence=0, logit=0 -- these can differ
    # (the presence weight matters) but neither should raise, and the
    # informative/inverted-only prediction should stay finite and bounded.
    p_explicit = fuser.predict(with_explicit_abstain)
    p_missing = fuser.predict(without_key_at_all)
    assert 0.0 <= p_explicit <= 1.0
    assert 0.0 <= p_missing <= 1.0

    vector_present = fuser.feature_vector({"abstaining": 0.5})
    vector_absent = fuser.feature_vector({})
    index = fuser.detectors.index("abstaining")
    n = len(fuser.detectors)
    assert vector_present[index] == vector_absent[index] == 0.0  # logit(0.5) either way
    assert vector_present[n + index] == 1.0  # present
    assert vector_absent[n + index] == 0.0  # absent


# --- Minimum support for the abstain band ---------------------------------
#
# A band the search accepts because it happens to leave only a handful of
# held-out images outside it is not evidence the band works (see
# ``docs/benchmarks/06_experiment_03_summary.md``, "Fusion on the
# cross-dataset head", where a [0.030, 0.980] band was supported by 3 of
# ~80 held-out images and abstained on 96.9% of the test set). The tests
# below pin the minimum-outside-images floor's three behaviours: it rejects
# an under-supported band even when that band reaches the target, it can be
# switched off to reproduce the pre-floor search exactly, and it never asks
# for more support than a held-out split actually has.


def _frozen_fit_band_no_min_support(
    y: np.ndarray, probs: np.ndarray, target: float
) -> tuple[float, float, float]:
    """A frozen copy of ``_fit_band`` as it was before the minimum-support floor existed.

    Kept only so :func:`test_fit_band_zero_floor_matches_frozen_pre_min_support_search` can
    pin ``_fit_band(..., min_outside=0)`` against the exact search it replaced, independent
    of the current implementation.
    """
    from imgforensics.fusion.stacking import _BAND_GRID, _balanced_accuracy_outside

    best_overall: tuple[float, float, float, float] | None = None  # (bacc, width, low, high)
    best_qualifying: tuple[float, float, float, float] | None = None

    for low_index, low in enumerate(_BAND_GRID):
        below = probs < low
        for high in _BAND_GRID[low_index:]:
            above = probs > high
            outside = below | above
            if not np.any(outside):
                continue

            bacc = _balanced_accuracy_outside(y[outside], above[outside])
            width = float(high - low)

            if (
                best_overall is None
                or bacc > best_overall[0] + 1e-9
                or (abs(bacc - best_overall[0]) <= 1e-9 and width > best_overall[1])
            ):
                best_overall = (bacc, width, float(low), float(high))

            if bacc >= target and (
                best_qualifying is None
                or width > best_qualifying[1] + 1e-12
                or (abs(width - best_qualifying[1]) <= 1e-12 and bacc > best_qualifying[0])
            ):
                best_qualifying = (bacc, width, float(low), float(high))

    chosen = best_qualifying if best_qualifying is not None else best_overall
    if chosen is None:
        return 0.5, 0.5, 0.0
    bacc, _width, low, high = chosen
    return low, high, bacc


def _two_regime_holdout() -> tuple[np.ndarray, np.ndarray]:
    """50 held-out points: 2 perfectly-separated extremes plus 48 overlapping "middle" ones.

    Built so that only the narrow band leaving the 2 extremes outside reaches a 0.9 balanced
    accuracy -- any band wide enough to leave 20+ points outside (the default floor at this
    holdout size) has to reach into the overlapping middle and falls to ~0.6.
    """
    n_mid = 48
    probs = np.zeros(2 + n_mid)
    y = np.zeros(2 + n_mid)

    probs[0], y[0] = 0.01, 0.0  # real, correctly below any low > 0.01
    probs[1], y[1] = 0.99, 1.0  # fake, correctly above any high < 0.99

    probs[2:] = np.linspace(0.30, 0.70, n_mid)
    y[2:] = np.array([0.0, 1.0] * (n_mid // 2))  # alternating labels over overlapping scores
    return y, probs


def test_min_outside_floor_uses_the_larger_bound_and_caps_at_n_holdout() -> None:
    # Count floor binds.
    assert _min_outside_floor(n_holdout=50, min_outside_fraction=0.10, min_outside_count=20) == 20
    # Fraction floor binds.
    assert _min_outside_floor(n_holdout=300, min_outside_fraction=0.10, min_outside_count=20) == 30
    # Both floors exceed n_holdout: capped.
    assert _min_outside_floor(n_holdout=5, min_outside_fraction=0.10, min_outside_count=20) == 5
    # Floor disabled entirely.
    assert _min_outside_floor(n_holdout=50, min_outside_fraction=0.0, min_outside_count=0) == 0


def test_fit_band_rejects_band_supported_by_too_few_images() -> None:
    y, probs = _two_regime_holdout()
    n_holdout = len(y)

    # The only band reaching bacc >= 0.9 leaves just the 2 extremes outside -- inadmissible
    # under the default floor (20 for a 50-image holdout split).
    default_floor = _min_outside_floor(n_holdout, min_outside_fraction=0.10, min_outside_count=20)
    assert default_floor == 20
    result = _fit_band(y, probs, target=0.9, min_outside=default_floor)
    assert (result.low, result.high) != (0.02, 0.98)
    assert result.outside_count >= default_floor
    assert result.target_met is False
    assert result.outside_bacc < 0.9

    # With the floor disabled, the search takes the under-supported band and reports 1.0.
    unconstrained = _fit_band(y, probs, target=0.9, min_outside=0)
    assert unconstrained.low == pytest.approx(0.02)
    assert unconstrained.high == pytest.approx(0.98)
    assert unconstrained.outside_count == 2
    assert unconstrained.target_met is True
    assert unconstrained.outside_bacc == pytest.approx(1.0)


def test_fit_band_zero_floor_matches_frozen_pre_min_support_search() -> None:
    rng = np.random.default_rng(7)
    n = 120
    y = (rng.random(n) < 0.5).astype(float)
    noise = rng.normal(0, 0.05, n)
    probs = np.clip(y * 0.9 + 0.05 + noise, 1e-3, 1 - 1e-3)

    expected_low, expected_high, expected_bacc = _frozen_fit_band_no_min_support(y, probs, 0.9)
    result = _fit_band(y, probs, 0.9, min_outside=0)

    assert result.low == expected_low
    assert result.high == expected_high
    assert result.outside_bacc == expected_bacc


def test_fit_fuser_zero_floor_reproduces_unconstrained_band() -> None:
    """Regression pin: ``min_outside_fraction=0.0, min_outside_count=0`` disables the floor."""
    records = synthetic_fusion_records(n_per_class=200, seed=0)

    with_floor = fit_fuser(records, seed=0)
    without_floor = fit_fuser(records, seed=0, min_outside_fraction=0.0, min_outside_count=0)

    assert without_floor.fit_info.min_outside_count == 0
    # The unconstrained search reaches the same band here -- the default floor (20) never
    # bound in the first place for this easily-separable synthetic data (83 images outside).
    assert without_floor.band == with_floor.band
    assert without_floor.metrics.outside_band_balanced_accuracy == pytest.approx(
        with_floor.metrics.outside_band_balanced_accuracy
    )
    assert with_floor.metrics.band_target_met is True
    assert without_floor.metrics.band_target_met is True


def test_fit_fuser_small_holdout_caps_floor_at_n_holdout() -> None:
    """A held-out split smaller than the default floor (20) must still be able to fit a band."""
    records = synthetic_fusion_records(n_per_class=12, seed=0)
    fuser = fit_fuser(records, seed=0)

    n_holdout = fuser.fit_info.n_fake + fuser.fit_info.n_real
    assert fuser.fit_info.min_outside_count < 20
    assert fuser.fit_info.min_outside_count <= n_holdout
    # The floor equals the number of held-out images actually used to fit the band, i.e. the
    # floor capped every candidate down to "leave every held-out image outside".
    assert fuser.metrics.outside_band_count == fuser.fit_info.min_outside_count
    assert fuser.metrics.abstain_rate == 0.0


def test_fuser_to_dict_round_trips_min_support_fields(tmp_path: Path) -> None:
    records = synthetic_fusion_records(n_per_class=200, seed=0)
    fuser = fit_fuser(records, seed=0, min_outside_fraction=0.10, min_outside_count=20)

    out_path = tmp_path / "fuser.json"
    fuser.save(out_path)
    loaded = Fuser.load(out_path)

    assert loaded.format_version == 2
    assert loaded.fit_info.min_outside_count == fuser.fit_info.min_outside_count
    assert loaded.metrics.outside_band_count == fuser.metrics.outside_band_count
    assert loaded.metrics.band_target_met == fuser.metrics.band_target_met


def test_fuser_from_dict_defaults_missing_min_support_fields() -> None:
    """A fuser.json written before the minimum-support floor existed still loads."""
    records = synthetic_fusion_records(n_per_class=200, seed=0)
    fuser = fit_fuser(records, seed=0)
    data = fuser.to_dict()

    # Simulate a format_version-1 file: no min_outside_count / outside_band_count /
    # band_target_met keys anywhere, as written by an older imgforensics (e.g. the
    # weights/fuser_wildrf.json shape).
    data["format_version"] = 1
    fit_data = data["fit"]
    assert isinstance(fit_data, dict)
    del fit_data["min_outside_count"]
    metrics_data = data["metrics"]
    assert isinstance(metrics_data, dict)
    del metrics_data["outside_band_count"]
    del metrics_data["band_target_met"]

    loaded = Fuser.from_dict(data)
    assert loaded.format_version == 1
    assert loaded.fit_info.min_outside_count == 0
    assert loaded.metrics.outside_band_count == 0
    assert loaded.metrics.band_target_met is True
    # Everything else still round-trips.
    assert loaded.band == fuser.band
    assert loaded.detectors == fuser.detectors


def test_band_fit_and_related_dataclasses_are_constructible() -> None:
    # Cheap smoke test that the new dataclasses hang together as documented -- BandFit is an
    # internal return type, Band/FitInfo/FuserMetrics are the public ones it feeds into.
    band_fit = BandFit(low=0.1, high=0.9, outside_bacc=0.95, outside_count=42, target_met=True)
    assert band_fit.target_met

    fit_info = FitInfo(
        n_images=100, n_fake=50, n_real=50, sources=["s"], levels=["clean"], min_outside_count=10
    )
    assert fit_info.min_outside_count == 10

    metrics = FuserMetrics(
        train_auc=0.9,
        holdout_auc=0.9,
        ece_before=0.1,
        ece_after=0.05,
        abstain_rate=0.2,
        outside_band_balanced_accuracy=0.9,
        outside_band_count=30,
        band_target_met=True,
    )
    assert metrics.outside_band_count == 30
    assert Band(low=band_fit.low, high=band_fit.high).low == 0.1
