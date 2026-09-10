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
    Fuser,
    FusionFeatures,
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
