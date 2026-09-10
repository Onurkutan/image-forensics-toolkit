"""Tests for imgforensics.fusion.evaluate: evaluate_fuser and FusionEvaluation.

Builds synthetic records for two detectors, ``d1`` and ``d2``, across three
robustness levels, each isolating one thing this module has to get right:

- ``"clean"`` (400 images, both labels): ``d1`` cleanly separates fake from
  real and scores every image; ``d2`` is a constant-abstain feature (score
  0.5) also present on every image, so it carries zero information and
  cannot flip any ranking -- used to check the fused row reaches AUC 1.0 and
  that the abstain band's reported rate matches an independent hand count.
- ``"partial"`` (the same 400 images): ``d1`` scores all of them again, but
  ``d2`` scores only a 150-image subset -- used to check that a detector's
  own row counts only the images it actually scored, unlike the fused row.
- ``"fake_only"`` (50 fake images, no real ones): used to check that a
  level with a single label yields a ``note``, not an exception.

No image, model, or file I/O is needed to build these records, so this whole
module runs fast and without the ``ml`` extra.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from imgforensics.eval.records import ScoreRecord
from imgforensics.fusion.evaluate import EvalRow, FusionEvaluation, evaluate_fuser
from imgforensics.fusion.stacking import fit_fuser

_N_PER_CLASS = 200


def _entry_path(label: str, index: int) -> str:
    return f"synthetic/{label}/{index:04d}.png"


def _record(entry_path: str, label: str, level: str, detector: str, score: float) -> ScoreRecord:
    return ScoreRecord(
        entry_path=entry_path,
        label=label,
        source="synthetic",
        generator=None,
        split=None,
        level=level,
        detector=detector,
        score=score,
        elapsed_ms=None,
    )


def _build_records() -> tuple[list[ScoreRecord], list[ScoreRecord]]:
    """Returns ``(fit_records, extra_records)`` -- see the module docstring."""
    rng = np.random.default_rng(0)
    fit_records: list[ScoreRecord] = []
    extra_records: list[ScoreRecord] = []

    for index in range(_N_PER_CLASS):
        for label in ("real", "fake"):
            path = _entry_path(label, index)
            base = 0.9 if label == "fake" else 0.1
            noise = float(np.clip(rng.normal(0, 0.03), -0.04, 0.04))
            d1_clean = base + noise  # fake in [0.86, 0.94], real in [0.06, 0.14]: no overlap

            fit_records.append(_record(path, label, "clean", "d1", d1_clean))
            fit_records.append(_record(path, label, "clean", "d2", 0.5))

            # "partial": d1 scores every image, d2 only the first 75 of each label.
            extra_records.append(_record(path, label, "partial", "d1", 0.5))
            if index < 75:
                extra_records.append(_record(path, label, "partial", "d2", 0.5))

            # "fake_only": both detectors, but only the fake label is present.
            if label == "fake" and index < 50:
                extra_records.append(_record(path, label, "fake_only", "d1", 0.7))
                extra_records.append(_record(path, label, "fake_only", "d2", 0.3))

    return fit_records, extra_records


def _row(evaluation: FusionEvaluation, level: str, scorer: str) -> EvalRow:
    return next(r for r in evaluation.rows if r.level == level and r.scorer == scorer)


def test_fused_row_reaches_perfect_auc_on_the_separable_level() -> None:
    fit_records, _ = _build_records()
    fuser = fit_fuser(fit_records, seed=0)
    assert set(fuser.detectors) == {"d1", "d2"}

    evaluation = evaluate_fuser(fuser, fit_records)
    fused = _row(evaluation, "clean", "fused")

    assert fused.auc == pytest.approx(1.0)
    assert fused.n == 2 * _N_PER_CLASS
    assert fused.note is None


def test_detector_own_row_counts_only_images_it_scored() -> None:
    fit_records, extra_records = _build_records()
    fuser = fit_fuser(fit_records, seed=0)

    evaluation = evaluate_fuser(fuser, fit_records + extra_records)

    d1_row = _row(evaluation, "partial", "d1")
    d2_row = _row(evaluation, "partial", "d2")
    fused_row = _row(evaluation, "partial", "fused")

    assert d1_row.n == 2 * _N_PER_CLASS
    assert d2_row.n == 150  # 75 real + 75 fake
    assert fused_row.n == 2 * _N_PER_CLASS  # every image, via imputation for d2


def test_fused_outside_band_abstain_rate_matches_hand_count() -> None:
    fit_records, _ = _build_records()
    fuser = fit_fuser(fit_records, seed=0)

    evaluation = evaluate_fuser(fuser, fit_records)
    outside_row = _row(evaluation, "clean", "fused_outside_band")

    by_entry: dict[str, dict[str, float]] = {}
    labels: dict[str, str] = {}
    for record in fit_records:
        by_entry.setdefault(record.entry_path, {})[record.detector] = record.score
        labels[record.entry_path] = record.label
    probabilities = {path: fuser.predict(scores) for path, scores in by_entry.items()}
    inside = sum(1 for p in probabilities.values() if fuser.band.low <= p <= fuser.band.high)
    expected_abstain_rate = inside / len(probabilities)
    expected_outside_n = len(probabilities) - inside

    assert outside_row.abstain_rate == pytest.approx(expected_abstain_rate)
    assert outside_row.n == expected_outside_n


def test_one_label_level_yields_a_note_not_an_exception() -> None:
    fit_records, extra_records = _build_records()
    fuser = fit_fuser(fit_records, seed=0)

    evaluation = evaluate_fuser(fuser, fit_records + extra_records)

    fake_only_fused = _row(evaluation, "fake_only", "fused")
    assert fake_only_fused.note is not None
    assert fake_only_fused.auc is None
    assert fake_only_fused.ap is None
    assert fake_only_fused.balanced_accuracy is None

    fake_only_d1 = _row(evaluation, "fake_only", "d1")
    assert fake_only_d1.note is not None
    assert fake_only_d1.n == 50


def test_to_markdown_has_one_table_per_level_with_fused_rows_last() -> None:
    fit_records, extra_records = _build_records()
    fuser = fit_fuser(fit_records, seed=0)
    evaluation = evaluate_fuser(fuser, fit_records + extra_records)

    markdown = evaluation.to_markdown()
    for level in evaluation.levels:
        assert f"## Level: {level}" in markdown

    for level in evaluation.levels:
        level_rows = [r for r in evaluation.rows if r.level == level]
        assert level_rows[-2].scorer == "fused"
        assert level_rows[-1].scorer == "fused_outside_band"


def test_to_dict_round_trips_through_json_dumps() -> None:
    fit_records, extra_records = _build_records()
    fuser = fit_fuser(fit_records, seed=0)
    evaluation = evaluate_fuser(fuser, fit_records + extra_records)

    payload = json.dumps(evaluation.to_dict())
    restored = json.loads(payload)

    assert restored["detectors"] == list(evaluation.detectors)
    assert restored["levels"] == evaluation.levels
    assert len(restored["rows"]) == len(evaluation.rows)
    assert restored["rows"][0]["scorer"] == evaluation.rows[0].scorer


def test_evaluate_fuser_rejects_empty_records() -> None:
    fit_records, _ = _build_records()
    fuser = fit_fuser(fit_records, seed=0)
    with pytest.raises(ValueError, match="no records"):
        evaluate_fuser(fuser, [])


def test_evaluate_fuser_rejects_records_without_any_fuser_detector() -> None:
    fit_records, _ = _build_records()
    fuser = fit_fuser(fit_records, seed=0)
    unrelated = [_record("x.png", "fake", "clean", "some_other_detector", 0.9)]
    with pytest.raises(ValueError, match="none of the fuser's detectors"):
        evaluate_fuser(fuser, unrelated)


def test_evaluate_fuser_rejects_unknown_level() -> None:
    fit_records, _ = _build_records()
    fuser = fit_fuser(fit_records, seed=0)
    with pytest.raises(ValueError, match="not present in records"):
        evaluate_fuser(fuser, fit_records, levels=["does-not-exist"])
