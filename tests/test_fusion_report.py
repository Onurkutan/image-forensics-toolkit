"""Tests for imgforensics.fusion.report: explain() and the detector-notes table."""

from __future__ import annotations

import numpy as np

from imgforensics.fusion.report import DETECTOR_NOTES, GENERIC_NOTE, explain
from imgforensics.fusion.stacking import Band, FitInfo, Fuser, FuserMetrics

_KNOWN_DETECTORS = (
    "metadata",
    "ela",
    "c2pa",
    "sd_watermark",
    "copy_move",
    "jpeg_ghost",
    "double_jpeg",
    "dinov2_head",
    "iml_vit",
)


def _manual_fuser(detectors: list[str]) -> Fuser:
    """A hand-built fuser (bypassing fit_fuser) so report tests don't depend on fitting."""
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


def test_every_known_detector_has_a_specific_note() -> None:
    for name in _KNOWN_DETECTORS:
        assert name in DETECTOR_NOTES
        assert DETECTOR_NOTES[name] != GENERIC_NOTE
        assert DETECTOR_NOTES[name]  # non-empty


def test_explain_uses_specific_notes_and_generic_fallback() -> None:
    detectors = [*_KNOWN_DETECTORS, "totally_unknown_detector"]
    fuser = _manual_fuser(detectors)
    contributions = explain(fuser, {name: 0.7 for name in detectors})
    notes = {c.detector: c.note for c in contributions}

    for name in _KNOWN_DETECTORS:
        assert notes[name] == DETECTOR_NOTES[name]
    assert notes["totally_unknown_detector"] == GENERIC_NOTE


def test_explain_orders_by_absolute_contribution_descending() -> None:
    fuser = _manual_fuser(["a", "b", "c"])
    # Weights are 1, 2, 3 for a, b, c (see _manual_fuser), so weight alone
    # would rank c > b > a; scores are chosen so |contribution| = |weight *
    # logit| instead ranks b > a > c, to confirm explain() sorts by the
    # product, not by weight or score alone.
    scores = {"a": 0.95, "b": 0.97, "c": 0.55}
    contributions = explain(fuser, scores)

    magnitudes = [abs(c.contribution) for c in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert [c.detector for c in contributions] == ["b", "a", "c"]


def test_explain_flags_missing_detectors_as_absent() -> None:
    fuser = _manual_fuser(["a", "b"])
    contributions = explain(fuser, {"a": 0.9})
    by_name = {c.detector: c for c in contributions}

    assert by_name["a"].present is True
    assert by_name["b"].present is False
    assert by_name["b"].score == 0.5
    assert by_name["b"].logit == 0.0
    assert by_name["b"].contribution == 0.0


def test_explain_contribution_is_weight_times_logit() -> None:
    fuser = _manual_fuser(["a", "b"])
    contributions = explain(fuser, {"a": 0.9, "b": 0.1})
    by_name = {c.detector: c for c in contributions}

    for name in ("a", "b"):
        contribution = by_name[name]
        assert contribution.contribution == contribution.weight * contribution.logit
