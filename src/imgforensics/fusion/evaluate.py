"""Reproducible per-level evaluation tables for a fitted stacking fuser.

``docs/benchmarks/04_fusion_wildrf.md`` reports, for each robustness level: a
detector's AUC / balanced accuracy / FPR / TPR alone at a fixed threshold,
the same numbers for the fused verdict over every image, and again over just
the images the fuser is willing to call (outside the abstain band -- see
:mod:`imgforensics.fusion.stacking`). Those numbers came from a script that
was run once and never committed. :func:`evaluate_fuser` is that script made
a first-class, tested function, and ``imgforensics fusion eval`` is the CLI
command built on it, so the same table can be reproduced from a fitted
``fuser.json`` and one or more saved
:class:`~imgforensics.eval.runner.BenchmarkResult` files whenever the fuser
is refit or the deployment data changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from imgforensics.eval.metrics import (
    average_precision,
    balanced_accuracy_at_threshold,
    expected_calibration_error,
    fpr_at_threshold,
    roc_auc,
    tpr_at_threshold,
)
from imgforensics.eval.records import ScoreRecord
from imgforensics.fusion.stacking import Fuser

_FUSED_ROW = "fused"
_FUSED_OUTSIDE_BAND_ROW = "fused_outside_band"
_ONE_LABEL_NOTE = "only one label present; AUC undefined"

_MARKDOWN_HEADER = "| scorer | n | auc | ap | balanced_accuracy | fpr | tpr | ece | abstain_rate |"
_MARKDOWN_SEPARATOR = "|---|---|---|---|---|---|---|---|---|"


def _labels_array(records: Sequence[ScoreRecord]) -> np.ndarray:
    return np.array([1.0 if r.label == "fake" else 0.0 for r in records], dtype=np.float64)


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "-"


@dataclass(frozen=True)
class EvalRow:
    """One scorer's metrics at one robustness level.

    ``scorer`` is a detector name from the fuser, or one of the two fused
    rows (``"fused"``, ``"fused_outside_band"``). ``n`` is how many images
    the metrics below were computed over: for a detector's own row, only the
    images it actually scored at this level -- it is silently dropped from
    that count for an image it never ran on, unlike the two fused rows,
    which impute a missing detector and therefore use every image present
    at the level (see :func:`evaluate_fuser`). ``auc``/``ap`` are ``None``
    when fewer than two labels are present among those ``n`` rows -- both
    are undefined with only one class. ``abstain_rate`` is only meaningful
    on the ``"fused_outside_band"`` row: the fraction of the level's images
    whose fused probability fell inside the abstain band and were therefore
    excluded from that row. ``note`` explains a degenerate row (no images,
    or only one label) instead of the row ever raising.
    """

    scorer: str
    level: str
    n: int
    auc: float | None
    ap: float | None
    balanced_accuracy: float | None
    fpr: float | None
    tpr: float | None
    ece: float | None
    abstain_rate: float | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """This row as a JSON-serialisable ``dict``."""
        return {
            "scorer": self.scorer,
            "level": self.level,
            "n": self.n,
            "auc": self.auc,
            "ap": self.ap,
            "balanced_accuracy": self.balanced_accuracy,
            "fpr": self.fpr,
            "tpr": self.tpr,
            "ece": self.ece,
            "abstain_rate": self.abstain_rate,
            "note": self.note,
        }


@dataclass(frozen=True)
class FusionEvaluation:
    """Every :class:`EvalRow` :func:`evaluate_fuser` computed, grouped by level."""

    detectors: tuple[str, ...]
    levels: list[str]
    rows: list[EvalRow]

    def to_markdown(self) -> str:
        """One table per level, headed ``## Level: <name>``, the two fused rows last."""
        lines: list[str] = []
        for level in self.levels:
            lines.append(f"## Level: {level}")
            lines.append("")
            lines.append(_MARKDOWN_HEADER)
            lines.append(_MARKDOWN_SEPARATOR)
            for row in self.rows:
                if row.level != level:
                    continue
                lines.append(
                    f"| {row.scorer} | {row.n} | {_fmt(row.auc)} | {_fmt(row.ap)} | "
                    f"{_fmt(row.balanced_accuracy)} | {_fmt(row.fpr)} | {_fmt(row.tpr)} | "
                    f"{_fmt(row.ece)} | {_fmt(row.abstain_rate)} |"
                )
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """This evaluation as a JSON-serialisable ``dict``."""
        return {
            "detectors": list(self.detectors),
            "levels": list(self.levels),
            "rows": [row.to_dict() for row in self.rows],
        }


def _scorer_row(
    scorer: str,
    level: str,
    y: np.ndarray,
    values: np.ndarray,
    threshold: float,
    *,
    empty_note: str,
) -> EvalRow:
    """A detector's own row, or the ``"fused"`` row: every metric ``None`` under one label.

    ``values`` is used both to rank (AUC/AP/threshold metrics) and as the
    calibrated probability (ECE) -- a raw detector score already plays both
    roles elsewhere in this project (see ``BenchmarkResult.image_table``),
    and the fused probability is already calibrated by construction.
    """
    n = int(len(y))
    if n == 0:
        return EvalRow(scorer, level, 0, None, None, None, None, None, None, note=empty_note)
    if len(np.unique(y)) < 2:
        return EvalRow(scorer, level, n, None, None, None, None, None, None, note=_ONE_LABEL_NOTE)
    return EvalRow(
        scorer,
        level,
        n,
        roc_auc(y, values),
        average_precision(y, values),
        balanced_accuracy_at_threshold(y, values, threshold),
        fpr_at_threshold(y, values, threshold),
        tpr_at_threshold(y, values, threshold),
        expected_calibration_error(y, values),
    )


def _outside_band_row(
    level: str,
    y: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    abstain_rate: float | None,
) -> EvalRow:
    """The ``"fused_outside_band"`` row: only AUC/AP go ``None`` under one label, not the rest.

    Unlike :func:`_scorer_row`, balanced accuracy / FPR / TPR / ECE stay
    meaningful with a single label outside the band (e.g. every image the
    fuser is willing to call there happens to be fake) -- what matters here
    is whether the fuser got them right, which those metrics still answer
    even without a negative to rank against.
    """
    n = int(len(y))
    if n == 0:
        return EvalRow(
            _FUSED_OUTSIDE_BAND_ROW,
            level,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            abstain_rate=abstain_rate,
            note="no images outside the abstain band",
        )
    two_labels = len(np.unique(y)) >= 2
    return EvalRow(
        _FUSED_OUTSIDE_BAND_ROW,
        level,
        n,
        roc_auc(y, probs) if two_labels else None,
        average_precision(y, probs) if two_labels else None,
        balanced_accuracy_at_threshold(y, probs, threshold),
        fpr_at_threshold(y, probs, threshold),
        tpr_at_threshold(y, probs, threshold),
        expected_calibration_error(y, probs),
        abstain_rate=abstain_rate,
        note=None if two_labels else "only one label present outside the band; AUC/AP undefined",
    )


def evaluate_fuser(
    fuser: Fuser,
    records: Sequence[ScoreRecord],
    *,
    levels: Sequence[str] | None = None,
    threshold: float = 0.5,
) -> FusionEvaluation:
    """Evaluate ``fuser`` against ``records``, one table of rows per robustness level.

    Records for a detector ``fuser`` was not fitted on are ignored. A
    detector ``fuser`` was fitted on but that did not score a given image is
    imputed as abstaining (score 0.5), exactly as :meth:`Fuser.predict`
    already does for a live ``{detector: score}`` dict -- the same rule
    ``analyze --fuser`` relies on. For every level, records are grouped by
    ``entry_path`` and turned into, in order:

    - one row per detector in ``fuser.detectors``, scored by that
      detector's own raw score, counting only the images it actually
      scored at this level;
    - a ``"fused"`` row over every image present at the level (any detector
      missing for a given image is imputed), scored by :meth:`Fuser.predict`;
    - a ``"fused_outside_band"`` row over the images whose fused
      probability falls outside ``[fuser.band.low, fuser.band.high]``, plus
      the fraction that fell inside (``abstain_rate``).

    Args:
        fuser: A fitted fuser (see
            :func:`~imgforensics.fusion.stacking.fit_fuser`).
        records: Score records, typically every record from one or more
            saved :class:`~imgforensics.eval.runner.BenchmarkResult` files.
        levels: Robustness levels to report. Defaults to every level
            present in ``records``.
        threshold: Operating threshold for balanced accuracy / FPR / TPR --
            a score (or fused probability) predicts "fake" iff it is
            strictly greater than this (see ``imgforensics.eval.metrics``'
            "Threshold convention").

    Raises:
        ValueError: if ``records`` is empty, none of ``fuser.detectors``
            appear anywhere in ``records``, or a requested level is absent
            from ``records``. A level where a scorer sees only one label
            never raises -- it yields a row with ``note`` set and its
            metrics ``None`` instead (see :class:`EvalRow`).
    """
    if not records:
        raise ValueError("no records given to evaluate")

    fuser_detectors = set(fuser.detectors)
    relevant = [r for r in records if r.detector in fuser_detectors]
    if not relevant:
        raise ValueError(
            f"none of the fuser's detectors ({', '.join(fuser.detectors)}) are present in "
            f"records. Available: {', '.join(sorted({r.detector for r in records}))}"
        )

    available_levels = sorted({r.level for r in records})
    resolved_levels = sorted(levels) if levels is not None else available_levels
    unknown_levels = sorted(set(resolved_levels) - set(available_levels))
    if unknown_levels:
        raise ValueError(
            f"level(s) not present in records: {', '.join(unknown_levels)}. "
            f"Available: {', '.join(available_levels)}"
        )

    rows: list[EvalRow] = []
    for level in resolved_levels:
        level_records = [r for r in relevant if r.level == level]

        entry_scores: dict[str, dict[str, float]] = {}
        entry_label: dict[str, str] = {}
        for record in level_records:
            entry_scores.setdefault(record.entry_path, {})[record.detector] = record.score
            entry_label[record.entry_path] = record.label

        for detector in fuser.detectors:
            detector_records = [r for r in level_records if r.detector == detector]
            rows.append(
                _scorer_row(
                    detector,
                    level,
                    _labels_array(detector_records),
                    np.array([r.score for r in detector_records], dtype=np.float64),
                    threshold,
                    empty_note=f"no records for {detector!r} at level {level!r}",
                )
            )

        entries = sorted(entry_scores)
        y_all = np.array(
            [1.0 if entry_label[path] == "fake" else 0.0 for path in entries], dtype=np.float64
        )
        fused_probs = np.array(
            [fuser.predict(entry_scores[path]) for path in entries], dtype=np.float64
        )
        rows.append(
            _scorer_row(
                _FUSED_ROW,
                level,
                y_all,
                fused_probs,
                threshold,
                empty_note=f"no images at level {level!r}",
            )
        )

        outside_mask: np.ndarray
        if len(fused_probs) == 0:
            abstain_rate: float | None = None
            outside_mask = np.zeros(0, dtype=bool)
        else:
            inside = (fused_probs >= fuser.band.low) & (fused_probs <= fuser.band.high)
            abstain_rate = float(np.mean(inside))
            outside_mask = ~inside
        rows.append(
            _outside_band_row(
                level, y_all[outside_mask], fused_probs[outside_mask], threshold, abstain_rate
            )
        )

    return FusionEvaluation(detectors=fuser.detectors, levels=resolved_levels, rows=rows)
