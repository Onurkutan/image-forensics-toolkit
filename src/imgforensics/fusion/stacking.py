"""Calibrated logistic stacking fuser over heterogeneous detector scores.

``docs/ROADMAP.md``, Phase 5: combine signal/detector/localizer scores into
one image-level probability with an abstain band, fitted on benchmark
records (:class:`~imgforensics.eval.records.ScoreRecord`). Pure numpy -- no
scikit-learn, no torch -- so fitting or applying a fuser never needs the
optional ``ml`` extra, and this module is safe to import in any test process
regardless of what else is running on the GPU.

**Feature construction.** Every detector score is mapped through a logit
transform (:func:`logit_from_score`), clipped to ``[1e-4, 1 - 1e-4]`` first
so the transform stays finite. Two properties fall out of that choice, both
load-bearing (see ``docs/benchmarks/01_experiment_summary.md`` for measured
examples of both):

- A detector that abstains at exactly 0.5 (every classical signal's
  documented behaviour on a file it cannot read a cue from) maps to logit 0
  -- it contributes nothing to the fused logit regardless of its learned
  weight, rather than voting "half fake".
- A detector that runs *inverted* relative to how it is meant to be read
  (``ela``/``jpeg_ghost`` on this project's own validation set) can still be
  used: gradient descent is free to fit a negative weight for it, which
  flips its vote back the right way instead of the fuser having to special
  case it.

Missing detectors (not run for a given image) are imputed as abstaining
(score 0.5, logit 0) *and* get a "present" indicator feature so the model
can learn to distinguish "this detector said uncertain" from "this detector
did not run at all" -- two situations that would otherwise look identical.

**Model.** L2-regularized logistic regression (class-balanced sample
weights) over ``[logits | presence indicators]``, fitted by full-batch
gradient descent with Armijo backtracking line search -- simple and
deterministic (zero initialization, no randomness in the optimizer itself).
A held-out 20% split (by hashing ``entry_path``, so every robustness level
of the same image lands on the same side) is used only for temperature
scaling and the abstain band, following the same "keep the identity unless
it measurably helps" calibration policy as
:mod:`imgforensics.detectors.train` (``_fit_calibration``).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from imgforensics import __version__
from imgforensics.core.types import Label
from imgforensics.eval.metrics import expected_calibration_error, roc_auc
from imgforensics.eval.records import ScoreRecord

#: Clip range applied to a raw [0, 1] score before taking its logit, so the
#: transform stays finite; matches the clamp already used for probabilities
#: elsewhere in the project (e.g. ``imgforensics.detectors.train._EPSILON``).
_EPSILON = 1e-4

#: Bounds the fitted temperature is clamped into before the ECE accept/reject
#: check -- the same bounds :mod:`imgforensics.detectors.train` uses, for the
#: same reason: an unconstrained fit on a small or separable split pushes
#: the temperature toward 0 (a step function).
_TEMPERATURE_BOUNDS = (0.05, 20.0)

#: ``format_version`` bumped 1 -> 2 for the ``outside_band_count`` / ``band_target_met`` /
#: ``min_outside_count`` fields added below. :meth:`Fuser.from_dict` defaults all three when
#: absent, so a version-1 file (no ``format_version`` key, or ``format_version: 1``) still loads.
_FORMAT_VERSION = 2
_HOLDOUT_FRACTION = 0.2
_GD_ITERATIONS = 300
_LINE_SEARCH_MAX_STEPS = 30
_LINE_SEARCH_SHRINK = 0.5
_ARMIJO_C = 1e-4

#: Grid the abstain band's low/high thresholds are searched over, and the
#: grid temperature candidates are searched over (log-spaced within
#: :data:`_TEMPERATURE_BOUNDS`). 101 points mirrors the pixel-threshold sweep
#: convention in ``imgforensics.eval.metrics._DEFAULT_PIXEL_THRESHOLDS``.
_BAND_GRID = np.linspace(0.0, 1.0, 101)
_TEMPERATURE_GRID_POINTS = 400

#: Default minimum-support floor for an abstain band candidate (see
#: :func:`_min_outside_floor`): a band is only admissible if it leaves at least this many
#: held-out images outside it, or ``ceil(_MIN_OUTSIDE_FRACTION * n_holdout)``, whichever is
#: larger. Without this floor the band search can accept a band a handful of held-out images
#: happen to fall outside of and report a perfect outside-band accuracy on them -- three
#: images out of ~80 is not evidence the band works, it is the search finding a loophole
#: (see ``docs/benchmarks/06_experiment_03_summary.md``, "Fusion on the cross-dataset head").
_MIN_OUTSIDE_COUNT = 20
_MIN_OUTSIDE_FRACTION = 0.10

#: Registry names :func:`default_detectors` excludes: the trivial baselines
#: (see ``imgforensics.eval.baselines``) are deliberately not detectors a
#: fuser should learn to weight -- they carry no forensic signal by
#: construction -- and ``signals_mean`` is itself already a fusion of the
#: signals, so including it alongside its inputs would double-count them.
_TRIVIAL_BASELINE_NAMES = frozenset({"random", "signals_mean"})


def _is_trivial_baseline(name: str) -> bool:
    return name.startswith("constant_") or name in _TRIVIAL_BASELINE_NAMES


def default_detectors(records: Sequence[ScoreRecord]) -> list[str]:
    """Sorted detector names in ``records``, minus the trivial baselines.

    This is the default detector list :meth:`FusionFeatures.build` and
    :func:`fit_fuser` use when ``detectors`` is not given explicitly.
    """
    names = {record.detector for record in records}
    return sorted(name for name in names if not _is_trivial_baseline(name))


def logit_from_score(score: float) -> float:
    """Logit of ``score``, clipped to ``[_EPSILON, 1 - _EPSILON]`` first.

    ``logit_from_score(0.5) == 0.0`` exactly -- an abstaining detector
    contributes nothing to a fused logit regardless of its weight. See the
    module docstring for why this matters for missing and inverted signals.
    """
    clipped = min(max(float(score), _EPSILON), 1.0 - _EPSILON)
    return float(np.log(clipped / (1.0 - clipped)))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _split_mask(entry_paths: Sequence[str], seed: int, holdout_fraction: float) -> np.ndarray:
    """Deterministic held-out mask: ``True`` for rows whose ``entry_path`` hashes into holdout.

    Hashing the path (not the row) means every robustness level of the same
    image gets the same split assignment, so no image leaks between train
    and held-out through a perturbed copy of itself. ``seed`` salts the
    hash, so a different seed reshuffles the split without touching
    anything else about the fit.
    """
    fractions = np.array(
        [
            int.from_bytes(hashlib.sha256(f"{path}:{seed}".encode()).digest()[:8], "big")
            / float(2**64)
            for path in entry_paths
        ]
    )
    return fractions < holdout_fraction


@dataclass(frozen=True)
class FusionFeatures:
    """The feature matrix built from a list of :class:`ScoreRecord`.

    One row per ``(entry_path, level)`` pair present in the input (after
    filtering to ``levels``), not one row per image: by default every
    robustness level is kept, so the same image contributes one training row
    per perturbation and the fuser also learns from perturbed scores. ``X``
    has ``2 * len(detectors)`` columns: the first ``len(detectors)`` are
    logit-transformed scores (:func:`logit_from_score`), imputed at 0 (score
    0.5) when a detector is missing for that row; the second
    ``len(detectors)`` are 1.0/0.0 "this detector ran" indicators.
    """

    detectors: tuple[str, ...]
    entry_paths: list[str]
    levels_used: list[str]
    sources: list[str]
    X: np.ndarray
    y: np.ndarray

    @property
    def n_detectors(self) -> int:
        return len(self.detectors)

    @classmethod
    def build(
        cls,
        records: Sequence[ScoreRecord],
        detectors: Sequence[str] | None = None,
        levels: Sequence[str] | None = None,
    ) -> FusionFeatures:
        """Group ``records`` by ``(entry_path, level)`` and build the feature matrix.

        Args:
            records: Score records, typically every record from one or more
                :class:`~imgforensics.eval.runner.BenchmarkResult` files.
            detectors: Feature columns, in this order. Defaults to
                :func:`default_detectors`.
            levels: Robustness levels to include. Defaults to every level
                present in ``records``.

        Raises:
            ValueError: if ``records`` is empty, a requested detector or
                level is absent from ``records``, no detectors resolve, or
                nothing is left after filtering.
        """
        if not records:
            raise ValueError("no records given to build fusion features from")

        available_detectors = {record.detector for record in records}
        if detectors is not None:
            resolved_detectors = tuple(sorted(detectors))
            unknown_detectors = sorted(set(resolved_detectors) - available_detectors)
            if unknown_detectors:
                raise ValueError(
                    f"detector(s) not present in records: {', '.join(unknown_detectors)}. "
                    f"Available: {', '.join(sorted(available_detectors))}"
                )
        else:
            resolved_detectors = tuple(default_detectors(records))
        if not resolved_detectors:
            raise ValueError(
                "no detectors to fuse -- records contain only trivial baselines, or an "
                "explicit --detector list resolved to nothing"
            )

        available_levels = sorted({record.level for record in records})
        resolved_levels = sorted(levels) if levels is not None else available_levels
        unknown_levels = sorted(set(resolved_levels) - set(available_levels))
        if unknown_levels:
            raise ValueError(
                f"level(s) not present in records: {', '.join(unknown_levels)}. "
                f"Available: {', '.join(available_levels)}"
            )

        selected = [record for record in records if record.level in resolved_levels]
        if not selected:
            raise ValueError("no records left after filtering by level")

        groups: dict[tuple[str, str], dict[str, object]] = {}
        for record in selected:
            key = (record.entry_path, record.level)
            group = groups.setdefault(
                key, {"label": record.label, "source": record.source, "scores": {}}
            )
            scores = group["scores"]
            assert isinstance(scores, dict)  # narrows for mypy; set above, never replaced
            scores[record.detector] = record.score

        rows = sorted(groups.keys())
        n_rows = len(rows)
        n_detectors = len(resolved_detectors)
        X = np.zeros((n_rows, 2 * n_detectors), dtype=np.float64)
        y = np.zeros(n_rows, dtype=np.float64)
        entry_paths: list[str] = []
        sources_seen: set[str] = set()

        for row_index, key in enumerate(rows):
            group = groups[key]
            scores_map = group["scores"]
            assert isinstance(scores_map, dict)
            for col_index, detector in enumerate(resolved_detectors):
                score = scores_map.get(detector, 0.5)
                X[row_index, col_index] = logit_from_score(score)
                X[row_index, n_detectors + col_index] = 1.0 if detector in scores_map else 0.0
            y[row_index] = 1.0 if group["label"] == "fake" else 0.0
            entry_paths.append(key[0])
            sources_seen.add(str(group["source"]))

        return cls(
            detectors=resolved_detectors,
            entry_paths=entry_paths,
            levels_used=resolved_levels,
            sources=sorted(sources_seen),
            X=X,
            y=y,
        )


def _class_balanced_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample weight so the positive and negative class contribute equally to the loss."""
    n = float(len(y))
    n_pos = float(np.sum(y == 1.0))
    n_neg = float(np.sum(y == 0.0))
    return np.where(y == 1.0, n / (2.0 * n_pos), n / (2.0 * n_neg))


def _weighted_bce(y: np.ndarray, p: np.ndarray, sample_weight: np.ndarray) -> float:
    p_clipped = np.clip(p, _EPSILON, 1.0 - _EPSILON)
    per_sample = -(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped))
    return float(np.mean(sample_weight * per_sample))


def _fit_logistic(
    X: np.ndarray, y: np.ndarray, *, l2: float, n_iter: int
) -> tuple[np.ndarray, float]:
    """Full-batch gradient descent with Armijo backtracking on class-balanced weighted BCE + L2.

    Deterministic: zero-initialized, no random draws. The step size grows
    back toward 1 after a successful step and shrinks geometrically inside
    the line search, so a run that starts making no progress (line search
    exhausted) stops early rather than looping uselessly.
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0
    sample_weight = _class_balanced_weights(y)

    def loss_and_grad(w: np.ndarray, b: float) -> tuple[float, np.ndarray, float]:
        z = X @ w + b
        p = _sigmoid(z)
        grad_common = sample_weight * (p - y) / n_samples
        grad_w = X.T @ grad_common + l2 * w
        grad_b = float(np.sum(grad_common))
        loss = _weighted_bce(y, p, sample_weight) + 0.5 * l2 * float(np.dot(w, w))
        return loss, grad_w, grad_b

    loss, grad_w, grad_b = loss_and_grad(weights, bias)
    step = 1.0
    for _ in range(n_iter):
        grad_norm_sq = float(np.dot(grad_w, grad_w) + grad_b**2)
        if grad_norm_sq < 1e-16:
            break

        accepted = False
        for _ in range(_LINE_SEARCH_MAX_STEPS):
            candidate_w = weights - step * grad_w
            candidate_b = bias - step * grad_b
            candidate_loss, candidate_grad_w, candidate_grad_b = loss_and_grad(
                candidate_w, candidate_b
            )
            if candidate_loss <= loss - _ARMIJO_C * step * grad_norm_sq:
                accepted = True
                break
            step *= _LINE_SEARCH_SHRINK

        if not accepted:
            break

        weights, bias = candidate_w, candidate_b
        loss, grad_w, grad_b = candidate_loss, candidate_grad_w, candidate_grad_b
        step = min(step / _LINE_SEARCH_SHRINK, 1.0)

    return weights, bias


def _fit_temperature(logits: np.ndarray, y: np.ndarray, ece_before: float) -> float:
    """Grid-search a scalar temperature minimizing held-out NLL; keep 1.0 unless ECE improves.

    Same accept/reject policy as ``imgforensics.detectors.train._fit_calibration``:
    the temperature that best fits the likelihood is only used if it does
    not increase the expected calibration error measured on the same split,
    otherwise the identity (``T = 1``) is recorded.
    """
    if len(np.unique(y)) < 2:
        return 1.0

    low, high = _TEMPERATURE_BOUNDS
    candidates = np.geomspace(low, high, _TEMPERATURE_GRID_POINTS)
    candidates = np.concatenate([candidates, [1.0]])
    uniform_weight = np.ones_like(y)

    best_t = 1.0
    best_nll = _weighted_bce(y, _sigmoid(logits), uniform_weight)
    for t in candidates:
        nll = _weighted_bce(y, _sigmoid(logits / t), uniform_weight)
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    ece_after = expected_calibration_error(y, _sigmoid(logits / best_t))
    return best_t if ece_after <= ece_before else 1.0


def _balanced_accuracy_outside(y_outside: np.ndarray, predicted_fake: np.ndarray) -> float:
    positive = y_outside == 1.0
    negative = y_outside == 0.0
    tpr = float(np.mean(predicted_fake[positive])) if np.any(positive) else 0.0
    tnr = float(np.mean(~predicted_fake[negative])) if np.any(negative) else 0.0
    return (tpr + tnr) / 2.0


def _min_outside_floor(n_holdout: int, min_outside_fraction: float, min_outside_count: int) -> int:
    """The minimum number of held-out images a band candidate must leave outside it.

    ``max(min_outside_count, ceil(min_outside_fraction * n_holdout))``, capped at
    ``n_holdout``: a held-out split smaller than the configured floor must still be able to
    fit a band, so the floor never asks for more outside images than exist. When the cap is
    what binds -- ``n_holdout`` itself is below the configured floor -- every held-out image
    must fall outside the band for a candidate to be admissible at all, i.e. "no abstention
    on the held-out split" is the only band :func:`_fit_band` can accept.
    """
    return min(max(min_outside_count, math.ceil(min_outside_fraction * n_holdout)), n_holdout)


@dataclass(frozen=True)
class BandFit:
    """The abstain band :func:`_fit_band` chose, and whether it met the accuracy target."""

    low: float
    high: float
    outside_bacc: float
    outside_count: int
    target_met: bool


def _fit_band(y: np.ndarray, probs: np.ndarray, target: float, *, min_outside: int) -> BandFit:
    """Choose ``(low, high)`` so real/fake predictions outside the band meet ``target``.

    Searches every ``(low, high)`` pair on :data:`_BAND_GRID` (``low <= high``), first
    discarding any candidate that leaves fewer than ``min_outside`` held-out images outside
    it (see :func:`_min_outside_floor` for how that floor is computed, and
    :data:`_MIN_OUTSIDE_COUNT` for why a band needs real support to mean anything). Among the
    admissible candidates, picks the widest whose *outside*-band balanced accuracy (real
    below low, fake above high) is at least ``target``, as before the floor existed. When no
    admissible candidate reaches ``target``, falls back to the admissible candidate with the
    highest achievable balanced accuracy, narrower band breaking a tie -- narrower because a
    fallback band already failed to earn its width by meeting the target, so nothing favours
    keeping it wide.

    Returns:
        :class:`BandFit`.
    """
    # Each candidate tuple is (bacc, width, low, high, outside_count).
    best_overall: tuple[float, float, float, float, int] | None = None
    best_qualifying: tuple[float, float, float, float, int] | None = None

    for low_index, low in enumerate(_BAND_GRID):
        below = probs < low
        for high in _BAND_GRID[low_index:]:
            above = probs > high
            outside = below | above
            outside_count = int(np.count_nonzero(outside))
            if outside_count == 0 or outside_count < min_outside:
                continue

            bacc = _balanced_accuracy_outside(y[outside], above[outside])
            width = float(high - low)

            if (
                best_overall is None
                or bacc > best_overall[0] + 1e-9
                or (abs(bacc - best_overall[0]) <= 1e-9 and width < best_overall[1])
            ):
                best_overall = (bacc, width, float(low), float(high), outside_count)

            if bacc >= target and (
                best_qualifying is None
                or width > best_qualifying[1] + 1e-12
                or (abs(width - best_qualifying[1]) <= 1e-12 and bacc > best_qualifying[0])
            ):
                best_qualifying = (bacc, width, float(low), float(high), outside_count)

    chosen = best_qualifying if best_qualifying is not None else best_overall
    if chosen is None:
        # No candidate cleared the min-outside floor (or, with min_outside == 0, every
        # candidate had zero coverage outside it -- degenerate held-out data either way).
        return BandFit(low=0.5, high=0.5, outside_bacc=0.0, outside_count=0, target_met=False)
    bacc, _width, low, high, outside_count = chosen
    return BandFit(
        low=low,
        high=high,
        outside_bacc=bacc,
        outside_count=outside_count,
        target_met=best_qualifying is not None,
    )


@dataclass(frozen=True)
class Band:
    """The abstain band: below ``low`` predicts real, above ``high`` predicts fake."""

    low: float
    high: float


@dataclass(frozen=True)
class FitInfo:
    """Provenance of the data a :class:`Fuser` was fitted on."""

    n_images: int
    n_fake: int
    n_real: int
    sources: list[str]
    levels: list[str]
    #: The effective minimum-outside-images floor used by :func:`_fit_band` for this fit
    #: (see :func:`_min_outside_floor`) -- not the raw ``min_outside_fraction`` /
    #: ``min_outside_count`` arguments, but what they resolved to against this fuser's
    #: actual held-out split size. Defaults to 0 when loaded from a file written before this
    #: field existed, i.e. before the minimum-support floor was enforced at all.
    min_outside_count: int = 0


@dataclass(frozen=True)
class FuserMetrics:
    """Metrics reported alongside a fitted :class:`Fuser`.

    ``train_auc``/``holdout_auc`` are computed on the raw (pre-temperature)
    probabilities -- :func:`~imgforensics.eval.metrics.roc_auc` is invariant
    to the monotone sigmoid/temperature transform, so this is also the AUC
    of the calibrated probabilities. Both are computed per fitted *row*
    (image, level), not deduplicated per image, consistent with how the
    robustness levels are folded into the fitting data (see the module
    docstring).
    """

    train_auc: float
    holdout_auc: float
    ece_before: float
    ece_after: float
    abstain_rate: float
    outside_band_balanced_accuracy: float
    #: How many held-out images the fitted band actually left outside it -- the raw count
    #: :attr:`outside_band_balanced_accuracy` was measured on. Defaults to 0 when loaded
    #: from a file written before the minimum-support floor existed.
    outside_band_count: int = 0
    #: Whether the fitted band met ``target_balanced_accuracy`` under the minimum-support
    #: floor, or is the best-effort fallback :func:`_fit_band` returns when nothing admissible
    #: did. Defaults to ``True`` when loaded from a file written before this field existed --
    #: those files predate the floor, so the target was met the unconstrained way.
    band_target_met: bool = True


@dataclass
class Fuser:
    """A fitted calibrated stacking fuser: logistic weights, temperature, and abstain band.

    ``logit_weights``/``presence_weights`` each have one entry per
    ``detectors[i]`` -- see :meth:`feature_vector` for how a live
    ``{detector: score}`` dict is turned back into that same layout.
    """

    detectors: tuple[str, ...]
    logit_weights: np.ndarray
    presence_weights: np.ndarray
    bias: float
    temperature: float
    band: Band
    fit_info: FitInfo
    metrics: FuserMetrics
    records_sha256: dict[str, str] = field(default_factory=dict)
    format_version: int = _FORMAT_VERSION
    created: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    package_version: str = __version__

    def feature_vector(self, scores: Mapping[str, float]) -> np.ndarray:
        """Build the ``[logits | presence indicators]`` row for a live score dict.

        A detector absent from ``scores`` is imputed as abstaining: logit 0
        (score 0.5) and presence 0.0 -- the same convention
        :meth:`FusionFeatures.build` uses for a detector missing from a
        record group.
        """
        n_detectors = len(self.detectors)
        vector = np.zeros(2 * n_detectors, dtype=np.float64)
        for index, detector in enumerate(self.detectors):
            present = detector in scores
            score = scores.get(detector, 0.5) if present else 0.5
            vector[index] = logit_from_score(score)
            vector[n_detectors + index] = 1.0 if present else 0.0
        return vector

    def predict_logit(self, scores: Mapping[str, float]) -> float:
        """The raw fused logit (before temperature scaling) for ``scores``."""
        vector = self.feature_vector(scores)
        weights = np.concatenate([self.logit_weights, self.presence_weights])
        return float(vector @ weights + self.bias)

    def predict(self, scores: Mapping[str, float]) -> float:
        """The calibrated probability that the image is fake, for a ``{detector: score}`` dict."""
        z = self.predict_logit(scores) / self.temperature
        return float(_sigmoid(np.array([z]))[0])

    def predict_label(self, scores: Mapping[str, float]) -> Label:
        """``"real"``/``"fake"``/``"uncertain"`` from :meth:`predict` and :attr:`band`."""
        probability = self.predict(scores)
        if probability < self.band.low:
            return "real"
        if probability > self.band.high:
            return "fake"
        return "uncertain"

    def to_dict(self) -> dict[str, object]:
        """This fuser as a JSON-serialisable ``dict`` (see :meth:`save`)."""
        return {
            "format_version": self.format_version,
            "created": self.created,
            "package_version": self.package_version,
            "detectors": list(self.detectors),
            "weights": {
                "logit": self.logit_weights.tolist(),
                "presence": self.presence_weights.tolist(),
            },
            "bias": self.bias,
            "temperature": self.temperature,
            "band": {"low": self.band.low, "high": self.band.high},
            "fit": {
                "n_images": self.fit_info.n_images,
                "n_fake": self.fit_info.n_fake,
                "n_real": self.fit_info.n_real,
                "sources": self.fit_info.sources,
                "levels": self.fit_info.levels,
                "min_outside_count": self.fit_info.min_outside_count,
            },
            "metrics": {
                "train_auc": self.metrics.train_auc,
                "holdout_auc": self.metrics.holdout_auc,
                "ece_before": self.metrics.ece_before,
                "ece_after": self.metrics.ece_after,
                "abstain_rate": self.metrics.abstain_rate,
                "outside_band_balanced_accuracy": self.metrics.outside_band_balanced_accuracy,
                "outside_band_count": self.metrics.outside_band_count,
                "band_target_met": self.metrics.band_target_met,
            },
            "records_sha256": dict(self.records_sha256),
        }

    def save(self, path: str | Path) -> None:
        """Write this fuser as ``fuser.json`` (see the module docstring for the schema)."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Fuser:
        """Build a :class:`Fuser` from :meth:`to_dict`'s output (see :meth:`load`)."""
        weights = data["weights"]
        assert isinstance(weights, dict)
        band_data = data["band"]
        assert isinstance(band_data, dict)
        fit_data = data["fit"]
        assert isinstance(fit_data, dict)
        metrics_data = data["metrics"]
        assert isinstance(metrics_data, dict)

        return cls(
            detectors=tuple(data["detectors"]),  # type: ignore[arg-type]
            logit_weights=np.array(weights["logit"], dtype=np.float64),
            presence_weights=np.array(weights["presence"], dtype=np.float64),
            bias=float(data["bias"]),  # type: ignore[arg-type]
            temperature=float(data["temperature"]),  # type: ignore[arg-type]
            band=Band(low=float(band_data["low"]), high=float(band_data["high"])),
            fit_info=FitInfo(
                n_images=int(fit_data["n_images"]),
                n_fake=int(fit_data["n_fake"]),
                n_real=int(fit_data["n_real"]),
                sources=list(fit_data["sources"]),
                levels=list(fit_data["levels"]),
                min_outside_count=int(fit_data.get("min_outside_count", 0)),
            ),
            metrics=FuserMetrics(
                train_auc=float(metrics_data["train_auc"]),
                holdout_auc=float(metrics_data["holdout_auc"]),
                ece_before=float(metrics_data["ece_before"]),
                ece_after=float(metrics_data["ece_after"]),
                abstain_rate=float(metrics_data["abstain_rate"]),
                outside_band_balanced_accuracy=float(
                    metrics_data["outside_band_balanced_accuracy"]
                ),
                outside_band_count=int(metrics_data.get("outside_band_count", 0)),
                band_target_met=bool(metrics_data.get("band_target_met", True)),
            ),
            records_sha256=dict(data.get("records_sha256", {})),  # type: ignore[call-overload]
            format_version=int(data.get("format_version", _FORMAT_VERSION)),  # type: ignore[call-overload]
            created=str(data.get("created", "")),
            package_version=str(data.get("package_version", "")),
        )

    @classmethod
    def load(cls, path: str | Path) -> Fuser:
        """Load a fuser previously written by :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def fit_fuser(
    records: Sequence[ScoreRecord],
    *,
    detectors: Sequence[str] | None = None,
    levels: Sequence[str] | None = None,
    target_balanced_accuracy: float = 0.9,
    l2: float = 1e-2,
    seed: int = 0,
    records_sha256: Mapping[str, str] | None = None,
    min_outside_fraction: float = _MIN_OUTSIDE_FRACTION,
    min_outside_count: int = _MIN_OUTSIDE_COUNT,
) -> Fuser:
    """Fit a calibrated stacking :class:`Fuser` on ``records``.

    Builds :class:`FusionFeatures`, splits it 80/20 by hashed ``entry_path``
    (see :func:`_split_mask`), fits the logistic weights on the 80% train
    side, then fits temperature scaling and the abstain band on the 20%
    held-out side.

    Args:
        records: Score records to fit on, typically every record from one
            or more saved :class:`~imgforensics.eval.runner.BenchmarkResult`
            files.
        detectors: Feature columns; defaults to :func:`default_detectors`.
        levels: Robustness levels to include; defaults to every level
            present in ``records``.
        target_balanced_accuracy: Passed to :func:`_fit_band`.
        l2: L2 regularization strength for the logistic weights (not the bias).
        seed: Salts the deterministic train/held-out split.
        records_sha256: ``{source file: sha256}`` recorded in the artifact
            for traceability (see the CLI's ``fusion fit`` command); not
            computed here since ``records`` are already in memory.
        min_outside_fraction: Minimum fraction of the held-out split a band
            candidate must leave outside it to be admissible (see
            :func:`_min_outside_floor`); a band supported by a handful of
            held-out images is not evidence it works.
        min_outside_count: Minimum absolute count for the same floor. The
            floor actually used is ``max(min_outside_count, ceil(
            min_outside_fraction * n_holdout))``, capped at ``n_holdout``
            (recorded as :attr:`FitInfo.min_outside_count`).

    Raises:
        ValueError: if the fitting data is too small to form a non-empty
            80/20 split, or either split ends up with only one label.
    """
    features = FusionFeatures.build(records, detectors=detectors, levels=levels)

    holdout_mask = _split_mask(features.entry_paths, seed, _HOLDOUT_FRACTION)
    train_mask = ~holdout_mask
    if not np.any(train_mask) or not np.any(holdout_mask):
        raise ValueError(
            "fitting data is too small to form a non-empty 80/20 train/held-out split "
            f"({len(set(features.entry_paths))} unique image(s))"
        )

    y_train = features.y[train_mask]
    y_holdout = features.y[holdout_mask]
    if len(np.unique(y_train)) < 2:
        raise ValueError("training split has only one label; cannot fit a classifier")
    if len(np.unique(y_holdout)) < 2:
        raise ValueError(
            "held-out split has only one label; try a different --seed or provide more data"
        )

    weights, bias = _fit_logistic(features.X[train_mask], y_train, l2=l2, n_iter=_GD_ITERATIONS)
    n_detectors = features.n_detectors
    logit_weights, presence_weights = weights[:n_detectors], weights[n_detectors:]

    train_logits = features.X[train_mask] @ weights + bias
    holdout_logits = features.X[holdout_mask] @ weights + bias
    train_probs_raw = _sigmoid(train_logits)
    holdout_probs_raw = _sigmoid(holdout_logits)

    ece_before = expected_calibration_error(y_holdout, holdout_probs_raw)
    temperature = _fit_temperature(holdout_logits, y_holdout, ece_before)
    holdout_probs = _sigmoid(holdout_logits / temperature)
    ece_after = expected_calibration_error(y_holdout, holdout_probs)

    min_outside = _min_outside_floor(len(y_holdout), min_outside_fraction, min_outside_count)
    band_fit = _fit_band(
        y_holdout, holdout_probs, target_balanced_accuracy, min_outside=min_outside
    )
    inside_band = (holdout_probs >= band_fit.low) & (holdout_probs <= band_fit.high)
    abstain_rate = float(np.mean(inside_band))

    train_auc = roc_auc(y_train, train_probs_raw)
    holdout_auc = roc_auc(y_holdout, holdout_probs_raw)

    label_by_image: dict[str, float] = dict(zip(features.entry_paths, features.y, strict=True))
    n_images = len(label_by_image)
    n_fake = int(sum(1 for label in label_by_image.values() if label == 1.0))

    return Fuser(
        detectors=features.detectors,
        logit_weights=logit_weights,
        presence_weights=presence_weights,
        bias=bias,
        temperature=temperature,
        band=Band(low=band_fit.low, high=band_fit.high),
        fit_info=FitInfo(
            n_images=n_images,
            n_fake=n_fake,
            n_real=n_images - n_fake,
            sources=features.sources,
            levels=features.levels_used,
            min_outside_count=min_outside,
        ),
        metrics=FuserMetrics(
            train_auc=train_auc,
            holdout_auc=holdout_auc,
            ece_before=ece_before,
            ece_after=ece_after,
            abstain_rate=abstain_rate,
            outside_band_balanced_accuracy=band_fit.outside_bacc,
            outside_band_count=band_fit.outside_count,
            band_target_met=band_fit.target_met,
        ),
        records_sha256=dict(records_sha256) if records_sha256 else {},
    )
