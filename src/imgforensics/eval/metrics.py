"""Evaluation metrics for image-level detectors and pixel-level localizers.

Pure numpy: no scikit-learn dependency (see ``docs/ROADMAP.md``, section 3,
"license-clean by default" / small-dependency design). Every function takes
1-D (or, for the pixel functions, 2-D) numpy-array-like inputs and returns a
plain Python ``float``.

**Tie handling.** :func:`roc_auc` and :func:`average_precision` are rank- and
threshold-based; both give scores tied at the same value the *average* of
the ranks/thresholds they would occupy if broken arbitrarily, rather than
depending on array order, so re-ordering equal-scored samples never changes
the result (:func:`roc_auc` in particular is invariant to any strictly
monotone transform of the scores -- see the property test in
``tests/test_eval_metrics.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ArrayLike = Any

# ---------------------------------------------------------------------------
# Image-level metrics
# ---------------------------------------------------------------------------


def roc_auc(y_true: ArrayLike, scores: ArrayLike) -> float:
    """ROC AUC via the Mann-Whitney U rank statistic.

    Ties in ``scores`` are broken by assigning every tied value the average
    of the ranks it spans, so ``AUC = (mean_rank(positives) - n_pos*(n_pos+1)/2)
    / (n_pos * n_neg)`` matches the probability that a random positive
    outranks a random negative, counting a tie as half a win. Equivalent to
    the area under the ROC curve, without needing to build the curve.

    Raises:
        ValueError: if ``y_true`` has no positive or no negative examples
            (AUC is undefined).
    """
    y_true_arr = np.asarray(y_true)
    scores_arr = np.asarray(scores, dtype=np.float64)
    n_pos = int(np.sum(y_true_arr == 1))
    n_neg = int(np.sum(y_true_arr == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc requires at least one positive and one negative example")

    ranks = _average_ranks(scores_arr)
    sum_ranks_pos = float(np.sum(ranks[y_true_arr == 1]))
    u_statistic = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_statistic / (n_pos * n_neg))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks of ``values`` in ascending order, with ties averaged."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)

    n = len(values)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        # Positions i..j (0-based) occupy ranks (i+1)..(j+1); average them.
        average_rank = (i + 1 + j + 1) / 2.0
        ranks[order[i : j + 1]] = average_rank
        i = j + 1
    return ranks


def average_precision(y_true: ArrayLike, scores: ArrayLike) -> float:
    """Average precision using the step-wise definition ``sum_n (R_n - R_{n-1}) * P_n``.

    Samples are sorted by descending score; tied scores are collapsed into a
    single threshold step (evaluated at the cumulative counts after the
    whole tied group), so the result does not depend on how ties are
    ordered. ``R_0 = 0`` by convention.

    Raises:
        ValueError: if ``y_true`` has no positive examples (precision/recall
            are undefined with nothing to retrieve).
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    n_pos = int(np.sum(y_true_arr == 1))
    if n_pos == 0:
        raise ValueError("average_precision requires at least one positive example")

    order = np.argsort(-scores_arr, kind="mergesort")
    y_sorted = y_true_arr[order]
    scores_sorted = scores_arr[order]

    distinct = np.where(np.diff(scores_sorted) != 0)[0]
    threshold_idxs = np.r_[distinct, len(y_sorted) - 1]

    true_positives = np.cumsum(y_sorted)[threshold_idxs]
    predicted_positive = threshold_idxs + 1
    false_positives = predicted_positive - true_positives

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / n_pos
    recall = np.r_[0.0, recall]

    return float(np.sum(np.diff(recall) * precision))


def accuracy_at_threshold(y_true: ArrayLike, scores: ArrayLike, threshold: float) -> float:
    """Fraction of correct predictions at ``threshold`` (score == threshold predicts positive)."""
    y_true_arr = np.asarray(y_true)
    predictions = np.asarray(scores) >= threshold
    return float(np.mean(predictions == y_true_arr))


def tpr_at_threshold(y_true: ArrayLike, scores: ArrayLike, threshold: float) -> float:
    """True positive rate (recall) at ``threshold``; ``0.0`` when there are no positives."""
    y_true_arr = np.asarray(y_true)
    positive_mask = y_true_arr == 1
    if not np.any(positive_mask):
        return 0.0
    predictions = np.asarray(scores) >= threshold
    return float(np.mean(predictions[positive_mask]))


def fpr_at_threshold(y_true: ArrayLike, scores: ArrayLike, threshold: float) -> float:
    """False positive rate at ``threshold``; ``0.0`` when there are no negatives."""
    y_true_arr = np.asarray(y_true)
    negative_mask = y_true_arr == 0
    if not np.any(negative_mask):
        return 0.0
    predictions = np.asarray(scores) >= threshold
    return float(np.mean(predictions[negative_mask]))


def balanced_accuracy_at_threshold(y_true: ArrayLike, scores: ArrayLike, threshold: float) -> float:
    """Mean of true positive rate and true negative rate at ``threshold``."""
    tpr = tpr_at_threshold(y_true, scores, threshold)
    tnr = 1.0 - fpr_at_threshold(y_true, scores, threshold)
    return float((tpr + tnr) / 2.0)


_THRESHOLD_OBJECTIVES = {
    "accuracy": accuracy_at_threshold,
    "balanced_accuracy": balanced_accuracy_at_threshold,
}


def best_threshold(
    y_true: ArrayLike, scores: ArrayLike, objective: str = "balanced_accuracy"
) -> tuple[float, float]:
    """Threshold, chosen from the unique score values, that maximizes ``objective``.

    Searching only the unique score values is sufficient because both
    supported objectives are piecewise-constant between consecutive
    distinct scores (moving the threshold without crossing a data point
    cannot change any prediction). Ties in the objective keep the first
    (lowest) threshold encountered when scanning scores in ascending order.

    Args:
        objective: ``"accuracy"`` or ``"balanced_accuracy"``.

    Returns:
        ``(threshold, objective_value)``.
    """
    try:
        metric_fn = _THRESHOLD_OBJECTIVES[objective]
    except KeyError as exc:
        available = ", ".join(sorted(_THRESHOLD_OBJECTIVES))
        raise ValueError(f"Unknown objective {objective!r}. Available: {available}") from exc

    y_true_arr = np.asarray(y_true)
    scores_arr = np.asarray(scores, dtype=np.float64)
    candidates = np.unique(scores_arr)

    best_value = -np.inf
    best_t = float(candidates[0])
    for candidate in candidates:
        value = metric_fn(y_true_arr, scores_arr, float(candidate))
        if value > best_value:
            best_value = value
            best_t = float(candidate)
    return best_t, float(best_value)


def expected_calibration_error(y_true: ArrayLike, probs: ArrayLike, n_bins: int = 15) -> float:
    """Equal-width-bin expected calibration error: ``sum_bins (n_bin/n) * |acc_bin - conf_bin|``.

    ``probs`` are expected in ``[0, 1]``. Returns ``0.0`` for an empty input.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    probs_arr = np.asarray(probs, dtype=np.float64)
    n = len(probs_arr)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probs_arr, bin_edges[1:-1], right=True)

    ece = 0.0
    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        count = int(np.sum(mask))
        if count == 0:
            continue
        accuracy = float(np.mean(y_true_arr[mask]))
        confidence = float(np.mean(probs_arr[mask]))
        ece += (count / n) * abs(accuracy - confidence)
    return float(ece)


def brier_score(y_true: ArrayLike, probs: ArrayLike) -> float:
    """Mean squared error between predicted probabilities and binary labels."""
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    probs_arr = np.asarray(probs, dtype=np.float64)
    return float(np.mean((probs_arr - y_true_arr) ** 2))


# ---------------------------------------------------------------------------
# Pixel-level metrics
# ---------------------------------------------------------------------------

_DEFAULT_PIXEL_THRESHOLDS = np.linspace(0.0, 1.0, 101)


def pixel_f1(mask: ArrayLike, prob: ArrayLike, threshold: float = 0.5) -> float:
    """Pixel-level F1 at a fixed threshold.

    ``mask`` is a binary (HxW) ground-truth map, ``prob`` an HxW probability
    map in ``[0, 1]`` thresholded at ``>= threshold``. Returns ``0.0`` (not
    NaN) when both the ground-truth mask and the thresholded prediction are
    entirely empty -- precision and recall are both 0/0 in that case, and
    scoring it as 0.0 avoids rewarding a trivial always-negative predictor
    on an empty mask.
    """
    mask_bin = np.asarray(mask).astype(bool)
    pred_bin = np.asarray(prob) >= threshold
    true_positive = int(np.sum(mask_bin & pred_bin))
    false_positive = int(np.sum(~mask_bin & pred_bin))
    false_negative = int(np.sum(mask_bin & ~pred_bin))
    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return float(2 * true_positive / denominator)


def pixel_best_f1(mask: ArrayLike, prob: ArrayLike, thresholds: ArrayLike | None = None) -> float:
    """Best :func:`pixel_f1` over a sweep of thresholds (default: 0.00 to 1.00 in 0.01 steps)."""
    threshold_values = _DEFAULT_PIXEL_THRESHOLDS if thresholds is None else np.asarray(thresholds)
    return float(max(pixel_f1(mask, prob, threshold=float(t)) for t in threshold_values))


def pixel_ap(mask: ArrayLike, prob: ArrayLike) -> float:
    """Pixel-level average precision: flattens both arrays and calls :func:`average_precision`.

    Returns ``0.0`` when the ground-truth mask has no positive pixels
    (average precision is otherwise undefined -- there is nothing to
    retrieve).
    """
    mask_flat = np.asarray(mask).astype(np.int64).ravel()
    prob_flat = np.asarray(prob, dtype=np.float64).ravel()
    if not np.any(mask_flat):
        return 0.0
    return average_precision(mask_flat, prob_flat)


def pixel_iou(mask: ArrayLike, prob: ArrayLike, threshold: float = 0.5) -> float:
    """Intersection-over-union between the ground-truth mask and the thresholded prediction.

    Returns ``0.0`` (not NaN) when both the mask and the thresholded
    prediction are entirely empty (the union is 0/0 in that case).
    """
    mask_bin = np.asarray(mask).astype(bool)
    pred_bin = np.asarray(prob) >= threshold
    intersection = int(np.sum(mask_bin & pred_bin))
    union = int(np.sum(mask_bin | pred_bin))
    if union == 0:
        return 0.0
    return float(intersection / union)


# ---------------------------------------------------------------------------
# Result bundles
# ---------------------------------------------------------------------------


@dataclass
class ImageMetrics:
    """Bundle of image-level metrics computed at a single operating threshold."""

    auc: float
    ap: float
    accuracy: float
    balanced_accuracy: float
    fpr: float
    tpr: float
    threshold: float
    ece: float
    brier: float

    @classmethod
    def compute(
        cls,
        y_true: ArrayLike,
        scores: ArrayLike,
        *,
        probs: ArrayLike | None = None,
        threshold: float | None = None,
        objective: str = "balanced_accuracy",
        n_bins: int = 15,
    ) -> ImageMetrics:
        """Compute every image-level metric.

        ``probs`` (calibrated probabilities in ``[0, 1]``, for
        :func:`expected_calibration_error` and :func:`brier_score`) defaults
        to ``scores`` when not given. ``threshold`` defaults to
        :func:`best_threshold` under ``objective``.
        """
        y_true_arr = np.asarray(y_true)
        scores_arr = np.asarray(scores, dtype=np.float64)
        probs_arr = scores_arr if probs is None else np.asarray(probs, dtype=np.float64)
        resolved_threshold = (
            best_threshold(y_true_arr, scores_arr, objective=objective)[0]
            if threshold is None
            else threshold
        )
        return cls(
            auc=roc_auc(y_true_arr, scores_arr),
            ap=average_precision(y_true_arr, scores_arr),
            accuracy=accuracy_at_threshold(y_true_arr, scores_arr, resolved_threshold),
            balanced_accuracy=balanced_accuracy_at_threshold(
                y_true_arr, scores_arr, resolved_threshold
            ),
            fpr=fpr_at_threshold(y_true_arr, scores_arr, resolved_threshold),
            tpr=tpr_at_threshold(y_true_arr, scores_arr, resolved_threshold),
            threshold=resolved_threshold,
            ece=expected_calibration_error(y_true_arr, probs_arr, n_bins=n_bins),
            brier=brier_score(y_true_arr, probs_arr),
        )

    def to_row(self) -> dict[str, float]:
        """This bundle as a flat ``{column: value}`` row, for a Markdown/CSV table."""
        return {
            "auc": self.auc,
            "ap": self.ap,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "fpr": self.fpr,
            "tpr": self.tpr,
            "threshold": self.threshold,
            "ece": self.ece,
            "brier": self.brier,
        }


@dataclass
class PixelMetrics:
    """Bundle of pixel-level metrics computed at a single operating threshold."""

    f1_at_threshold: float
    best_f1: float
    ap: float
    iou: float
    threshold: float

    @classmethod
    def compute(cls, mask: ArrayLike, prob: ArrayLike, *, threshold: float = 0.5) -> PixelMetrics:
        """Compute every pixel-level metric for one (mask, probability map) pair."""
        return cls(
            f1_at_threshold=pixel_f1(mask, prob, threshold=threshold),
            best_f1=pixel_best_f1(mask, prob),
            ap=pixel_ap(mask, prob),
            iou=pixel_iou(mask, prob, threshold=threshold),
            threshold=threshold,
        )

    def to_row(self) -> dict[str, float]:
        """This bundle as a flat ``{column: value}`` row, for a Markdown/CSV table."""
        return {
            "f1_at_threshold": self.f1_at_threshold,
            "best_f1": self.best_f1,
            "ap": self.ap,
            "iou": self.iou,
            "threshold": self.threshold,
        }
