"""Tests for imgforensics.eval.metrics: hand-computed cases plus a monotone-invariance test."""

from __future__ import annotations

import numpy as np
import pytest

from imgforensics.eval.metrics import (
    ImageMetrics,
    PixelMetrics,
    accuracy_at_threshold,
    average_precision,
    balanced_accuracy_at_threshold,
    best_threshold,
    brier_score,
    expected_calibration_error,
    fpr_at_threshold,
    pixel_ap,
    pixel_best_f1,
    pixel_f1,
    pixel_iou,
    roc_auc,
    tpr_at_threshold,
)

# ---------------------------------------------------------------------------
# roc_auc
# ---------------------------------------------------------------------------


def test_roc_auc_hand_computed_example() -> None:
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    assert roc_auc(y_true, scores) == pytest.approx(0.75)


def test_roc_auc_perfect_ranking() -> None:
    y_true = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert roc_auc(y_true, scores) == pytest.approx(1.0)


def test_roc_auc_inverted_ranking() -> None:
    y_true = [0, 0, 0, 1, 1, 1]
    scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    assert roc_auc(y_true, scores) == pytest.approx(0.0)


def test_roc_auc_all_tied_scores_is_half() -> None:
    y_true = [0, 1, 0, 1]
    scores = [0.5, 0.5, 0.5, 0.5]
    assert roc_auc(y_true, scores) == pytest.approx(0.5)


def test_roc_auc_ties_between_classes() -> None:
    # One positive and one negative share a score; a tie counts as half a win.
    y_true = [0, 1, 1]
    scores = [0.5, 0.5, 0.9]
    # positives: rank(0.5)=1.5 (average of ranks 1,2 shared with the negative), rank(0.9)=3
    # U = (1.5 + 3) - 2*3/2 = 4.5 - 3 = 1.5; AUC = 1.5 / (2*1) = 0.75
    assert roc_auc(y_true, scores) == pytest.approx(0.75)


def test_roc_auc_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="positive and one negative"):
        roc_auc([1, 1, 1], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="positive and one negative"):
        roc_auc([0, 0, 0], [0.1, 0.2, 0.3])


@pytest.mark.parametrize(
    "transform",
    [
        lambda x: x,
        lambda x: x**3,
        lambda x: np.exp(x),
        lambda x: np.log1p(x),
        lambda x: 5.0 * x + 100.0,
        lambda x: -1.0 / (x + 1.0),  # strictly increasing for x >= 0
    ],
)
def test_roc_auc_invariant_to_monotone_transforms(transform) -> None:
    rng = np.random.default_rng(42)
    for _ in range(20):
        n = rng.integers(5, 40)
        y_true = rng.integers(0, 2, size=n)
        if y_true.sum() == 0 or y_true.sum() == n:
            continue
        scores = rng.uniform(0.0, 3.0, size=n)
        baseline = roc_auc(y_true, scores)
        transformed = roc_auc(y_true, transform(scores))
        assert transformed == pytest.approx(baseline)


# ---------------------------------------------------------------------------
# average_precision
# ---------------------------------------------------------------------------


def test_average_precision_hand_computed_example() -> None:
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    assert average_precision(y_true, scores) == pytest.approx(5 / 6, abs=1e-4)


def test_average_precision_perfect_ranking_is_one() -> None:
    y_true = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert average_precision(y_true, scores) == pytest.approx(1.0)


def test_average_precision_inverted_ranking() -> None:
    y_true = [0, 0, 0, 1, 1, 1]
    scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    # Positives are retrieved last: recall reaches 1/3, 2/3, 1 only at ranks 4, 5, 6.
    ap = average_precision(y_true, scores)
    assert ap == pytest.approx((1 / 3) * (1 / 4) + (1 / 3) * (2 / 5) + (1 / 3) * (3 / 6))


def test_average_precision_requires_a_positive() -> None:
    with pytest.raises(ValueError, match="positive example"):
        average_precision([0, 0, 0], [0.1, 0.2, 0.3])


# ---------------------------------------------------------------------------
# threshold-based metrics
# ---------------------------------------------------------------------------


def test_accuracy_at_threshold() -> None:
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.6, 0.4, 0.9]
    # predictions at 0.5: [0, 1, 0, 1] vs y_true [0,0,1,1] -> 2/4 correct
    assert accuracy_at_threshold(y_true, scores, 0.5) == pytest.approx(0.5)


def test_accuracy_at_threshold_boundary_counts_as_positive() -> None:
    assert accuracy_at_threshold([1], [0.5], 0.5) == pytest.approx(1.0)
    assert accuracy_at_threshold([0], [0.5], 0.5) == pytest.approx(0.0)


def test_tpr_fpr_at_threshold() -> None:
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.6, 0.4, 0.9]
    assert tpr_at_threshold(y_true, scores, 0.5) == pytest.approx(0.5)  # only 0.9 >= 0.5
    assert fpr_at_threshold(y_true, scores, 0.5) == pytest.approx(0.5)  # only 0.6 >= 0.5


def test_tpr_fpr_no_positives_or_negatives_returns_zero() -> None:
    assert tpr_at_threshold([0, 0, 0], [0.1, 0.5, 0.9], 0.5) == 0.0
    assert fpr_at_threshold([1, 1, 1], [0.1, 0.5, 0.9], 0.5) == 0.0


def test_balanced_accuracy_at_threshold() -> None:
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.6, 0.4, 0.9]
    # tpr=0.5, tnr=1-0.5=0.5 -> balanced accuracy 0.5
    assert balanced_accuracy_at_threshold(y_true, scores, 0.5) == pytest.approx(0.5)


def test_best_threshold_finds_perfect_separator() -> None:
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    threshold, value = best_threshold(y_true, scores, objective="balanced_accuracy")
    assert value == pytest.approx(1.0)
    assert 0.2 < threshold <= 0.8


def test_best_threshold_unknown_objective_raises() -> None:
    with pytest.raises(ValueError, match="Unknown objective"):
        best_threshold([0, 1], [0.1, 0.9], objective="not-a-thing")


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def test_expected_calibration_error_perfectly_calibrated_is_zero() -> None:
    # Every predicted probability equals the empirical accuracy within its bin.
    y_true = [0, 0, 1, 1, 0, 1]
    probs = [0.0, 0.0, 1.0, 1.0, 0.5, 0.5]
    assert expected_calibration_error(y_true, probs, n_bins=10) == pytest.approx(0.0)


def test_expected_calibration_error_detects_miscalibration() -> None:
    y_true = [0, 0, 0, 0]
    probs = [0.9, 0.9, 0.9, 0.9]
    # accuracy in the bin is 0.0 but confidence is 0.9 -> ECE = 0.9
    assert expected_calibration_error(y_true, probs, n_bins=10) == pytest.approx(0.9)


def test_expected_calibration_error_empty_input_is_zero() -> None:
    assert expected_calibration_error([], []) == 0.0


def test_brier_score_perfect_predictions_is_zero() -> None:
    assert brier_score([0, 1, 1, 0], [0.0, 1.0, 1.0, 0.0]) == pytest.approx(0.0)


def test_brier_score_worst_case_is_one() -> None:
    assert brier_score([0, 1], [1.0, 0.0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# pixel-level metrics
# ---------------------------------------------------------------------------


def _mask_4x4() -> np.ndarray:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1  # 2x2 square in the middle: 4 positive pixels
    return mask


def test_pixel_f1_perfect_prediction() -> None:
    mask = _mask_4x4()
    prob = mask.astype(np.float64)
    assert pixel_f1(mask, prob) == pytest.approx(1.0)


def test_pixel_f1_hand_computed() -> None:
    mask = _mask_4x4()
    prob = np.zeros((4, 4), dtype=np.float64)
    prob[1:3, 1:4] = 1.0  # predicts a 2x3 region: overlaps the 2x2 mask fully, plus 2 extra pixels
    # TP=4, FP=2, FN=0 -> F1 = 2*4 / (2*4+2+0) = 8/10 = 0.8
    assert pixel_f1(mask, prob) == pytest.approx(0.8)


def test_pixel_f1_empty_mask_and_empty_prediction_is_zero() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    prob = np.zeros((4, 4), dtype=np.float64)
    assert pixel_f1(mask, prob) == 0.0
    assert pixel_iou(mask, prob) == 0.0


def test_pixel_iou_hand_computed() -> None:
    mask = _mask_4x4()
    prob = np.zeros((4, 4), dtype=np.float64)
    prob[1:3, 1:4] = 1.0
    # intersection=4, union = 4 (mask) + 2 (extra predicted) = 6 -> IoU = 4/6
    assert pixel_iou(mask, prob) == pytest.approx(4 / 6)


def test_pixel_best_f1_is_at_least_f1_at_default_threshold() -> None:
    mask = _mask_4x4()
    rng = np.random.default_rng(0)
    prob = mask.astype(np.float64) * 0.6 + rng.uniform(0, 0.1, size=mask.shape)
    assert pixel_best_f1(mask, prob) >= pixel_f1(mask, prob, threshold=0.5)


def test_pixel_ap_perfect_ranking_is_one() -> None:
    mask = _mask_4x4()
    prob = mask.astype(np.float64)
    assert pixel_ap(mask, prob) == pytest.approx(1.0)


def test_pixel_ap_empty_mask_is_zero() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    prob = np.random.default_rng(0).uniform(0, 1, size=(4, 4))
    assert pixel_ap(mask, prob) == 0.0


# ---------------------------------------------------------------------------
# result bundles
# ---------------------------------------------------------------------------


def test_image_metrics_compute_and_to_row() -> None:
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    metrics = ImageMetrics.compute(y_true, scores)
    assert metrics.auc == pytest.approx(0.75)
    row = metrics.to_row()
    assert set(row) == {
        "auc",
        "ap",
        "accuracy",
        "balanced_accuracy",
        "fpr",
        "tpr",
        "threshold",
        "ece",
        "brier",
    }


def test_pixel_metrics_compute_and_to_row() -> None:
    mask = _mask_4x4()
    prob = mask.astype(np.float64)
    metrics = PixelMetrics.compute(mask, prob)
    assert metrics.best_f1 == pytest.approx(1.0)
    row = metrics.to_row()
    assert set(row) == {"f1_at_threshold", "best_f1", "ap", "iou", "threshold"}
