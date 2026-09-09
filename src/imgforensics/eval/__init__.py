"""Metrics, protocols, and a benchmark runner for detectors and localizers.

See :mod:`imgforensics.eval.metrics` for the numpy-only image-level and
pixel-level metric functions (``docs/ROADMAP.md``, section 4).
"""

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

__all__ = [
    "ImageMetrics",
    "PixelMetrics",
    "accuracy_at_threshold",
    "average_precision",
    "balanced_accuracy_at_threshold",
    "best_threshold",
    "brier_score",
    "expected_calibration_error",
    "fpr_at_threshold",
    "pixel_ap",
    "pixel_best_f1",
    "pixel_f1",
    "pixel_iou",
    "roc_auc",
    "tpr_at_threshold",
]
