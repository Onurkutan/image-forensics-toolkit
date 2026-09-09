"""Metrics, protocols, and a benchmark runner for detectors and localizers.

See :mod:`imgforensics.eval.metrics` for the numpy-only image-level and
pixel-level metric functions, :mod:`imgforensics.eval.robustness` for the
deterministic robustness suite, :mod:`imgforensics.eval.preprocess` for
shared crop/augmentation helpers, :mod:`imgforensics.eval.baselines` for the
trivial baseline detectors, and :mod:`imgforensics.eval.runner` for the
benchmark runner (``docs/ROADMAP.md``, section 4).
"""

from imgforensics.eval.baselines import (
    ConstantDetector,
    RandomDetector,
    SignalsMeanDetector,
    baseline_detectors,
)
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
from imgforensics.eval.preprocess import (
    AugmentationConfig,
    augment,
    center_crop,
    grid_crops,
    random_crops,
)
from imgforensics.eval.robustness import Perturbation, RobustnessSuite
from imgforensics.eval.runner import (
    BenchmarkConfig,
    BenchmarkResult,
    PixelRecord,
    ScoreRecord,
    run_benchmark,
)

__all__ = [
    "AugmentationConfig",
    "BenchmarkConfig",
    "BenchmarkResult",
    "ConstantDetector",
    "ImageMetrics",
    "Perturbation",
    "PixelMetrics",
    "PixelRecord",
    "RandomDetector",
    "RobustnessSuite",
    "ScoreRecord",
    "SignalsMeanDetector",
    "accuracy_at_threshold",
    "augment",
    "average_precision",
    "balanced_accuracy_at_threshold",
    "baseline_detectors",
    "best_threshold",
    "brier_score",
    "center_crop",
    "expected_calibration_error",
    "fpr_at_threshold",
    "grid_crops",
    "pixel_ap",
    "pixel_best_f1",
    "pixel_f1",
    "pixel_iou",
    "random_crops",
    "roc_auc",
    "run_benchmark",
    "tpr_at_threshold",
]
