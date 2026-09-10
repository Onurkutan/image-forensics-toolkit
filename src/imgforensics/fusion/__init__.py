"""Score fusion and explanation utilities that combine detector outputs."""

from imgforensics.fusion.evaluate import EvalRow, FusionEvaluation, evaluate_fuser
from imgforensics.fusion.explain_report import ReportPaths, build_report
from imgforensics.fusion.report import Contribution, explain, fusion_payload
from imgforensics.fusion.stacking import (
    Band,
    FitInfo,
    Fuser,
    FuserMetrics,
    FusionFeatures,
    fit_fuser,
)

__all__ = [
    "Band",
    "Contribution",
    "EvalRow",
    "FitInfo",
    "FusionEvaluation",
    "FusionFeatures",
    "Fuser",
    "FuserMetrics",
    "ReportPaths",
    "build_report",
    "evaluate_fuser",
    "explain",
    "fit_fuser",
    "fusion_payload",
]
