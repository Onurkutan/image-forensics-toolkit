"""Score fusion and explanation utilities that combine detector outputs."""

from imgforensics.fusion.explain_report import ReportPaths, build_report
from imgforensics.fusion.report import Contribution, explain
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
    "FitInfo",
    "FusionFeatures",
    "Fuser",
    "FuserMetrics",
    "ReportPaths",
    "build_report",
    "explain",
    "fit_fuser",
]
