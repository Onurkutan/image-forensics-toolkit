"""Score fusion and explanation utilities that combine detector outputs."""

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
    "explain",
    "fit_fuser",
]
