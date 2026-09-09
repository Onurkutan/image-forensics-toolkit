"""imgforensics: detect AI-generated images, AI-inpainted regions, and classic manipulations."""

from imgforensics.core.base import BaseDetector
from imgforensics.core.image import ForensicImage
from imgforensics.core.registry import available, get, register
from imgforensics.core.types import DetectionResult, Label, label_from_score

__version__ = "0.1.0"

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "ForensicImage",
    "Label",
    "__version__",
    "available",
    "get",
    "label_from_score",
    "register",
]
