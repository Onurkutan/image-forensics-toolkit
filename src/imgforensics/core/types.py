"""Core data types shared across detectors, fusion, and the CLI."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

Label = Literal["real", "fake", "uncertain"]

#: What a registered tool is, for a caller that groups them or decides which
#: ones to run: a classical low-level ``signal``, an image-level ``detector``,
#: a pixel-level ``localizer``, or a ``view`` -- a map shown on every image
#: that claims no verdict at all (see :mod:`imgforensics.views`).
ToolKind = Literal["signal", "detector", "localizer", "view"]


class DetectionResult(BaseModel):
    """Result of running a single detector on a single image.

    ``score`` is the probability that the image is generated or manipulated,
    in the range [0, 1], where 0 means confidently real and 1 means
    confidently fake (AI-generated, AI-inpainted, or classically manipulated).

    The two optional maps answer different questions and are deliberately kept
    apart. ``heatmap`` is *what the detector found*: a probability per pixel,
    on the same scale as ``score``, so 0.8 means the same thing in every map
    and in every image. ``attribution`` is *where the detector looked*: a
    saliency map normalized to a maximum of 1 within its own image, which
    ranks pixels against each other and carries no meaning across images --
    it is not a probability, and a bright pixel marks evidence the detector
    used, not a claim that the pixel was manipulated. Both are 2-D float
    arrays with values in [0, 1], shaped like the image.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    detector: str
    score: float = Field(ge=0.0, le=1.0)
    label: Label
    heatmap: np.ndarray | None = None
    attribution: np.ndarray | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: float | None = None

    @field_validator("heatmap", "attribution")
    @classmethod
    def _validate_map(cls, value: np.ndarray | None, info: ValidationInfo) -> np.ndarray | None:
        if value is None:
            return value
        name = info.field_name
        if value.ndim != 2:
            raise ValueError(f"{name} must be 2-D, got shape {value.shape}")
        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"{name} must be a float array, got dtype {value.dtype}")
        if value.size and (value.min() < 0.0 or value.max() > 1.0):
            raise ValueError(f"{name} values must be within [0.0, 1.0]")
        return value


def label_from_score(score: float, low: float = 0.35, high: float = 0.65) -> Label:
    """Map a fake-probability score to a discrete label using two thresholds."""
    if score < low:
        return "real"
    if score > high:
        return "fake"
    return "uncertain"
