"""Core data types shared across detectors, fusion, and the CLI."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

Label = Literal["real", "fake", "uncertain"]


class DetectionResult(BaseModel):
    """Result of running a single detector on a single image.

    ``score`` is the probability that the image is generated or manipulated,
    in the range [0, 1], where 0 means confidently real and 1 means
    confidently fake (AI-generated, AI-inpainted, or classically manipulated).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    detector: str
    score: float = Field(ge=0.0, le=1.0)
    label: Label
    heatmap: np.ndarray | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: float | None = None

    @field_validator("heatmap")
    @classmethod
    def _validate_heatmap(cls, value: np.ndarray | None) -> np.ndarray | None:
        if value is None:
            return value
        if value.ndim != 2:
            raise ValueError(f"heatmap must be 2-D, got shape {value.shape}")
        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"heatmap must be a float array, got dtype {value.dtype}")
        if value.size and (value.min() < 0.0 or value.max() > 1.0):
            raise ValueError("heatmap values must be within [0.0, 1.0]")
        return value


def label_from_score(score: float, low: float = 0.35, high: float = 0.65) -> Label:
    """Map a fake-probability score to a discrete label using two thresholds."""
    if score < low:
        return "real"
    if score > high:
        return "fake"
    return "uncertain"
