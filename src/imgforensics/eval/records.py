"""Plain, picklable record types shared by the sequential and process-pool
benchmark paths.

Split out of :mod:`imgforensics.eval.runner` so that
:mod:`imgforensics.eval.workers` (imported inside a worker process) can build
:class:`ScoreRecord`/:class:`PixelRecord` instances without importing
``runner`` itself -- ``runner`` dispatches work to ``workers``, so the
reverse import would be circular.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from imgforensics.eval.metrics import PixelMetrics


@dataclass
class ScoreRecord:
    """One (image, robustness level, detector) score."""

    entry_path: str
    label: str
    source: str
    generator: str | None
    split: str | None
    level: str
    detector: str
    score: float
    elapsed_ms: float | None


@dataclass
class PixelRecord:
    """One (image, detector) pixel-level metric bundle, computed at the clean level."""

    entry_path: str
    detector: str
    metrics: PixelMetrics


def _load_mask(mask_path: Path, heatmap_shape: tuple[int, int]) -> np.ndarray:
    """Load a binary mask from ``mask_path``, resizing (nearest) to ``heatmap_shape``."""
    with Image.open(mask_path) as mask_img:
        mask = np.asarray(mask_img.convert("L"), dtype=np.uint8) > 127
    if mask.shape != heatmap_shape:
        resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (heatmap_shape[1], heatmap_shape[0]), Image.Resampling.NEAREST
        )
        mask = np.asarray(resized) > 127
    return mask
