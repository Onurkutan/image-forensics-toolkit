"""Plain, picklable record types shared by the sequential and process-pool
benchmark paths, plus the one function that produces a pixel record.

Split out of :mod:`imgforensics.eval.runner` so that
:mod:`imgforensics.eval.workers` (imported inside a worker process) can build
:class:`ScoreRecord`/:class:`PixelRecord` instances without importing
``runner`` itself -- ``runner`` dispatches work to ``workers``, so the
reverse import would be circular. :func:`record_pixel_metrics` lives here for
the same reason and is called by both paths, so the rule deciding *when* a
pixel record is written cannot drift between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from imgforensics.data.manifest import ManifestEntry
from imgforensics.eval.metrics import PixelMetrics
from imgforensics.eval.robustness import Perturbation, preserves_geometry


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
    """One (image, robustness level, detector) pixel-level metric bundle.

    ``level`` is the robustness level the heatmap was computed at. Only
    levels whose perturbation preserves the pixel grid
    (:func:`imgforensics.eval.robustness.preserves_geometry`) can carry one,
    since the manifest's mask is stored against the original geometry. It
    defaults to ``"clean"`` so a result file written before this field existed
    still loads with the right value -- ``clean`` was the only level recorded
    then.
    """

    entry_path: str
    detector: str
    metrics: PixelMetrics
    level: str = "clean"


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


def record_pixel_metrics(
    *,
    entry: ManifestEntry,
    perturbation: Perturbation,
    detector: str,
    heatmap: np.ndarray | None,
    root: Path,
    pixel_records: list[PixelRecord],
    missing_files: list[str],
) -> None:
    """Append a :class:`PixelRecord` when this (entry, level, detector) can carry one.

    Three conditions have to hold: the level's perturbation preserves the
    pixel grid, the entry has a mask, and the detector actually produced a
    heatmap. A mask file that cannot be read is reported in ``missing_files``
    and skipped, which never affects the score record already written for the
    same call.

    Args:
        entry: The manifest entry being scored; supplies ``path`` and
            ``mask_path``.
        perturbation: The robustness level the heatmap was computed at.
        detector: Name of the detector that produced ``heatmap``.
        heatmap: The detector's heatmap, or ``None`` when it produced none.
        root: Dataset root that ``entry.mask_path`` is relative to.
        pixel_records: List the new record is appended to, in place.
        missing_files: List an unreadable mask is reported in, in place.
    """
    if not preserves_geometry(perturbation.kind) or entry.mask_path is None or heatmap is None:
        return
    try:
        mask = _load_mask(root / entry.mask_path, heatmap.shape)
    except OSError as exc:
        missing_files.append(f"{entry.mask_path}: {exc}")
        return
    pixel_records.append(
        PixelRecord(
            entry_path=entry.path,
            detector=detector,
            metrics=PixelMetrics.compute(mask, heatmap),
            level=perturbation.name,
        )
    )
