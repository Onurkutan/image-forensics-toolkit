"""Error Level Analysis (ELA): a JPEG re-compression residual heatmap."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image

from imgforensics.core.base import BaseDetector
from imgforensics.core.image import ForensicImage
from imgforensics.core.registry import register
from imgforensics.core.types import DetectionResult, label_from_score

DEFAULT_QUALITY = 95


def _diff_max(rgb: Image.Image, quality: int) -> np.ndarray:
    """Unscaled per-pixel max-over-channel absolute difference between ``rgb``
    and its JPEG re-encoding at ``quality``.

    Returns a float32 HxW array of raw (0-255 range) differences, with no
    percentile scaling or clipping applied.
    """
    rgb = rgb.convert("RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as recompressed:
        recompressed_arr = np.asarray(recompressed.convert("RGB"), dtype=np.float32)

    original_arr = np.asarray(rgb, dtype=np.float32)
    return np.abs(original_arr - recompressed_arr).max(axis=2)


def _scale_diff(diff_max: np.ndarray) -> np.ndarray:
    """Scale a raw difference map so its 99th percentile maps to 1.0, clipped
    to [0, 1] (a divide-by-zero when the 99th-percentile difference is ~0,
    e.g. a flat image, yields an all-zero map instead of raising)."""
    p99 = float(np.percentile(diff_max, 99))
    scaled = np.zeros_like(diff_max, dtype=np.float32) if p99 <= 1e-6 else diff_max / p99
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def _block_inhomogeneity(diff_max: np.ndarray, block_size: int = 16) -> float:
    """Measure how unevenly the raw (unscaled) difference is spread across
    the image, as a ratio of high-error blocks to typical-error blocks.

    Splits ``diff_max`` into non-overlapping ``block_size`` x ``block_size``
    blocks (cropping any remainder row/column that doesn't fill a full
    block), takes the mean absolute difference within each block, and
    returns ``(p95 - median) / (median + 1e-6)`` over those block means. A
    value near 0 means the error is spread uniformly; a large value means a
    small fraction of blocks (e.g. a spliced-in region) stand out sharply
    from the rest. This is informational only, intended for a future fusion
    layer, and does not feed into this signal's own score.
    """
    height, width = diff_max.shape
    blocks_y, blocks_x = height // block_size, width // block_size
    if blocks_y == 0 or blocks_x == 0:
        return 0.0
    cropped = diff_max[: blocks_y * block_size, : blocks_x * block_size]
    blocks = cropped.reshape(blocks_y, block_size, blocks_x, block_size)
    block_means = blocks.mean(axis=(1, 3))
    median = float(np.median(block_means))
    p95 = float(np.percentile(block_means, 95))
    return float((p95 - median) / (median + 1e-6))


def ela_map(rgb: Image.Image, quality: int = DEFAULT_QUALITY) -> np.ndarray:
    """Compute the Error Level Analysis heatmap for an RGB image.

    Re-encodes ``rgb`` as JPEG at ``quality`` in memory, takes the per-pixel
    absolute difference against the original, reduces RGB to a single channel
    by taking the max over channels, and scales so the 99th percentile of the
    difference maps to 1.0 (clipped to [0, 1]; a divide-by-zero when the
    99th-percentile difference is ~0, e.g. a flat image, yields an all-zero
    map instead of raising).

    Returns a float32 HxW array with values in [0, 1].
    """
    return _scale_diff(_diff_max(rgb, quality))


@register("ela")
class ELASignal(BaseDetector):
    """Error Level Analysis.

    Re-encodes the image at a fixed JPEG quality and measures how much each
    region changes. Regions that were originally saved at a different
    compression level than the rest of the image (typical of a spliced-in
    or locally re-edited patch) tend to show a different error level.

    This signal is deliberately an *explanation aid*, not a standalone
    classifier: re-encoding an already-uniform image, or one that has been
    recompressed once already by a sharing platform, equalises the error
    level everywhere and destroys the signal. The score is therefore bounded
    to [0.3, 0.65] -- capped at the "uncertain" ceiling of
    :func:`~imgforensics.core.types.label_from_score` -- so ELA can never by
    itself push the label to "fake"; it should always be read together with
    the heatmap.
    """

    name = "ela"

    def __init__(self, quality: int = DEFAULT_QUALITY) -> None:
        self.quality = quality

    def predict(self, image: ForensicImage) -> DetectionResult:
        diff_max = _diff_max(image.rgb, self.quality)
        heatmap = _scale_diff(diff_max)
        mean_ela = float(heatmap.mean())
        p99_ela = float(np.percentile(heatmap, 99))
        high_region_fraction = float(np.mean(heatmap > 0.5))
        block_inhomogeneity = _block_inhomogeneity(diff_max)

        raw_score = 0.5 + 0.3 * np.tanh((high_region_fraction - 0.05) / 0.05)
        score = float(np.clip(raw_score, 0.3, 0.65))
        label = label_from_score(score)

        details: dict[str, Any] = {
            "quality": self.quality,
            "mean_ela": mean_ela,
            "p99_ela": p99_ela,
            "high_region_fraction": high_region_fraction,
            "block_inhomogeneity": block_inhomogeneity,
            "note": "ELA is an explanation aid, not a classifier; uniform re-encoding equalises it",
        }
        return DetectionResult(
            detector=self.name,
            score=score,
            label=label,
            heatmap=heatmap,
            details=details,
        )
