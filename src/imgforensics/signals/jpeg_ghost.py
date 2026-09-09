"""JPEG ghost analysis (Farid, 2009): locate a region whose most recent JPEG
save was at a different quality than the rest of the image.

The idea: re-save the whole image at a range of candidate qualities and diff
each re-save against the original. A region genuinely saved at quality ``q``
shows unusually *little* extra error when re-saved again at ``q`` -- its
"ghost". A spliced-in region that was previously compressed at a different
quality than its new surroundings gives itself away by dipping at a
different quality than the rest of the image.

This implements the textbook algorithm (re-save at ``range(50, 100, 5)``,
16x16 block averaging, per-quality-map z-scoring), refined here in three
ways, each justified by a concrete failure mode found while calibrating on
synthetic fixtures (see ``tests/test_signals_jpeg_ghost.py``):

1. **Per-block "best quality" uses the first local minimum along the quality
   axis, not the global minimum.** A block double-compressed at a coarser
   quality ``q1`` then a finer quality ``q2`` (``q1 < q2``) shows its
   genuine ghost dip at ``q1`` -- but then stays *flat*, not rising again,
   for every quality from ``q1`` up to ``q2`` (a finer re-quantization of
   already-coarse pixel data changes it very little). Blind global argmin
   over that plateau lands wherever a shrinking whole-image mean happens to
   make the numbers line up best, which measurably drifted 15-20 quality
   points from the true prior quality in testing. The first local minimum
   (a point strictly lower than both neighbours) reliably lands on the
   genuine, earlier dip instead.
2. **Depth is measured against the *dominant* quality's z-score, not the
   block's own median.** A block with an entirely ordinary, single
   compression history still shows real quality-dependent variation in its
   own error curve (it has its own real dip, at the image's true quality),
   so "distance from my own median" flags authentic blocks almost as often
   as spliced ones. Distance from the *dominant* (majority) quality's score
   isolates blocks that disagree with the image's consensus history, which
   is what actually localizes a splice; measured contrast between a spliced
   patch and its background went from about 1.3x (own-median depth) to
   effectively unbounded (dominant-relative depth, background near zero).
3. **Near-flat blocks are excluded**, the same guard :mod:`copymove` uses.
   A solid-colour or otherwise near-uniform block has tiny absolute error at
   every quality, so its z-score is dominated by rounding noise rather than
   signal; left in, they were the majority contributor to false ghosts on
   untouched images in testing.

``dominant_quality`` itself is estimated from the whole-image (not
per-block) mean squared re-save error -- the standard single-image JPEG
quality estimator, far less noisy than taking the mode of ten thousand
per-block votes; ``best_quality_histogram`` is still reported per block for
transparency.

Blind spot: this is an explanation aid, like :mod:`imgforensics.signals.ela`
(score bounded to [0.3, 0.7]). It requires the image to actually be
JPEG-derived -- a region that was never JPEG-compressed has no quantization
ghost to find, and a platform that re-encodes the *whole* image one more
time after the fact (nearly every social platform) equalises the "most
recent quality" everywhere and can hide the underlying history.
"""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
from PIL import Image

from imgforensics.core.base import BaseDetector
from imgforensics.core.image import ForensicImage
from imgforensics.core.registry import register
from imgforensics.core.types import DetectionResult, label_from_score

_QUALITIES: tuple[int, ...] = tuple(range(50, 100, 5))
_GHOST_BLOCK = 16
# Block variance (0-255 luma intensity^2) below which a block is treated as
# near-flat and excluded -- see module docstring, refinement 3.
_FLAT_VAR_THRESHOLD = 20.0
# A block's depth (see module docstring, refinement 2) must reach this to
# count toward ghost_fraction; calibrated so a genuinely spliced 128x128
# q60-into-q90 patch clears it comfortably while every untouched synthetic
# fixture tried during calibration stays well under it (see the test file).
_DEPTH_THRESHOLD = 0.4
# A block's best quality must differ from the dominant quality by at least
# this much to count as anomalous.
_DIFFERS_THRESHOLD = 10

# Working-resolution cap, same rationale as imgforensics.signals.copymove:
# ten full JPEG re-encodes of a 12-megapixel image take over two seconds on
# their own; downscaling first keeps the whole signal within budget without
# losing the block-level (not pixel-level) structure this analysis reads.
_MAX_WORKING_SIDE = 1024


def _resave_diff_map(rgb: np.ndarray, quality: int) -> np.ndarray:
    """Per-pixel squared difference (averaged over RGB channels) between
    ``rgb`` (HxWx3 uint8) and its JPEG re-encoding at ``quality``."""
    image = Image.fromarray(rgb, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as recompressed:
        recompressed_arr = np.asarray(recompressed.convert("RGB"), dtype=np.float32)
    diff = (rgb.astype(np.float32) - recompressed_arr) ** 2
    return diff.mean(axis=2)


def _block_reduce(map2d: np.ndarray, block: int, reducer: str) -> np.ndarray:
    """Reduce ``map2d`` to non-overlapping ``block``x``block`` cells (any
    remainder row/column is cropped), taking either the mean or the
    variance of each cell."""
    height, width = map2d.shape
    blocks_y, blocks_x = height // block, width // block
    cropped = map2d[: blocks_y * block, : blocks_x * block]
    reshaped = cropped.reshape(blocks_y, block, blocks_x, block)
    if reducer == "mean":
        return reshaped.mean(axis=(1, 3))
    return reshaped.var(axis=(1, 3))


def _first_local_minimum_index(stacked: np.ndarray) -> np.ndarray:
    """For each block, the index along axis 0 of the first (lowest-quality)
    local minimum of ``stacked`` (a ``(Q, bh, bw)`` array); falls back to the
    global minimum for a block whose curve has no interior local minimum
    (monotonic over the whole search range). See module docstring,
    refinement 1."""
    num_qualities = stacked.shape[0]
    is_minimum = np.zeros_like(stacked, dtype=bool)
    is_minimum[0] = stacked[0] < stacked[1]
    is_minimum[-1] = stacked[-1] < stacked[-2]
    for i in range(1, num_qualities - 1):
        is_minimum[i] = (stacked[i] < stacked[i - 1]) & (stacked[i] < stacked[i + 1])

    index_grid = np.where(is_minimum, np.arange(num_qualities)[:, None, None], num_qualities)
    first_index = index_grid.min(axis=0)
    has_local_minimum = is_minimum.any(axis=0)
    global_argmin = np.argmin(stacked, axis=0)
    return np.where(has_local_minimum, first_index, global_argmin)


def _ghost_analysis(
    rgb_image: Image.Image,
    qualities: tuple[int, ...] = _QUALITIES,
    block: int = _GHOST_BLOCK,
) -> dict[str, Any]:
    """Run the full JPEG-ghost analysis on ``rgb_image`` and return every
    intermediate result, for both :class:`JPEGGhostSignal` and direct test
    inspection (a private helper: not part of the public API, but stable
    enough to import in tests).

    Returns a dict with ``qualities``, ``best_quality_map`` (int32,
    per-block), ``depth_map`` (float32, per-block, can be negative --
    clipped to >= 0 only when building the heatmap), ``flat_mask`` (bool,
    per-block), ``best_quality_histogram`` (over non-flat blocks),
    ``dominant_quality``, ``ghost_fraction`` and ``ghost_mask``.
    """
    rgb = np.asarray(rgb_image.convert("RGB"), dtype=np.uint8)

    raw_maps: list[np.ndarray] = []
    z_maps: list[np.ndarray] = []
    for quality in qualities:
        diff_map = _resave_diff_map(rgb, quality)
        block_map = _block_reduce(diff_map, block, "mean")
        raw_maps.append(block_map)
        mean, std = float(block_map.mean()), float(block_map.std())
        z_maps.append((block_map - mean) / std if std > 1e-9 else np.zeros_like(block_map))
    raw_stack = np.stack(raw_maps, axis=0)
    z_stack = np.stack(z_maps, axis=0)

    # Whole-image (not per-block) quality estimate: the quality whose global
    # mean squared re-save error is smallest. Far less noisy than a per-block
    # vote because it averages over every pixel in the image.
    global_mean_per_quality = raw_stack.mean(axis=(1, 2))
    dominant_index = int(np.argmin(global_mean_per_quality))
    dominant_quality = qualities[dominant_index]

    best_index = _first_local_minimum_index(z_stack)
    best_quality_map = np.asarray(qualities, dtype=np.int32)[best_index]

    z_at_dominant = z_stack[dominant_index]
    z_at_best = np.take_along_axis(z_stack, best_index[None, :, :], axis=0)[0]
    depth_map = (z_at_dominant - z_at_best).astype(np.float32)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    variance_map = _block_reduce(gray, block, "var")
    flat_mask = variance_map < _FLAT_VAR_THRESHOLD

    values, counts = np.unique(best_quality_map[~flat_mask], return_counts=True)
    best_quality_histogram = {
        int(v): int(c) for v, c in zip(values.tolist(), counts.tolist(), strict=False)
    }

    differs = np.abs(best_quality_map.astype(np.int64) - dominant_quality) >= _DIFFERS_THRESHOLD
    ghost_mask = differs & (depth_map >= _DEPTH_THRESHOLD) & ~flat_mask
    ghost_fraction = float(ghost_mask.mean()) if ghost_mask.size else 0.0

    return {
        "qualities": list(qualities),
        "best_quality_map": best_quality_map,
        "depth_map": depth_map,
        "flat_mask": flat_mask,
        "best_quality_histogram": best_quality_histogram,
        "dominant_quality": int(dominant_quality),
        "ghost_fraction": ghost_fraction,
        "ghost_mask": ghost_mask,
    }


def _scale_depth_for_heatmap(depth_map: np.ndarray) -> np.ndarray:
    """Clip ``depth_map`` to >= 0 and scale so its 99th percentile maps to
    1.0 (clipped to [0, 1]; an all-zero/near-zero map yields an all-zero
    heatmap instead of dividing by zero), matching
    :mod:`imgforensics.signals.ela`'s heatmap convention."""
    positive = np.clip(depth_map, 0.0, None)
    p99 = float(np.percentile(positive, 99))
    scaled = np.zeros_like(positive, dtype=np.float32) if p99 <= 1e-6 else positive / p99
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


@register("jpeg_ghost")
class JPEGGhostSignal(BaseDetector):
    """JPEG ghost analysis: locate a region compressed at a different quality
    than the rest of the image.

    Downscales to at most 1024 px on the longer side (``details["scale"]``)
    before re-saving at each quality in ``range(50, 100, 5)``, for the same
    performance reason as :class:`imgforensics.signals.copymove.CopyMoveSignal`.
    See the module docstring for the full algorithm, including three
    measured refinements over the textbook algorithm.

    Score: bounded to [0.3, 0.7] like ELA (an explanation aid, not a
    standalone classifier): ``0.5 + 0.2 * tanh((ghost_fraction - 0.03) /
    0.03)``, clipped.

    Every computed field is placed in ``details``; this signal never raises,
    catching unexpected errors into ``details["error"]`` with score 0.5.
    """

    name = "jpeg_ghost"

    def __init__(self, qualities: tuple[int, ...] = _QUALITIES) -> None:
        self.qualities = qualities

    def predict(self, image: ForensicImage) -> DetectionResult:
        try:
            return self._predict(image)
        except Exception as exc:  # never raise: record and abstain
            return DetectionResult(
                detector=self.name,
                score=0.5,
                label="uncertain",
                details={"error": f"{type(exc).__name__}: {exc}"},
            )

    def _predict(self, image: ForensicImage) -> DetectionResult:
        rgb_image = image.rgb
        orig_width, orig_height = rgb_image.width, rgb_image.height

        if min(orig_width, orig_height) < _GHOST_BLOCK:
            return DetectionResult(
                detector=self.name,
                score=0.5,
                label="uncertain",
                details={
                    "reason": (
                        f"image too small ({orig_width}x{orig_height}); jpeg_ghost needs "
                        f"at least one {_GHOST_BLOCK}x{_GHOST_BLOCK} block"
                    )
                },
            )

        long_side = max(orig_width, orig_height)
        scale = 1.0 if long_side <= _MAX_WORKING_SIDE else _MAX_WORKING_SIDE / long_side
        if scale < 1.0:
            new_width = max(_GHOST_BLOCK, round(orig_width * scale))
            new_height = max(_GHOST_BLOCK, round(orig_height * scale))
            working_image = rgb_image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        else:
            working_image = rgb_image

        analysis = _ghost_analysis(working_image, self.qualities)

        heatmap_working = _scale_depth_for_heatmap(analysis["depth_map"])
        heatmap = cv2.resize(
            heatmap_working, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        heatmap = np.clip(heatmap, 0.0, 1.0)

        ghost_fraction = analysis["ghost_fraction"]
        raw_score = 0.5 + 0.2 * np.tanh((ghost_fraction - 0.03) / 0.03)
        score = float(np.clip(raw_score, 0.3, 0.7))
        label = label_from_score(score)

        details: dict[str, Any] = {
            "scale": scale,
            "qualities": analysis["qualities"],
            "best_quality_histogram": analysis["best_quality_histogram"],
            "dominant_quality": analysis["dominant_quality"],
            "ghost_fraction": ghost_fraction,
            "best_quality_map_shape": analysis["best_quality_map"].shape,
            "note": (
                "JPEG ghost is an explanation aid, not a classifier; requires the image to "
                "be JPEG-derived, and a uniform re-encode by any platform after the fact "
                "can hide the underlying compression history"
            ),
        }

        return DetectionResult(
            detector=self.name, score=score, label=label, heatmap=heatmap, details=details
        )
