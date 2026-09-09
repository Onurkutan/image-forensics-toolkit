"""Copy-move forgery detection: classic block-matching duplicated-region search.

Implements the Fridrich (2003) / Popescu & Farid (2004) style pipeline: slide
overlapping blocks over the image, reduce each block to a small, JPEG-robust
feature vector, sort the feature vectors so duplicates land next to each
other, and vote on the spatial shift between matching pairs. A forged image
typically shows one dominant shift with many votes (the pasted region and its
source, both sliding along the same grid); an untouched natural photo does
not.

Known blind spots (see the class docstring for the score rules that follow
from them): a clone that was rotated or rescaled before pasting has a
different local DCT signature at every block and is invisible to this exact
axis-aligned, unscaled matcher; very heavy recompression can, in principle,
push genuinely different blocks into the same quantization bucket; and
naturally repetitive textures (tiled floors, fences, brick walls) produce
many real block matches that look like a forgery. The ``min_distance`` gate
(no two matched blocks may be near each other) and the flat-block variance
filter rule out the easy cases of the last blind spot; a periodic texture
that survives both still tends to vote for *several* competing shifts rather
than one dominant one, so a "shift consistency" guard additionally rejects
the detection outright when more than a few distinct shifts each gather
enough votes on their own (see ``_MAX_DISTINCT_DOMINANT_SHIFTS``).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from imgforensics.core.base import BaseDetector
from imgforensics.core.image import ForensicImage
from imgforensics.core.registry import register
from imgforensics.core.types import DetectionResult, Label
from imgforensics.utils.image_io import to_numpy

# Working-resolution cap: the block search runs on a grayscale copy
# downscaled so its longer side is at most this many pixels (details["scale"]
# records the factor actually used, 1.0 when no downscale was needed). Block
# matching cost scales with pixel count, so this is what keeps the signal
# inside the performance budget on multi-megapixel images.
_MAX_WORKING_SIDE = 1024

_BLOCK = 16
_STEP = 4
_NEIGHBORS = 5
_MIN_DISTANCE = 24.0
_MIN_VOTES = 12
_QUANT_STEP = 8
# Block variance (in 0-255 intensity^2 units) below which a block is
# considered near-flat (sky, a wall, an out-of-focus background) and skipped
# entirely -- such blocks would otherwise "match" almost anything.
_FLAT_VAR_THRESHOLD = 12.0
# A candidate pair's shift counts toward the heatmap when it is within this
# many pixels (both axes) of the dominant shift, or belongs to its own
# separately-accepted shift bucket (see _MIN_VOTES).
_SHIFT_TOLERANCE = 2
# Periodic-texture guard: if more than this many *distinct* shifts each reach
# _MIN_VOTES on their own, the many matches most likely come from a
# repetitive texture (tiles, a fence, a checkerboard) rather than one
# deliberate copy-paste, so the detection is rejected regardless of how many
# votes the single most popular shift received.
_MAX_DISTINCT_DOMINANT_SHIFTS = 3
_MIN_MATCHED_FRACTION_FOR_FAKE = 0.01


def _zigzag_order(n: int) -> list[tuple[int, int]]:
    """Row/col indices of an ``n``x``n`` block in classic JPEG zig-zag scan order."""
    diagonals: list[list[tuple[int, int]]] = [[] for _ in range(2 * n - 1)]
    for i in range(n):
        for j in range(n):
            s = i + j
            if s % 2 == 0:
                diagonals[s].insert(0, (i, j))
            else:
                diagonals[s].append((i, j))
    return [index for diagonal in diagonals for index in diagonal]


_ZIGZAG_9 = _zigzag_order(_BLOCK)[:9]
_ZIGZAG_ROWS = [row for row, _col in _ZIGZAG_9]
_ZIGZAG_COLS = [col for _row, col in _ZIGZAG_9]


def _dct_matrix(n: int) -> np.ndarray:
    """The orthonormal ``n``x``n`` DCT-II basis matrix ``C`` such that, for a
    2-D block ``B``, ``C @ B @ C.T`` equals ``cv2.dct(B)`` (verified in
    tests)."""
    basis = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        for i in range(n):
            basis[k, i] = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    basis[0, :] *= 1.0 / np.sqrt(n)
    basis[1:, :] *= np.sqrt(2.0 / n)
    return basis


_DCT_C16 = _dct_matrix(_BLOCK).astype(np.float32)


def _block_features(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slide a ``_BLOCK``x``_BLOCK`` window over ``gray`` with stride ``_STEP``.

    Returns ``(positions, features)``: ``positions`` is an ``(N, 2)`` int32
    array of each surviving block's top-left ``(x, y)``, and ``features`` is
    an ``(N, 9)`` int32 array of its quantized zig-zag DCT coefficients
    (``round(coef / _QUANT_STEP)``, which is what makes the feature robust to
    moderate JPEG recompression). Near-flat blocks (variance below
    ``_FLAT_VAR_THRESHOLD``) are dropped before the DCT step.

    Fully vectorized (a strided view of every overlapping block plus one
    batched einsum for the DCT of the surviving blocks) rather than a
    per-block Python loop, which is what keeps this signal fast enough for
    multi-megapixel images.
    """
    height, width = gray.shape
    if height < _BLOCK or width < _BLOCK:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0, 9), dtype=np.int32)

    windows = sliding_window_view(gray, (_BLOCK, _BLOCK))[::_STEP, ::_STEP]
    variance = windows.var(axis=(2, 3))
    ys, xs = np.nonzero(variance >= _FLAT_VAR_THRESHOLD)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0, 9), dtype=np.int32)

    blocks = windows[ys, xs].astype(np.float32)
    coeffs = np.einsum("ki,nij,jl->nkl", _DCT_C16, blocks, _DCT_C16.T, optimize=True)
    zigzag_coeffs = coeffs[:, _ZIGZAG_ROWS, _ZIGZAG_COLS]
    quantized = np.round(zigzag_coeffs / _QUANT_STEP).astype(np.int32)
    positions = np.stack([xs * _STEP, ys * _STEP], axis=1).astype(np.int32)
    return positions, quantized


def _candidate_pairs(positions: np.ndarray, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find candidate copy-move block pairs.

    Sorts blocks lexicographically by their quantized feature vector so
    identical (post-quantization) blocks land next to each other, then
    compares each block only against its next ``_NEIGHBORS`` entries in that
    order -- two blocks that are truly duplicates end up adjacent after the
    sort regardless of ``_NEIGHBORS``, since nothing else can carry the same
    key and land between them. A pair is a candidate when the features are
    exactly equal (post-quantization) and the two block positions are at
    least ``_MIN_DISTANCE`` apart, which rules out a block simply matching
    its own near-identical neighbour in a smooth (but not flat) area.

    Returns ``(shifts, pair_positions)``: ``shifts`` is an ``(M, 2)`` int32
    array of canonical ``(dx, dy)`` shift vectors -- the raster-earlier
    (smaller ``(y, x)``) block's position is always subtracted from the
    other's, so a duplicated region's two copies vote for the same bucket
    regardless of which one happened to sort first -- and ``pair_positions``
    is an ``(M, 2, 2)`` int32 array of the two blocks' ``(x, y)`` positions,
    for heatmap marking.
    """
    n = positions.shape[0]
    if n < 2:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0, 2, 2), dtype=np.int32)

    sort_keys = [features[:, col] for col in range(features.shape[1] - 1, -1, -1)]
    order = np.lexsort(sort_keys)
    sorted_positions = positions[order]
    sorted_features = features[order]

    shift_chunks: list[np.ndarray] = []
    pair_position_chunks: list[np.ndarray] = []
    for k in range(1, _NEIGHBORS + 1):
        if k >= n:
            break
        equal = np.all(sorted_features[:-k] == sorted_features[k:], axis=1)
        if not np.any(equal):
            continue
        idx = np.nonzero(equal)[0]
        pos_a = sorted_positions[idx]
        pos_b = sorted_positions[idx + k]
        distance = np.sqrt(((pos_a - pos_b) ** 2).sum(axis=1))
        far_enough = distance >= _MIN_DISTANCE
        if not np.any(far_enough):
            continue
        pos_a = pos_a[far_enough]
        pos_b = pos_b[far_enough]

        a_is_raster_first = (pos_a[:, 1] < pos_b[:, 1]) | (
            (pos_a[:, 1] == pos_b[:, 1]) & (pos_a[:, 0] <= pos_b[:, 0])
        )
        shift = np.where(a_is_raster_first[:, None], pos_b - pos_a, pos_a - pos_b)
        pair_positions = np.stack([pos_a, pos_b], axis=1)

        shift_chunks.append(shift)
        pair_position_chunks.append(pair_positions)

    if not shift_chunks:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0, 2, 2), dtype=np.int32)
    return (
        np.concatenate(shift_chunks, axis=0),
        np.concatenate(pair_position_chunks, axis=0),
    )


def _vote_shifts(shifts: np.ndarray) -> dict[tuple[int, int], int]:
    """Count how many candidate pairs vote for each exact ``(dx, dy)`` shift."""
    votes: dict[tuple[int, int], int] = {}
    for dx, dy in shifts.tolist():
        key = (int(dx), int(dy))
        votes[key] = votes.get(key, 0) + 1
    return votes


def _dominant_shift(votes: dict[tuple[int, int], int]) -> tuple[tuple[int, int], int]:
    """The most-voted shift, ties broken deterministically by shift value."""
    if not votes:
        return (0, 0), 0
    best = max(votes.items(), key=lambda item: (item[1], -item[0][0], -item[0][1]))
    return best[0], best[1]


def _build_heatmap(
    working_shape: tuple[int, int],
    shifts: np.ndarray,
    pair_positions: np.ndarray,
    votes: dict[tuple[int, int], int],
    dominant_shift: tuple[int, int],
) -> tuple[np.ndarray, float]:
    """Mark both blocks of every "qualifying" candidate pair on a working-resolution
    canvas and normalize to [0, 1] (fraction of the peak vote count).

    A pair qualifies when its shift is within ``_SHIFT_TOLERANCE`` pixels
    (both axes) of ``dominant_shift``, or its own exact shift independently
    reached ``_MIN_VOTES`` (covers the case of more than one genuinely
    duplicated region in the same image).

    Returns ``(heatmap, matched_fraction)`` where ``matched_fraction`` is the
    working-resolution fraction of pixels covered by at least one qualifying
    block (scale-independent, so it is reported as-is in ``details``).
    """
    heatmap = np.zeros(working_shape, dtype=np.float32)
    if shifts.shape[0] == 0:
        return heatmap, 0.0

    dx, dy = shifts[:, 0], shifts[:, 1]
    near_dominant = (np.abs(dx - dominant_shift[0]) <= _SHIFT_TOLERANCE) & (
        np.abs(dy - dominant_shift[1]) <= _SHIFT_TOLERANCE
    )
    high_vote = np.array([votes[(int(a), int(b))] >= _MIN_VOTES for a, b in shifts], dtype=bool)
    qualifying = near_dominant | high_vote

    for pos_a, pos_b in pair_positions[qualifying]:
        xa, ya = pos_a
        xb, yb = pos_b
        heatmap[ya : ya + _BLOCK, xa : xa + _BLOCK] += 1.0
        heatmap[yb : yb + _BLOCK, xb : xb + _BLOCK] += 1.0

    matched_fraction = float(np.mean(heatmap > 0))
    max_count = float(heatmap.max())
    if max_count > 0:
        heatmap /= max_count
    return heatmap, matched_fraction


@register("copy_move")
class CopyMoveSignal(BaseDetector):
    """Block-matching copy-move (duplicated-region) forgery detector.

    Downscales a grayscale copy of the image so its longer side is at most
    1024 px (``details["scale"]``), slides overlapping 16x16 blocks
    (``step=4``) over it, keeps a 9-value quantized zig-zag-DCT feature per
    non-flat block, sorts blocks by feature so duplicates land next to each
    other, and votes on the shift vector between matching pairs. See the
    module docstring for the full algorithm and its blind spots.

    Score rules, evaluated in order:

    1. Accepted (dominant shift has at least 12 votes, and the periodic-
       texture guard did not trigger) and ``matched_fraction >= 0.01`` ->
       score 0.85, label "fake".
    2. Accepted but the matched area is smaller than that -> score 0.60,
       label "uncertain".
    3. Not accepted -> score 0.45, label "uncertain",
       ``details["note"]`` explains that absence of a match is not evidence
       of authenticity (most of the time nothing is duplicated in the first
       place).

    Every computed field is placed in ``details``; this signal never raises,
    catching unexpected errors into ``details["error"]`` with score 0.5.
    """

    name = "copy_move"

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
        rgb = to_numpy(image.rgb)
        gray_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        orig_height, orig_width = gray_full.shape
        long_side = max(orig_height, orig_width)

        scale = 1.0 if long_side <= _MAX_WORKING_SIDE else _MAX_WORKING_SIDE / long_side
        if scale < 1.0:
            new_width = max(_BLOCK, round(orig_width * scale))
            new_height = max(_BLOCK, round(orig_height * scale))
            gray = cv2.resize(gray_full, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            gray = gray_full

        positions, features = _block_features(gray)
        shifts, pair_positions = _candidate_pairs(positions, features)
        votes = _vote_shifts(shifts)
        dominant_shift, dominant_votes = _dominant_shift(votes)

        distinct_shifts_at_min_votes = sum(1 for v in votes.values() if v >= _MIN_VOTES)
        periodic_guard_triggered = distinct_shifts_at_min_votes > _MAX_DISTINCT_DOMINANT_SHIFTS
        raw_accepted = dominant_votes >= _MIN_VOTES
        accepted = raw_accepted and not periodic_guard_triggered

        heatmap_working, matched_fraction = _build_heatmap(
            gray.shape, shifts, pair_positions, votes, dominant_shift
        )
        heatmap = cv2.resize(
            heatmap_working, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        heatmap = np.clip(heatmap, 0.0, 1.0)

        original_shift = (
            round(dominant_shift[0] / scale),
            round(dominant_shift[1] / scale),
        )

        details: dict[str, Any] = {
            "scale": scale,
            "block": _BLOCK,
            "step": _STEP,
            "candidate_pairs": int(shifts.shape[0]),
            "dominant_shift": original_shift,
            "dominant_votes": int(dominant_votes),
            "matched_fraction": matched_fraction,
            "accepted": accepted,
            "distinct_shifts_at_min_votes": distinct_shifts_at_min_votes,
            "periodic_guard_triggered": bool(raw_accepted and periodic_guard_triggered),
        }

        score: float
        label: Label
        if accepted and matched_fraction >= _MIN_MATCHED_FRACTION_FOR_FAKE:
            score, label = 0.85, "fake"
        elif accepted:
            score, label = 0.60, "uncertain"
        else:
            score, label = 0.45, "uncertain"
            details["note"] = "no duplicated regions found; absence is not evidence of authenticity"

        return DetectionResult(
            detector=self.name, score=score, label=label, heatmap=heatmap, details=details
        )
