"""Double-JPEG detection via two independent pixel-domain checks.

**(a) Blocking-grid offset.** A JPEG codec quantizes 8x8 blocks starting at pixel (0, 0);
cropping after compression and re-saving offsets the surviving grid by the crop amount, measured
via the mean absolute first difference along each axis, binned into 8 phase buckets by pixel
index mod 8. ``grid_phase`` reports the strongest bucket as the crop itself: phase 0 means
"already aligned"; phase ``c`` means "trimming ``c`` more pixels would realign it".
``grid_strength`` is the winning bucket's energy over the mean of the other 7; ``grid_margin``
is the weaker axis's winner/runner-up ratio. A second resave re-quantizes at the new origin, so
an older, pre-crop grid can survive as a weaker secondary comb
(``grid_secondary_phase``/``_strength``); ``secondary_grid_detected`` fires when the primary
phase is (0, 0) yet that bucket is nonzero and strong.

**(b) Step-normalized double-quantization periodicity.** Histogramming *raw* rounded DCT
coefficients measures how coarse the current quantization is, not whether it was compressed
twice. This reads the JPEG's luma quantization table and, per coefficient position, forms the
*dequantization index* ``k = round(coefficient / current_step)``: a coarser first step ``s1``
before today's finer step ``s2`` makes ``k`` comb periodically with period ``s1 / s2``. The DCT
runs on the BT.601 luma of the *original* encoded pixels. Since a single save's k-histogram is
not flat, each spectrum is detrended against its own locally-smoothed baseline (41-bin average)
before the peak ratio inside ``_DQ_BAND``.

**Calibration** (``natural_like_image`` fixture; full tables in
tests/test_signals_double_jpeg.py). Grid: every untouched, single-save JPEG at quality 60-95 (5
seeds, 2 sizes -- 70 cases) reports phase (0, 0); a genuine block edge or crop clears
``grid_margin`` (>=1.13, typically 1.2-1.5) while a noise-driven near-tie sits close to 1.0.
``_GRID_STRENGTH_THRESHOLD`` (1.3) separates a JPEG-derived grid (quality <= ~85) from an
untouched PNG or fine-quality JPEG (~1.0); ``_GRID_MARGIN`` (1.15) sits below every measured
genuine margin. Secondary grid: across every q1/crop/q2 combination tried,
``grid_secondary_strength`` never exceeds ~1.24 for either an untouched or cropped-then-resaved
JPEG; ``_SECONDARY_GRID_STRENGTH_THRESHOLD`` (1.6) sits above that ceiling. Double-quantization:
single compression measures 1.02-1.51 on ``dq_peak_ratio``; coarse-then-fine doubles measure
2.30-6.75 (weakest q50->q85: 2.30-2.57); ``_DQ_RATIO_THRESHOLD`` (1.9) is the midpoint.

**Limits.** Blocking-grid detection decays into the noise floor above quality 85: a crop of a
q85-or-finer JPEG is not detected. Only coarse-then-fine double compression is detected --
reverse-order doubles (e.g. q90->q60) measure 1.05-1.25, indistinguishable from a single save --
an inherent limit, not a calibration gap. A second JPEG save after a crop can mask the older
grid (the crop-then-resave primary phase does not reliably land at (0, 0) on this fixture, nor
does ``grid_secondary_strength`` separate cropped from untouched images), so the secondary-grid
check has no demonstrated positive case on synthetic fixtures -- it is conservative.

**Score**, evaluated in order: ``grid_phase`` misaligned (nonzero, strong, clearing its margin)
-> 0.75 "fake"; ``secondary_grid_detected`` -> 0.60 "uncertain";
``double_quantization_suspected`` -> 0.60 "uncertain"; non-JPEG with no grid signal -> 0.45
"uncertain"; otherwise 0.40 "uncertain" (a strong, aligned grid on a non-JPEG file also flags
``details["jpeg_history_detected"]``). None of this proves splicing -- double compression is
common for any re-shared image, so it scores as weak, uncertain evidence alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from imgforensics.core.base import BaseDetector
from imgforensics.core.image import ForensicImage
from imgforensics.core.registry import register
from imgforensics.core.types import DetectionResult, Label
from imgforensics.utils.image_io import to_numpy

# --- (a) blocking-grid offset -------------------------------------------

# Calibrated on synthetic fixtures (see module docstring): a JPEG-derived
# image at quality <= ~85 measures above this; an untouched PNG (or a JPEG
# whose blocking has decayed into noise at very fine quality) measures near
# 1.0, consistent with published guidance of > 1.3 for JPEG q<=90 and ~1.0
# for PNG.
_GRID_STRENGTH_THRESHOLD = 1.3

# Calibrated on synthetic fixtures (see module docstring and
# tests/test_signals_double_jpeg.py for the calibration table): guards
# against a thin-margin, noise-driven near-tie between the winning phase
# bucket and its runner-up, which even a single-bucket-per-boundary
# statistic can produce under noise. A genuine block edge (untouched, at
# phase (0, 0)) or a genuine crop boundary both beat their runner-up
# bucket comfortably (>~1.15-1.4 measured); requiring the margin here
# catches the rare thin-margin case before it is reported as a confident
# misalignment.
_GRID_MARGIN = 1.15

# Set conservatively above the observed noise ceiling rather than tuned to
# a demonstrated positive case (see module docstring, Limits). Across every
# q1/crop/q2 combination tried on natural_like_image (the
# q80->crop(3,5)->q90/95/75 scenarios plus several others, up to 10 seeds
# each), secondary_strength never exceeded ~1.24 for *either* an untouched
# single JPEG or a cropped-then-resaved one -- this fixture's noise floor
# swamps whatever residual signal a masked older grid leaves behind, so no
# threshold in that range would separate genuine detections from false
# ones. This threshold sits safely above that observed ceiling so the
# check is conservative (never fires on any tested scenario, matching
# "prefer false negatives") rather than tuned to a positive case that
# could not be demonstrated.
_SECONDARY_GRID_STRENGTH_THRESHOLD = 1.6


def _first_diff_phase_energy(gray: np.ndarray, axis: int) -> np.ndarray:
    """Mean absolute first difference of ``gray`` along ``axis`` (1 =
    columns/horizontal, 0 = rows/vertical), binned into 8 phase buckets by
    ``i % 8``, where ``i`` is the index of the *left/upper* pixel of the
    differenced pair (``diff[i] = gray[i + 1] - gray[i]``).

    A 2-tap first difference maps each block boundary to exactly one bucket:
    the boundary between block-local columns 7 and 8 (0-indexed) is the pair
    ``(i=7, i+1=8)``, so it lands only in bucket 7, and every subsequent
    boundary (``i=15, 23, ...``) lands there too (``i % 8 == 7``). This
    replaced an earlier 3-tap *second*-difference stencil (nulls out smooth
    gradients, so in principle should isolate curvature at a block edge more
    cleanly) that centered its response on pixel ``i + 1`` for
    ``second[i] = gray[i+2] - 2*gray[i+1] + gray[i]``: a single boundary at
    old-columns (7, 8) falls inside *two* overlapping 3-tap windows -- the one
    centered at 7 (taps 6, 7, 8) and the one centered at 8 (taps 7, 8, 9) --
    so it spreads across raw buckets 7 and 8 (i.e. 0) with nearly equal
    energy. On real (noisy) images the argmax between those two adjacent,
    near-tied buckets flips with the noise, and after the raw-to-phase
    conversion the flip reported a spurious nonzero phase on untouched,
    single-save JPEGs (observed at quality 80; see
    tests/test_signals_double_jpeg.py for the reproduction). The first
    difference has no such overlap: a genuine block boundary contributes to
    one and only one ``i % 8`` bucket.

    Returns an 8-element array indexed by ``i % 8`` (the *raw* bucket; see
    :func:`_grid_analysis` for the conversion to the reported, crop-amount
    phase convention).
    """
    first = np.diff(gray, axis=axis)
    abs_first = np.abs(first)
    length = abs_first.shape[axis]
    left_index = np.arange(length)  # left/upper pixel of the differenced pair
    raw_bucket = left_index % 8
    energies = np.empty(8, dtype=np.float64)
    for bucket in range(8):
        selector = raw_bucket == bucket
        if not selector.any():
            energies[bucket] = 0.0
            continue
        energies[bucket] = (
            abs_first[:, selector].mean() if axis == 1 else abs_first[selector, :].mean()
        )
    return energies


# The secondary-phase search excludes only the primary bucket itself, not
# its neighborhood: the first-difference statistic (see
# :func:`_first_diff_phase_energy`) has no cross-bucket overlap by
# construction -- each sample belongs to exactly one bucket. Measuring the
# neighbor bucket's energy directly confirms there is no leakage to guard
# against: with only the primary bucket itself excluded, the runner-up
# bucket on plain, uncropped JPEGs (quality 60-95, 10 seeds) still tops out
# at ~1.24, safely below ``_SECONDARY_GRID_STRENGTH_THRESHOLD`` (1.6) -- no
# better separated by also excluding the neighbors (~1.08-1.09 with them
# excluded too).


def _strongest_bucket(
    energy: np.ndarray, exclude: frozenset[int] = frozenset()
) -> tuple[int, float, float]:
    """``(raw_bucket, strength, margin)`` of the strongest of ``energy``'s 8
    phase buckets, optionally excluding a set of buckets from the search, the
    mean used to normalize ``strength``, and the runner-up used for
    ``margin`` (used to find the secondary phase: the strongest bucket
    outside the primary one, relative to the remaining buckets' own mean).

    ``strength`` is the winning bucket's energy divided by the mean of the
    other (non-excluded) buckets -- how far the winner stands out from the
    grid's overall floor. ``margin`` is the winning bucket's energy divided
    by the *runner-up* bucket's energy -- how decisively it beat its closest
    competitor, which is what actually distinguishes a genuine single-bucket
    block-edge response from two near-tied buckets whose argmax flips with
    noise (see :func:`_first_diff_phase_energy` and ``_GRID_MARGIN``).
    """
    mask = np.ones(8, dtype=bool)
    for bucket in exclude:
        mask[bucket % 8] = False
    remaining = energy[mask]
    remaining_indices = np.nonzero(mask)[0]
    order = np.argsort(remaining)[::-1]
    raw_bucket = int(remaining_indices[order[0]])
    winner = float(remaining[order[0]])
    mean = float(remaining.mean())
    strength = winner / mean if mean > 1e-9 else 1.0
    if remaining.size > 1:
        runner_up = float(remaining[order[1]])
        margin = winner / runner_up if runner_up > 1e-9 else float("inf")
    else:
        margin = float("inf")
    return raw_bucket, strength, margin


def _grid_analysis(gray: np.ndarray) -> dict[str, Any]:
    """Primary and secondary 8x8 blocking-grid phase/strength for luma image
    ``gray``. See the module docstring for the phase convention, its
    derivation, and why a second JPEG resave leaves the older grid as a
    secondary comb rather than erasing it.

    Returns a dict with ``phase``, ``strength``, ``margin`` (the primary
    signal -- see ``_GRID_MARGIN``), and ``secondary_phase``,
    ``secondary_strength`` (the strongest phase bucket outside the primary
    bucket itself, per axis, energy normalized against the mean of the
    remaining buckets -- see the comment above ``_strongest_bucket`` for why
    only the primary bucket, not its neighborhood, needs excluding). Both
    ``strength`` values average the two axes' own strengths; ``margin`` is
    the *weaker* of the two axes' own margins (whichever axis is closer to a
    noise-driven tie is what should gate a reported misalignment).
    """
    energy_x = _first_diff_phase_energy(gray, axis=1)
    energy_y = _first_diff_phase_energy(gray, axis=0)

    raw_x, strength_x, margin_x = _strongest_bucket(energy_x)
    raw_y, strength_y, margin_y = _strongest_bucket(energy_y)
    phase_x, phase_y = (7 - raw_x) % 8, (7 - raw_y) % 8

    sec_raw_x, sec_strength_x, _ = _strongest_bucket(energy_x, exclude=frozenset({raw_x}))
    sec_raw_y, sec_strength_y, _ = _strongest_bucket(energy_y, exclude=frozenset({raw_y}))
    sec_phase_x, sec_phase_y = (7 - sec_raw_x) % 8, (7 - sec_raw_y) % 8

    return {
        "phase": (phase_x, phase_y),
        "strength": (strength_x + strength_y) / 2.0,
        "margin": min(margin_x, margin_y),
        "secondary_phase": (sec_phase_x, sec_phase_y),
        "secondary_strength": (sec_strength_x + sec_strength_y) / 2.0,
    }


# --- (b) step-normalized aligned double-quantization periodicity --------

_DCT_POSITIONS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1))
_HIST_RANGE = 64  # dequantization indices clipped to [-64, 64] -> 129 bins
_HIST_BINS = 2 * _HIST_RANGE + 1
# FFT peak search band, in rfft-output index units for a 129-point histogram
# (65 output bins). Excludes only the lowest handful of bins, where the
# smoothing baseline below is least reliable (edge-padding artifacts); see
# module docstring for why a *detrended* peak, not this band alone, is what
# actually separates single from double compression.
_DQ_BAND = (15, 64)
# Width of the moving-average baseline each spectrum is detrended against
# before the peak search (see module docstring). Wide relative to the
# 65-bin spectrum so it tracks the coefficient histogram's own smooth decay
# without being pulled up by the comb bump itself.
_DQ_SMOOTHING_WINDOW = 41
# Calibrated on natural_like_image (3 seeds; see module docstring and
# tests/test_signals_double_jpeg.py for the full table): every single
# compression (quality 60-95) measured 1.02-1.51 on this ratio; the two
# required coarse-then-fine doubles (q60->q90, q50->q85) measured 2.30-3.91,
# the weakest (q50->q85) with a ~0.4 margin below its minimum. Threshold
# sits at the midpoint.
_DQ_RATIO_THRESHOLD = 1.9


def _dct_matrix(n: int) -> np.ndarray:
    """The orthonormal ``n``x``n`` DCT-II basis matrix (``C @ B @ C.T``
    equals ``cv2.dct(B)`` for a 2-D block ``B``; see
    :mod:`imgforensics.signals.copymove` for the same construction)."""
    basis = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        for i in range(n):
            basis[k, i] = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    basis[0, :] *= 1.0 / np.sqrt(n)
    basis[1:, :] *= np.sqrt(2.0 / n)
    return basis


_DCT_C8 = _dct_matrix(8)


def _aligned_block_dct(plane: np.ndarray) -> np.ndarray:
    """8x8-aligned, non-overlapping block DCT-II of ``plane``, fully
    vectorized via one batched matrix product (any remainder row/column is
    cropped). Returns an ``(N, 8, 8)`` array of coefficients."""
    height, width = plane.shape
    blocks_y, blocks_x = height // 8, width // 8
    cropped = plane[: blocks_y * 8, : blocks_x * 8].astype(np.float64)
    blocks = cropped.reshape(blocks_y, 8, blocks_x, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    return np.einsum("ki,nij,jl->nkl", _DCT_C8, blocks, _DCT_C8.T, optimize=True)


def _jpeg_luma(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma (``0.299 R + 0.587 G + 0.114 B``) of float64 ``rgb``,
    level-shifted by -128 so the result sits on the same scale the JPEG
    codec's own DCT operates on. Deliberately not :func:`cv2.cvtColor`,
    which rounds to 8-bit gray and would reintroduce exactly the kind of
    rounding mismatch this check is trying to remove (see module
    docstring)."""
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    return 0.299 * r + 0.587 * g + 0.114 * b - 128.0


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average of ``values`` with edge-padding, same length
    as the input."""
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def _periodicity_ratio(
    histogram: np.ndarray,
    band: tuple[int, int] = _DQ_BAND,
    smoothing_window: int = _DQ_SMOOTHING_WINDOW,
) -> float:
    """Peak, within ``band``, of ``histogram``'s FFT magnitude spectrum
    *detrended* against its own locally-smoothed baseline (see module
    docstring for why a flat band-mean ratio cannot separate single from
    double compression on a step-normalized histogram, and a local baseline
    can)."""
    spectrum = np.abs(np.fft.rfft(histogram))
    baseline = np.maximum(_moving_average(spectrum, smoothing_window), 1e-9)
    residual = spectrum / baseline
    windowed = residual[band[0] : band[1]]
    return float(windowed.max()) if windowed.size else 0.0


def _double_quantization_analysis(
    luma: np.ndarray, quant_table: Sequence[int]
) -> tuple[float, dict[str, float]]:
    """``(dq_peak_ratio, ratios_by_position)`` for the step-normalized
    dequantization-index histograms of luma image ``luma``, given its
    JPEG luma ``quant_table`` (natural row-major order, 64 entries)."""
    coeffs = _aligned_block_dct(luma)
    ratios: dict[str, float] = {}
    for row, col in _DCT_POSITIONS:
        step = max(1, int(quant_table[row * 8 + col]))
        k = np.clip(np.round(coeffs[:, row, col] / step), -_HIST_RANGE, _HIST_RANGE)
        histogram, _ = np.histogram(
            k, bins=_HIST_BINS, range=(-_HIST_RANGE - 0.5, _HIST_RANGE + 0.5)
        )
        ratios[f"{row}{col}"] = _periodicity_ratio(histogram.astype(np.float64))
    return float(np.mean(list(ratios.values()))), ratios


def _jpeg_luma_and_quant_table(image: ForensicImage) -> tuple[np.ndarray, tuple[int, ...]] | None:
    """``(luma, quant_table)`` computed from ``image``'s original encoded
    bytes -- not the EXIF-transposed, already-decoded ``image.rgb`` -- so the
    pixel grid matches the JPEG codec's own coordinate frame exactly.

    Returns ``None`` when ``image`` is not JPEG-derived, carries no encoded
    original (``image.raw is None``), or the original has no readable luma
    quantization table (e.g. a corrupt or unusual encoder).
    """
    if image.format != "JPEG" or image.raw is None:
        return None
    try:
        with image.open_original() as original:
            quantization = getattr(original, "quantization", None)
            if not quantization or 0 not in quantization:
                return None
            table = tuple(int(v) for v in quantization[0])
            if len(table) != 64:
                return None
            rgb = np.array(original.convert("RGB"), dtype=np.float64)
    except Exception:
        return None
    return _jpeg_luma(rgb), table


@register("double_jpeg")
class DoubleJPEGSignal(BaseDetector):
    """Double-JPEG detection via blocking-grid offset and aligned
    double-quantization periodicity. See the module docstring for both
    checks' derivation and calibration.

    Score rules, evaluated in order:

    1. ``grid_phase != (0, 0)`` and ``grid_strength > 1.3`` and
       ``grid_margin > 1.15`` -> score 0.75, label "fake". The 8x8 blocking
       grid no longer starts at the image origin: the image was cropped (or
       a JPEG-derived region was pasted in) after JPEG compression. The
       margin term guards against a noise-driven near-tie between the
       winning phase bucket and its runner-up being reported as a confident
       misalignment (see the module docstring's Calibration section).
    2. ``secondary_grid_detected`` -> score 0.60, label "uncertain". The
       primary grid is aligned to the origin, but a second, weaker grid at
       a different phase survives underneath it -- a crop or composite that
       was JPEG-resaved afterward, with the newest save's own grid
       (correctly) reported as primary.
    3. ``double_quantization_suspected`` -> score 0.60, label "uncertain".
       Weak evidence alone -- most re-shared images have been JPEG-
       compressed more than once by ordinary sharing platforms.
    4. Not JPEG-derived and no grid signal either -> score 0.45,
       label "uncertain": no JPEG history detectable at all.
    5. Otherwise -> score 0.40, label "uncertain". If a strong-but-aligned
       blocking grid was found on a non-JPEG file, this is additionally
       flagged informationally via ``details["jpeg_history_detected"]``
       (the image was JPEG-compressed at some point before being saved in
       its current format) without changing the score.

    Every computed field is placed in ``details``; this signal never
    raises, catching unexpected errors into ``details["error"]`` with score
    0.5.
    """

    name = "double_jpeg"

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
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)

        grid = _grid_analysis(gray)
        grid_phase: tuple[int, int] = grid["phase"]
        grid_strength: float = grid["strength"]
        grid_margin: float = grid["margin"]
        grid_secondary_phase: tuple[int, int] = grid["secondary_phase"]
        grid_secondary_strength: float = grid["secondary_strength"]

        dq_peak_ratio: float | None
        dq_ratios_by_position: dict[str, float] | None
        double_quantization_suspected: bool | None

        jpeg_source = _jpeg_luma_and_quant_table(image)
        if jpeg_source is not None:
            luma, quant_table = jpeg_source
            dq_peak_ratio, dq_ratios_by_position = _double_quantization_analysis(luma, quant_table)
            double_quantization_suspected = dq_peak_ratio > _DQ_RATIO_THRESHOLD
        else:
            dq_peak_ratio = None
            dq_ratios_by_position = None
            double_quantization_suspected = None

        grid_misaligned = (
            grid_phase != (0, 0)
            and grid_strength > _GRID_STRENGTH_THRESHOLD
            and grid_margin > _GRID_MARGIN
        )
        secondary_grid_detected = (
            grid_phase == (0, 0)
            and grid_secondary_phase != (0, 0)
            and grid_secondary_strength > _SECONDARY_GRID_STRENGTH_THRESHOLD
        )
        non_jpeg_grid_signal = image.format != "JPEG" and grid_strength > _GRID_STRENGTH_THRESHOLD

        details: dict[str, Any] = {
            "grid_phase": grid_phase,
            "grid_strength": grid_strength,
            "grid_margin": grid_margin,
            "grid_secondary_phase": grid_secondary_phase,
            "grid_secondary_strength": grid_secondary_strength,
            "secondary_grid_detected": secondary_grid_detected,
            "dq_peak_ratio": dq_peak_ratio,
            "dq_ratios_by_position": dq_ratios_by_position,
            "double_quantization_suspected": double_quantization_suspected,
            "jpeg_history_detected": non_jpeg_grid_signal,
        }

        score: float
        label: Label
        if grid_misaligned:
            score, label = 0.75, "fake"
            details["note"] = "8x8 grid misaligned: cropped or composited after JPEG compression"
        elif secondary_grid_detected:
            score, label = 0.60, "uncertain"
            details["note"] = (
                "a second, offset 8x8 grid suggests a crop or composite before the last JPEG save"
            )
        elif double_quantization_suspected:
            score, label = 0.60, "uncertain"
            details["note"] = (
                "saved as JPEG at least twice; common for any re-shared image, weak evidence alone"
            )
        elif image.format != "JPEG" and grid_strength <= _GRID_STRENGTH_THRESHOLD:
            score, label = 0.45, "uncertain"
            details["note"] = "no JPEG history detectable"
        elif non_jpeg_grid_signal:
            score, label = 0.40, "uncertain"
            details["note"] = (
                "8x8 blocking artifacts found in a non-JPEG file: the image was "
                "JPEG-compressed before being saved in this format"
            )
        else:
            score, label = 0.40, "uncertain"
            details["note"] = "no evidence of grid misalignment or repeated JPEG compression"

        return DetectionResult(detector=self.name, score=score, label=label, details=details)
