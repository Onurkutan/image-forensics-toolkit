"""Tests for imgforensics.signals.double_jpeg.DoubleJPEGSignal. Fixtures: ``natural_like_image``
(tests/conftest.py) adds per-pixel Gaussian noise, used throughout so results reflect realistic
conditions. Grid-phase tests use quality 75, not 90, because grid_strength decays into the noise
floor by q90 on this fixture (Table 1).

Table 1 -- untouched single-save JPEG, 512x512/640x480, seeds 0-4 (10 cases/quality):
grid_strength/grid_margin, misalignment rule fires 0/10 at every quality -- q60
1.730-1.832/1.409-1.693, q70 1.587-1.674/1.278-1.489, q75 1.508-1.579/1.243-1.419, q80
1.366-1.425/1.180-1.335, q85 1.205-1.248/1.100-1.178, q90 1.032-1.065/1.001-1.040, q95
1.018-1.063/1.002-1.016.

Table 2 -- cropped (3, 5) then saved as PNG, base quality q1: grid_strength/grid_margin, whether
the misalignment rule fires -- q1=60 1.727-1.833/1.408-1.664 fires, q1=75
1.506-1.581/1.243-1.422 fires, q1=85 1.204-1.249/1.100-1.177 does not fire (same noise-floor
strength as an untouched image at that quality).

Table 3 -- dq_peak_ratio, 512x512, seeds 0-2: single saves (q60-95) measure 1.02-1.51;
coarse-then-fine doubles measure 2.30-6.75 (weakest q50->q85: 2.30-2.57); reverse-order doubles
(fine-then-coarse) measure 1.05-1.25.

Limits: grid-phase strength falls below threshold by q90-95 even though the phase itself is
still reported correctly; only coarse-then-fine double compression is detected; the
secondary-grid (masked-grid) check has no demonstrated positive case on this fixture across many
q1/crop/q2/seed combinations, so only its negative (no false positives) is asserted below.
"""

from __future__ import annotations

import io
import time

from conftest import natural_like_image
from PIL import Image

from imgforensics.core.image import ForensicImage
from imgforensics.signals.double_jpeg import DoubleJPEGSignal

_SIZE = (512, 512)
_GRID_TEST_QUALITY = 75  # grid_strength decays into the noise floor by q90 on this fixture
_CROP = (3, 5)
_SEEDS = (0, 1, 2)


def _as_jpeg_bytes(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _double_compressed_bytes(base: Image.Image, q1: int, q2: int) -> bytes:
    once_bytes = _as_jpeg_bytes(base, q1)
    once = Image.open(io.BytesIO(once_bytes)).convert("RGB")
    return _as_jpeg_bytes(once, q2)


def test_single_jpeg_grid_phase_zero() -> None:
    # Uses _GRID_TEST_QUALITY (75), not 90: grid_strength decays into the noise
    # floor by q90 on this fixture (see module docstring, Table 1).
    base = natural_like_image(size=_SIZE, seed=0)
    fi = ForensicImage.from_bytes(_as_jpeg_bytes(base, _GRID_TEST_QUALITY))

    result = DoubleJPEGSignal().predict(fi)

    assert result.details["grid_phase"] == (0, 0), result.details


def test_single_jpeg_q75_not_double() -> None:
    # A single save at the grid-test quality must not be flagged as
    # double-compressed. See the dq_peak_ratio calibration in Table 3 above.
    base = natural_like_image(size=_SIZE, seed=0)
    fi = ForensicImage.from_bytes(_as_jpeg_bytes(base, 75))

    result = DoubleJPEGSignal().predict(fi)

    assert result.details["double_quantization_suspected"] is False, result.details


def test_single_jpeg_q90_not_double() -> None:
    # Composite score/label check: at this quality grid_strength is at the
    # noise floor (see module docstring) and no other rule fires, so score
    # falls through to the 0.40 default.
    base = natural_like_image(size=_SIZE, seed=0)
    fi = ForensicImage.from_bytes(_as_jpeg_bytes(base, 90))

    result = DoubleJPEGSignal().predict(fi)

    assert result.details["double_quantization_suspected"] is False, result.details
    assert result.score == 0.40
    assert result.label == "uncertain"


def test_double_compression_60_90_is_suspected() -> None:
    # q60 (coarse) -> q90 (fine), s1/s2 ratio ~4.5-5.0 at the analysed
    # positions. See the dq_peak_ratio calibration in Table 3 above.
    base = natural_like_image(size=_SIZE, seed=0)
    fi = ForensicImage.from_bytes(_double_compressed_bytes(base, 60, 90))

    result = DoubleJPEGSignal().predict(fi)

    assert result.details["double_quantization_suspected"] is True, result.details
    assert result.score == 0.60
    assert result.label == "uncertain"


def test_double_compression_50_85_is_suspected() -> None:
    # q50 (coarse) -> q85 (fine): s1/s2 ratio ~3.0-3.7, the smallest margin
    # over the threshold of any required double. See Table 3 above.
    for seed in _SEEDS:
        base = natural_like_image(size=_SIZE, seed=seed)
        fi = ForensicImage.from_bytes(_double_compressed_bytes(base, 50, 85))

        result = DoubleJPEGSignal().predict(fi)

        assert result.details["double_quantization_suspected"] is True, (seed, result.details)


def test_double_compression_reverse_order_not_suspected() -> None:
    # Documented limit: fine-then-coarse (q90 -> q60) leaves no comb to
    # find -- there is no coarser first step for the current, coarser step
    # to reveal. Measures 1.05-1.25 (Table 3), indistinguishable from a
    # single save.
    for seed in _SEEDS:
        base = natural_like_image(size=_SIZE, seed=seed)
        fi = ForensicImage.from_bytes(_double_compressed_bytes(base, 90, 60))

        result = DoubleJPEGSignal().predict(fi)

        assert result.details["double_quantization_suspected"] is False, (seed, result.details)


def test_png_input_gives_none_dq_fields() -> None:
    # A non-JPEG input (or a JPEG with no encoded original available) skips
    # the DQ check entirely rather than running it on a meaningless/absent
    # quantization table.
    fi = ForensicImage.from_pil(natural_like_image(size=_SIZE, seed=0))

    result = DoubleJPEGSignal().predict(fi)

    assert result.details["dq_peak_ratio"] is None
    assert result.details["dq_ratios_by_position"] is None
    assert result.details["double_quantization_suspected"] is None


def test_cropped_after_jpeg_then_saved_png_detects_grid_offset() -> None:
    base = natural_like_image(size=_SIZE, seed=0)
    jpeg_bytes = _as_jpeg_bytes(base, _GRID_TEST_QUALITY)
    jpeg_image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")

    cropped = jpeg_image.crop((_CROP[0], _CROP[1], jpeg_image.width, jpeg_image.height))
    png_buffer = io.BytesIO()
    cropped.save(png_buffer, format="PNG")
    fi = ForensicImage.from_bytes(png_buffer.getvalue())

    result = DoubleJPEGSignal().predict(fi)

    assert result.details["grid_phase"] == _CROP, result.details
    assert result.score == 0.75
    assert result.label == "fake"
    assert "note" in result.details
    # The margin that gates the misalignment rule is reported and clears
    # _GRID_MARGIN (1.15) comfortably here -- see Table 2 above (q1=75
    # crop: margin 1.243-1.422).
    assert "grid_margin" in result.details
    assert result.details["grid_margin"] > 1.15, result.details


def test_untouched_jpeg_q80_never_misaligned_across_seeds() -> None:
    # Regression test: an untouched single-save JPEG at q80 must never
    # report a nonzero, above-threshold grid_phase. All 5 seeds land on
    # phase (0, 0) (see Table 1 above: q80, fires 0/10).
    for seed in range(5):
        base = natural_like_image(size=_SIZE, seed=seed)
        fi = ForensicImage.from_bytes(_as_jpeg_bytes(base, 80))

        result = DoubleJPEGSignal().predict(fi)

        assert result.details["grid_phase"] == (0, 0), (seed, result.details)
        assert result.score != 0.75, (seed, result.details)
        assert result.label != "fake", (seed, result.details)


def test_grid_misalignment_rule_has_no_false_positives_across_seeds() -> None:
    # The misalignment rule run over 20 random natural_like_image seeds at
    # q80 and q85 must never fire on an untouched image (measured: 0/20 at
    # both qualities).
    for quality in (80, 85):
        false_positives = 0
        for seed in range(1000, 1020):
            base = natural_like_image(size=_SIZE, seed=seed)
            fi = ForensicImage.from_bytes(_as_jpeg_bytes(base, quality))

            result = DoubleJPEGSignal().predict(fi)

            if result.score == 0.75:
                false_positives += 1

        assert false_positives == 0, (quality, false_positives)


def test_cropped_png_grid_phase_correct_at_q60_and_q85() -> None:
    # Cropped (3, 5) then saved as PNG at q in {60, 75, 85}. q75 is covered
    # by test_cropped_after_jpeg_then_saved_png_detects_grid_offset above
    # (using _GRID_TEST_QUALITY); this test covers the other two
    # qualities. q60 fires exactly like q75; at q85 the phase is still
    # exactly correct, but grid_strength (1.20-1.25) sits in the same
    # noise-floor range an untouched image measures at that quality, so
    # the misalignment rule does not fire. See Table 2 in the module
    # docstring above for the full numbers.
    for quality, expect_fires in ((60, True), (85, False)):
        base = natural_like_image(size=_SIZE, seed=0)
        jpeg_bytes = _as_jpeg_bytes(base, quality)
        jpeg_image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")

        cropped = jpeg_image.crop((_CROP[0], _CROP[1], jpeg_image.width, jpeg_image.height))
        png_buffer = io.BytesIO()
        cropped.save(png_buffer, format="PNG")
        fi = ForensicImage.from_bytes(png_buffer.getvalue())

        result = DoubleJPEGSignal().predict(fi)

        assert result.details["grid_phase"] == _CROP, (quality, result.details)
        if expect_fires:
            assert result.score == 0.75, (quality, result.details)
            assert result.label == "fake", (quality, result.details)
        else:
            assert result.score != 0.75, (quality, result.details)
            assert result.label != "fake", (quality, result.details)


def test_cropped_png_grid_phase_correct_for_single_axis_crops() -> None:
    # Crop (1, 0) and (0, 7) at q75 (x-only and y-only crops) must each
    # report the correct phase.
    for crop in ((1, 0), (0, 7)):
        base = natural_like_image(size=_SIZE, seed=0)
        jpeg_bytes = _as_jpeg_bytes(base, _GRID_TEST_QUALITY)
        jpeg_image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")

        cropped = jpeg_image.crop((crop[0], crop[1], jpeg_image.width, jpeg_image.height))
        png_buffer = io.BytesIO()
        cropped.save(png_buffer, format="PNG")
        fi = ForensicImage.from_bytes(png_buffer.getvalue())

        result = DoubleJPEGSignal().predict(fi)

        assert result.details["grid_phase"] == crop, (crop, result.details)
        assert result.score == 0.75, (crop, result.details)
        assert result.label == "fake", (crop, result.details)


def test_untouched_png_shows_no_grid_signal() -> None:
    fi = ForensicImage.from_pil(natural_like_image(size=_SIZE, seed=1))

    result = DoubleJPEGSignal().predict(fi)

    assert abs(result.details["grid_strength"] - 1.0) < 0.3, result.details
    assert result.score == 0.45
    assert result.label == "uncertain"


def test_untouched_jpeg_does_not_trigger_secondary_grid() -> None:
    # Negative case, robust across seeds: on this fixture no positive
    # secondary-grid case was reproducible (see module docstring, Limits),
    # so only the negative is asserted here. Checked on
    # secondary_grid_detected directly rather than the composite
    # score/label, since rule 1 (grid misalignment; see
    # test_untouched_jpeg_q80_never_misaligned_across_seeds) can
    # independently affect the composite result for some seeds at this
    # quality.
    for seed in _SEEDS:
        base = natural_like_image(size=_SIZE, seed=seed)
        fi = ForensicImage.from_bytes(_as_jpeg_bytes(base, 80))

        result = DoubleJPEGSignal().predict(fi)

        assert result.details["secondary_grid_detected"] is False, (seed, result.details)


def test_double_jpeg_runs_fast_on_12mp_image() -> None:
    fi = ForensicImage.from_pil(natural_like_image(size=(3000, 4000), seed=2))

    start = time.perf_counter()
    result = DoubleJPEGSignal().predict(fi)
    elapsed = time.perf_counter() - start

    assert elapsed < 4.0, f"double_jpeg took {elapsed:.2f}s on a 12MP image"
    assert result.label in {"real", "fake", "uncertain"}


def test_double_jpeg_runs_fast_on_12mp_jpeg_image() -> None:
    # Separate from the PNG-backed test above: only a JPEG-derived
    # ForensicImage (with raw bytes available) exercises the
    # double-quantization path (open_original, full-resolution luma
    # conversion, block DCT), which is the most expensive part of this
    # signal -- worth timing on its own.
    base = natural_like_image(size=(4000, 3000), seed=2)
    fi = ForensicImage.from_bytes(_as_jpeg_bytes(base, 85))

    start = time.perf_counter()
    result = DoubleJPEGSignal().predict(fi)
    elapsed = time.perf_counter() - start

    assert elapsed < 4.0, f"double_jpeg took {elapsed:.2f}s on a 12MP JPEG image"
    assert result.details["dq_peak_ratio"] is not None
    assert result.label in {"real", "fake", "uncertain"}


def test_double_jpeg_never_raises_on_tiny_image() -> None:
    tiny = Image.new("RGB", (4, 4), color=(120, 120, 120))
    fi = ForensicImage.from_pil(tiny)

    result = DoubleJPEGSignal().predict(fi)

    assert result.label in {"real", "fake", "uncertain"}
