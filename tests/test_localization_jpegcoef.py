"""Tests for the JPEG coefficient reader and its two decoding paths.

No torch, no weights, no network: this is the piece CAT-Net's DCT stream is
fed from, and it is checked against Pillow's own libjpeg two ways.

1. **Round-trip** (the strict one): dequantize the decoded coefficients,
   inverse-DCT them in numpy and compare the result to the pixels libjpeg
   decoded from the same file. Two independent implementations of the same
   transform agree to a rounding step or they do not; anything wrong with the
   Huffman decoding, the de-zigzagging, the DC prediction, the block layout
   or the quantization table shows up here as a large error.
2. **Re-quantization** (the loose one): forward-DCT libjpeg's decoded pixels,
   divide by the quantization table and round. This is the encoder's own
   operation run backwards, so it reproduces most coefficients exactly and a
   few off by one -- enough to catch a systematically mis-scaled table, which
   the round-trip alone could hide if the same error appeared on both sides.

The tables are also compared against ``Image.quantization`` directly, which
pins down the natural (de-zigzagged) ordering both this module and ``jpegio``
use.

Both checks run twice, through the ``decode`` fixture: once as
``read_luma_coefficients`` resolves itself (libjpeg when ``jpeglib`` is
installed) and once with the ``jpeglib`` import forced to fail, so the
pure-Python fallback keeps its own coverage instead of becoming dead code the
day the fast path started answering everything. On top of that,
``test_the_two_decoders_agree_*`` asserts the two return *identical* arrays,
which is the claim the fast path is allowed to exist on.
"""

from __future__ import annotations

import io
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import natural_like_image
from PIL import Image

from imgforensics.localization import _jpegcoef
from imgforensics.localization._jpegcoef import (
    JpegCoefficients,
    UnsupportedJpegError,
    read_luma_coefficients,
)

_SIZE = (131, 77)  # (width, height): neither side a multiple of 8

#: Real JPEGs to sweep, when this checkout has the dataset (it is never
#: committed, so CI skips the sweep).
_REAL_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "ITW-SM" / "0_real"
#: Files the cheap accept/refuse invariant is checked on, strided across the
#: sorted listing so every platform prefix is represented.
_REAL_FILES = 20
#: Files decoded both ways and compared array-for-array. The pure-Python
#: decoder costs seconds per megapixel, so these are the cheapest accepted
#: files rather than the first ones.
_REAL_COMPARISONS = 12


def _idct_basis() -> np.ndarray:
    """The 8-point DCT-III basis, so ``basis.T @ coefficients @ basis`` is a 2-D IDCT."""
    basis = np.zeros((8, 8), dtype=np.float64)
    for frequency in range(8):
        scale = np.sqrt(1 / 8) if frequency == 0 else np.sqrt(2 / 8)
        for position in range(8):
            basis[frequency, position] = scale * np.cos((2 * position + 1) * frequency * np.pi / 16)
    return basis


_BASIS = _idct_basis()


def _as_blocks(spatial: np.ndarray) -> np.ndarray:
    """``(H, W)`` laid out as blocks -> ``(rows, cols, 8, 8)``."""
    height, width = spatial.shape
    return spatial.reshape(height // 8, 8, width // 8, 8).transpose(0, 2, 1, 3)


def _as_spatial(blocks: np.ndarray) -> np.ndarray:
    """``(rows, cols, 8, 8)`` -> ``(rows * 8, cols * 8)``."""
    rows, cols = blocks.shape[:2]
    return blocks.transpose(0, 2, 1, 3).reshape(rows * 8, cols * 8)


def _libjpeg_luma(data: bytes) -> np.ndarray:
    """The Y channel libjpeg decodes from ``data``, without a detour through RGB."""
    with Image.open(io.BytesIO(data)) as image:
        image.draft("YCbCr", image.size)
        return np.asarray(image.convert("YCbCr"))[:, :, 0].astype(np.int32)


def _encoded(**options: object) -> bytes:
    buffer = io.BytesIO()
    natural_like_image(size=_SIZE, seed=7).save(buffer, format="JPEG", **options)
    return buffer.getvalue()


#: The encodings both paths are checked on: the subsampling ratios, the two
#: ways of writing restart markers, an optimized (non-default Huffman table)
#: file, and the quality-100 4:4:4 stream CAT-Net makes of every non-JPEG.
_ENCODINGS = [
    ("4:4:4 q90", {"quality": 90, "subsampling": 0}),
    ("4:2:0 q75", {"quality": 75, "subsampling": 2}),
    ("4:2:2 q80", {"quality": 80, "subsampling": 1}),
    ("q100", {"quality": 100, "subsampling": 0}),
    ("optimized q85", {"quality": 85, "optimize": True}),
    ("restart every row", {"quality": 85, "restart_marker_rows": 1}),
    ("restart every 7 blocks", {"quality": 85, "restart_marker_blocks": 7}),
]


@pytest.fixture(params=["as installed", "python only"])
def decode(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Callable[[bytes], JpegCoefficients]:
    """``read_luma_coefficients``, once as it resolves and once without libjpeg."""
    if request.param == "python only":
        monkeypatch.setattr(_jpegcoef, "_libjpeg", lambda: None)
    return read_luma_coefficients


@pytest.mark.parametrize(("label", "options"), _ENCODINGS)
def test_dequantizing_and_inverting_the_coefficients_reproduces_libjpegs_pixels(
    decode: Callable[[bytes], JpegCoefficients], label: str, options: dict[str, object]
) -> None:
    data = _encoded(**options)
    decoded = decode(data)

    blocks = _as_blocks(decoded.coefficients).astype(np.float64) * decoded.quantization
    pixels = np.einsum("ux,rcuv,vy->rcxy", _BASIS, blocks, _BASIS) + 128.0
    reconstructed = _as_spatial(np.clip(np.round(pixels), 0, 255).astype(np.int32))

    reference = _libjpeg_luma(data)
    assert (decoded.height, decoded.width) == (_SIZE[1], _SIZE[0])
    # The coefficient array covers the smallest 8-pixel grid holding the image.
    assert decoded.coefficients.shape == (80, 136)
    error = np.abs(reconstructed[: _SIZE[1], : _SIZE[0]] - reference)
    # One unit: our float IDCT versus libjpeg's integer one, plus the level
    # shift's rounding. Anything structurally wrong would be far larger.
    assert error.max() <= 1, f"{label}: max abs error {error.max()}"
    assert error.mean() < 0.1


@pytest.mark.parametrize(("quality", "tolerance"), [(75, 0), (95, 1)])
def test_requantizing_libjpegs_pixels_recovers_the_same_coefficients(
    decode: Callable[[bytes], JpegCoefficients], quality: int, tolerance: int
) -> None:
    data = _encoded(quality=quality, subsampling=0)
    decoded = decode(data)

    luma = _libjpeg_luma(data).astype(np.float64) - 128.0
    padded = np.zeros(decoded.coefficients.shape, dtype=np.float64)
    padded[: luma.shape[0], : luma.shape[1]] = luma
    forward = np.einsum("ux,rcxy,vy->rcuv", _BASIS, _as_blocks(padded), _BASIS)
    requantized = np.round(forward / decoded.quantization).astype(np.int32)

    # Only whole blocks: libjpeg pads a partial edge block by replicating its
    # last pixel, not with the zeros used above, so the two disagree there for
    # reasons that have nothing to do with the decoder.
    whole = (_SIZE[1] // 8, _SIZE[0] // 8)
    difference = np.abs(requantized - _as_blocks(decoded.coefficients))[: whole[0], : whole[1]]
    # At quality 75 the quantization steps are coarse enough that a rounding
    # error in the decoded pixels cannot move a coefficient at all; at 95 they
    # are down at 1-3, so one grey level is worth one coefficient step.
    assert difference.max() <= tolerance


def test_the_quantization_table_matches_pillows_in_natural_order(
    decode: Callable[[bytes], JpegCoefficients],
) -> None:
    data = _encoded(quality=72)
    with Image.open(io.BytesIO(data)) as image:
        expected = np.asarray(image.quantization[0], dtype=np.int32).reshape(8, 8)

    assert np.array_equal(decode(data).quantization, expected)


def test_a_grayscale_jpeg_decodes_through_the_single_component_path(
    decode: Callable[[bytes], JpegCoefficients],
) -> None:
    buffer = io.BytesIO()
    natural_like_image(size=_SIZE, seed=3).convert("L").save(buffer, format="JPEG", quality=88)
    data = buffer.getvalue()

    decoded = decode(data)
    blocks = _as_blocks(decoded.coefficients).astype(np.float64) * decoded.quantization
    pixels = np.einsum("ux,rcuv,vy->rcxy", _BASIS, blocks, _BASIS) + 128.0
    reconstructed = _as_spatial(np.clip(np.round(pixels), 0, 255).astype(np.int32))

    with Image.open(io.BytesIO(data)) as image:
        reference = np.asarray(image.convert("L")).astype(np.int32)
    assert np.abs(reconstructed[: _SIZE[1], : _SIZE[0]] - reference).max() <= 1


def test_a_progressive_jpeg_is_rejected_by_name(
    decode: Callable[[bytes], JpegCoefficients],
) -> None:
    data = _encoded(quality=85, progressive=True)

    with pytest.raises(UnsupportedJpegError, match="progressive"):
        decode(data)


def test_a_file_that_is_not_a_jpeg_is_rejected(
    decode: Callable[[bytes], JpegCoefficients],
) -> None:
    buffer = io.BytesIO()
    natural_like_image(size=(16, 16), seed=1).save(buffer, format="PNG")

    with pytest.raises(UnsupportedJpegError, match="not a JPEG"):
        decode(buffer.getvalue())


# --- the libjpeg fast path -----------------------------------------------------

_needs_libjpeg = pytest.mark.skipif(
    _jpegcoef._libjpeg() is None, reason="jpeglib is not installed (the 'ml' extra)"
)


def _assert_identical(data: bytes, label: str) -> None:
    """The two paths return the same arrays, dtypes, shape and dimensions."""
    fast = _jpegcoef._decode_with_libjpeg(data)
    assert fast is not None, f"{label}: the fast path refused a stream it should read"
    slow = _jpegcoef._decode_in_python(data)

    assert fast.coefficients.dtype == slow.coefficients.dtype, label
    assert fast.coefficients.shape == slow.coefficients.shape, label
    assert np.array_equal(fast.coefficients, slow.coefficients), label
    assert np.array_equal(fast.quantization, slow.quantization), label
    assert fast.quantization.dtype == slow.quantization.dtype, label
    assert (fast.height, fast.width) == (slow.height, slow.width), label


@_needs_libjpeg
@pytest.mark.parametrize(("label", "options"), _ENCODINGS)
def test_the_two_decoders_agree_bit_for_bit(label: str, options: dict[str, object]) -> None:
    _assert_identical(_encoded(**options), label)


@_needs_libjpeg
def test_the_two_decoders_agree_on_a_grayscale_jpeg() -> None:
    buffer = io.BytesIO()
    natural_like_image(size=_SIZE, seed=3).convert("L").save(buffer, format="JPEG", quality=88)

    _assert_identical(buffer.getvalue(), "grayscale q88")


@_needs_libjpeg
@pytest.mark.parametrize(
    ("label", "make"),
    [
        ("progressive", lambda: _encoded(quality=85, progressive=True)),
        ("truncated", lambda: _encoded(quality=85)[:-200]),
        ("not a JPEG", lambda: b"\x89PNG\r\n\x1a\n" + b"\x00" * 64),
        ("empty", lambda: b""),
        ("SOI only", lambda: b"\xff\xd8"),
    ],
)
def test_the_fast_path_refuses_what_it_would_read_differently(
    label: str, make: Callable[[], bytes]
) -> None:
    # Refusing means falling back, not failing: whatever the Python decoder
    # did with these streams -- raise by name, or pad a truncated one with
    # zero bits -- is still what the caller gets.
    assert _jpegcoef._decode_with_libjpeg(make()) is None, label


@_needs_libjpeg
def test_read_luma_coefficients_uses_the_fast_path_when_it_can() -> None:
    data = _encoded(quality=90, subsampling=0)

    assert _jpegcoef._reads_identically(data)
    assert np.array_equal(
        read_luma_coefficients(data).coefficients,
        _jpegcoef._decode_with_libjpeg(data).coefficients,
    )


@_needs_libjpeg
@pytest.mark.skipif(not _REAL_DIR.is_dir(), reason=f"{_REAL_DIR} is not in this checkout")
def test_the_two_decoders_agree_on_real_jpegs() -> None:
    """Sweep real in-the-wild JPEGs: platform encoders, EXIF, odd Huffman tables.

    Two passes, because the two things worth checking have opposite costs.

    The accept/refuse invariant -- the fast path answers exactly when
    :func:`_reads_identically` says it may -- is free, so it runs on
    :data:`_REAL_FILES` files strided across the sorted listing. The stride
    matters: the names sort into one block per platform, so the first twenty
    are all Facebook's. 80% of ITW-SM is progressive (X, Instagram and
    LinkedIn re-encode that way and every one of those is refused); Facebook
    writes baseline, which is why a strided sample still has files to compare.

    The array-for-array comparison costs a Python decode, seconds per
    megapixel, so it takes the :data:`_REAL_COMPARISONS` cheapest accepted
    files by size on disk -- about five seconds for a dozen of them.
    """
    paths = sorted(_REAL_DIR.glob("*.jpg"))
    assert paths, f"{_REAL_DIR} holds no .jpg files"

    step = max(1, len(paths) // _REAL_FILES)
    for path in paths[::step][:_REAL_FILES]:
        data = path.read_bytes()
        if not _jpegcoef._reads_identically(data):
            assert _jpegcoef._decode_with_libjpeg(data) is None, path.name

    compared = 0
    for path in sorted(paths, key=lambda candidate: candidate.stat().st_size):
        if compared >= _REAL_COMPARISONS:
            break
        data = path.read_bytes()
        if not _jpegcoef._reads_identically(data):
            continue
        _assert_identical(data, path.name)
        compared += 1

    assert compared == _REAL_COMPARISONS, f"only {compared} of {len(paths)} files were comparable"


def _entropy_start(data: bytes) -> int:
    """Offset of the first byte of entropy-coded data, just past the scan header."""
    position = 2
    while position < len(data):
        while data[position] == 0xFF:
            position += 1
        marker = data[position]
        position += 1
        if marker in {0xD8, 0x01} or 0xD0 <= marker <= 0xD7:
            continue
        length = int.from_bytes(data[position : position + 2], "big")
        position += length
        if marker == 0xDA:
            return position
    raise AssertionError("no scan header in the fixture")


@_needs_libjpeg
def test_corrupt_entropy_data_can_part_the_two_decoders() -> None:
    """The one divergence there is, pinned here so it is not rediscovered.

    :func:`_reads_identically` gates on the *markers*; it cannot see that the
    entropy-coded data behind them is damaged. When it is, libjpeg
    resynchronizes and this project's decoder does not, and libjpeg sometimes
    returns coefficients where the other raises. That is documented rather
    than fixed -- reading libjpeg's warnings would mean capturing C stderr,
    and a corrupt JPEG has no one right answer anyway -- so this test asserts
    the divergence exists rather than that it does not.
    """
    data = _encoded(quality=85)
    start = _entropy_start(data)

    for offset in range(start, min(start + 200, len(data) - 2)):
        damaged = bytearray(data)
        damaged[offset] ^= 0x01
        candidate = bytes(damaged)
        if not _jpegcoef._reads_identically(candidate):
            continue
        fast = _jpegcoef._decode_with_libjpeg(candidate)
        try:
            _jpegcoef._decode_in_python(candidate)
        except UnsupportedJpegError:
            if fast is not None:
                return  # libjpeg read what the Python decoder gave up on

    pytest.fail("no single-byte corruption parted the two decoders; the caveat may be stale")


@_needs_libjpeg
def test_a_non_ascii_temporary_directory_disables_the_fast_path_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """libjpeg's C ``fopen`` cannot reach such a path, so it is refused up front.

    Both module globals are set through ``monkeypatch`` so pytest puts them
    back: leaving the fast path disabled would quietly slow down every test
    after this one.
    """
    directory = tmp_path / "geçici-örnek"
    directory.mkdir()
    monkeypatch.setattr(_jpegcoef, "_LIBJPEG_DISABLED", False)
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    data = _encoded(quality=90, subsampling=0)

    with pytest.warns(RuntimeWarning, match="not an ASCII path"):
        assert _jpegcoef._decode_with_libjpeg(data) is None
    assert _jpegcoef._LIBJPEG_DISABLED

    # Refused again, and silently: the warning is a once-per-process notice,
    # not one per image.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _jpegcoef._decode_with_libjpeg(data) is None

    # And the caller still gets its coefficients, from the fallback.
    assert read_luma_coefficients(data).coefficients.shape == (80, 136)


def test_an_unexpected_libjpeg_array_shape_falls_back_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shape surprise must not escape as a bare ``ValueError``.

    ``UnsupportedJpegError`` is a ``ValueError`` subclass, so a reshape that
    raised on its own would slip straight through every ``except
    UnsupportedJpegError`` between here and the API's generic handler.
    """

    class _BadJpeg:
        Y = np.zeros((10, 17, 7, 9), dtype=np.int16)  # not 8x8 blocks
        qt = np.ones((2, 8, 8), dtype=np.uint16)
        quant_tbl_no = np.array([0, 1, 1])
        height, width = _SIZE[1], _SIZE[0]

    monkeypatch.setattr(
        _jpegcoef, "_libjpeg", lambda: SimpleNamespace(read_dct=lambda path: _BadJpeg())
    )
    data = _encoded(quality=90, subsampling=0)

    assert _jpegcoef._decode_with_libjpeg(data) is None
    assert read_luma_coefficients(data).coefficients.shape == (80, 136)
