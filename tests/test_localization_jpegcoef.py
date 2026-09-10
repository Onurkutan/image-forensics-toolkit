"""Tests for the pure-numpy JPEG coefficient reader.

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
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from conftest import natural_like_image
from PIL import Image

from imgforensics.localization._jpegcoef import (
    UnsupportedJpegError,
    read_luma_coefficients,
)

_SIZE = (131, 77)  # (width, height): neither side a multiple of 8


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


@pytest.mark.parametrize(
    ("label", "options"),
    [
        ("4:4:4 q90", {"quality": 90, "subsampling": 0}),
        ("4:2:0 q75", {"quality": 75, "subsampling": 2}),
        ("4:2:2 q80", {"quality": 80, "subsampling": 1}),
        ("q100", {"quality": 100, "subsampling": 0}),
        ("optimized q85", {"quality": 85, "optimize": True}),
        ("restart every row", {"quality": 85, "restart_marker_rows": 1}),
        ("restart every 7 blocks", {"quality": 85, "restart_marker_blocks": 7}),
    ],
)
def test_dequantizing_and_inverting_the_coefficients_reproduces_libjpegs_pixels(
    label: str, options: dict[str, object]
) -> None:
    data = _encoded(**options)
    decoded = read_luma_coefficients(data)

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
    quality: int, tolerance: int
) -> None:
    data = _encoded(quality=quality, subsampling=0)
    decoded = read_luma_coefficients(data)

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


def test_the_quantization_table_matches_pillows_in_natural_order() -> None:
    data = _encoded(quality=72)
    with Image.open(io.BytesIO(data)) as image:
        expected = np.asarray(image.quantization[0], dtype=np.int32).reshape(8, 8)

    assert np.array_equal(read_luma_coefficients(data).quantization, expected)


def test_a_grayscale_jpeg_decodes_through_the_single_component_path() -> None:
    buffer = io.BytesIO()
    natural_like_image(size=_SIZE, seed=3).convert("L").save(buffer, format="JPEG", quality=88)
    data = buffer.getvalue()

    decoded = read_luma_coefficients(data)
    blocks = _as_blocks(decoded.coefficients).astype(np.float64) * decoded.quantization
    pixels = np.einsum("ux,rcuv,vy->rcxy", _BASIS, blocks, _BASIS) + 128.0
    reconstructed = _as_spatial(np.clip(np.round(pixels), 0, 255).astype(np.int32))

    with Image.open(io.BytesIO(data)) as image:
        reference = np.asarray(image.convert("L")).astype(np.int32)
    assert np.abs(reconstructed[: _SIZE[1], : _SIZE[0]] - reference).max() <= 1


def test_a_progressive_jpeg_is_rejected_by_name() -> None:
    data = _encoded(quality=85, progressive=True)

    with pytest.raises(UnsupportedJpegError, match="progressive"):
        read_luma_coefficients(data)


def test_a_file_that_is_not_a_jpeg_is_rejected() -> None:
    buffer = io.BytesIO()
    natural_like_image(size=(16, 16), seed=1).save(buffer, format="PNG")

    with pytest.raises(UnsupportedJpegError, match="not a JPEG"):
        read_luma_coefficients(buffer.getvalue())
