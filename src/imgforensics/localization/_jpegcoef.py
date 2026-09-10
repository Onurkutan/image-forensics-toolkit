"""Quantized DCT coefficients and quantization tables of a baseline JPEG.

CAT-Net's second stream reads the numbers *inside* the JPEG stream -- the
quantized DCT coefficients and the quantization table that produced them --
not the decoded pixels. Nothing in this project's dependency set exposes
those: Pillow decodes straight to pixels and offers only
``Image.quantization``.

**Why this file exists instead of ``jpegio``.** Upstream CAT-Net reads them
with ``jpegio`` (Apache-2.0, a C extension around libjpeg). Its releases on
PyPI (latest 0.2.8, uploaded 2021-10-15, checked 2026-09-10) ship
``manylinux`` wheels for CPython 3.8, 3.9 and 3.10 only -- no Windows wheel
at all, and no wheel for any interpreter newer than 3.10. On this project's
Windows development machine it would have to build libjpeg from the sdist,
and on a Linux CI runner it would work only while the runner stays on Python
3.10. A ~250-line pure numpy/Python decoder has neither problem, so the
dependency was not added.

**What is decoded.** Baseline and extended-sequential Huffman JPEGs (SOF0 and
SOF1), including chroma subsampling and restart intervals. Progressive
(SOF2), arithmetic-coded and lossless JPEGs, and anything above 8-bit sample
precision, raise :class:`UnsupportedJpegError` -- see
:mod:`imgforensics.localization.catnet` for what the localizer does with
that.

**Ordering.** libjpeg de-zigzags on the way in: ``jdhuff.c`` writes each
coefficient to ``(*block)[jpeg_natural_order[k]]`` and ``jdmarker.c`` stores
``quantval[jpeg_natural_order[i]]``, so both the coefficient blocks and the
quantization tables that ``jpegio`` hands CAT-Net are in **natural**
(row-major frequency) order. This module matches that, which matters twice
over: the DCT stream multiplies coefficient ``(u, v)`` by quantization entry
``(u, v)``, and it also reshapes the 64 within-block positions into 64
channels, so a zigzag-ordered array would feed the trained convolutions the
wrong frequency in every channel. Pillow's ``Image.quantization`` is
de-zigzagged the same way, which is what the tests check the tables against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["JpegCoefficients", "UnsupportedJpegError", "read_luma_coefficients"]

#: Zig-zag scan position -> index into a row-major 8x8 block, i.e. libjpeg's
#: ``jpeg_natural_order``. Used both for the coefficients (entropy-coded in
#: zig-zag order) and for the DQT tables (stored in zig-zag order).
_NATURAL_ORDER = (
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
)  # fmt: skip

_SOI, _EOI, _SOS, _DQT, _DHT, _DRI = 0xD8, 0xD9, 0xDA, 0xDB, 0xC4, 0xDD
_SOF_BASELINE = (0xC0, 0xC1)
_SOF_UNSUPPORTED = {
    0xC2: "progressive",
    0xC3: "lossless",
    0xC5: "differential sequential",
    0xC6: "differential progressive",
    0xC7: "differential lossless",
    0xC9: "arithmetic-coded extended sequential",
    0xCA: "arithmetic-coded progressive",
    0xCB: "arithmetic-coded lossless",
    0xCD: "arithmetic-coded differential sequential",
    0xCE: "arithmetic-coded differential progressive",
    0xCF: "arithmetic-coded differential lossless",
}
#: Markers that stand alone: no length field and no payload follows them.
_STANDALONE = {_SOI, 0x01, *range(0xD0, 0xD8)}

_HUFFMAN_LOOKUP_BITS = 16


class UnsupportedJpegError(ValueError):
    """The file is not a JPEG this module can decode coefficients from."""


@dataclass(frozen=True)
class JpegCoefficients:
    """The luminance channel of one JPEG stream, as the encoder stored it.

    Attributes:
        coefficients: ``(rows * 8, cols * 8)`` int32 array of *quantized* DCT
            coefficients, laid out spatially -- block ``(r, c)`` occupies
            ``[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8]`` in natural frequency
            order. Cropped to the smallest 8-pixel grid containing the image,
            which is what CAT-Net's dataset does with ``jpegio``'s
            MCU-aligned array.
        quantization: The ``(8, 8)`` int32 luminance quantization table, in
            the same natural order, so ``coefficients * quantization`` (per
            block) is the dequantized transform.
        height: Image height in pixels.
        width: Image width in pixels.
    """

    coefficients: np.ndarray
    quantization: np.ndarray
    height: int
    width: int


class _Component:
    """One frame component: its sampling factors and its slice of the output."""

    __slots__ = ("identifier", "horizontal", "vertical", "quant_id", "rows", "cols", "store")

    def __init__(self, identifier: int, horizontal: int, vertical: int, quant_id: int) -> None:
        self.identifier = identifier
        self.horizontal = horizontal
        self.vertical = vertical
        self.quant_id = quant_id
        self.rows = 0
        self.cols = 0
        self.store: list[int] = []


class _BitReader:
    """MSB-first bit reader over one restart interval's entropy-coded bytes.

    Reading past the end yields zero bits rather than raising: a truncated
    final block is the encoder's problem, and libjpeg pads the same way.

    Decoding is the slowest part of a CAT-Net prediction on a small image --
    a few hundred milliseconds for a 256x256 quality-100 JPEG, which is what
    every non-JPEG input is turned into and which leaves hardly a coefficient
    zero. Pulling bytes in wider words was measured and made no reliable
    difference, so this stays the obvious version.
    """

    __slots__ = ("_data", "_position", "_bits", "_count")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._position = 0
        self._bits = 0
        self._count = 0

    def peek(self, count: int) -> int:
        """The next ``count`` bits, without consuming them."""
        while self._count < count:
            byte = self._data[self._position] if self._position < len(self._data) else 0
            self._position += 1
            self._bits = (self._bits << 8) | byte
            self._count += 8
        return (self._bits >> (self._count - count)) & ((1 << count) - 1)

    def skip(self, count: int) -> None:
        """Consume ``count`` bits already peeked at."""
        self._count -= count
        self._bits &= (1 << self._count) - 1

    def read(self, count: int) -> int:
        """Consume and return the next ``count`` bits."""
        value = self.peek(count)
        self.skip(count)
        return value


def _huffman_lookup(counts: bytes, symbols: bytes) -> list[int]:
    """Flatten one Huffman table into a 2**16-entry ``(length << 8) | symbol`` table.

    Canonical JPEG codes are at most 16 bits, so peeking a fixed 16 bits and
    indexing decodes any symbol in one step -- worth the 64 K-entry table,
    since the alternative walks up to 16 bit-by-bit comparisons per
    coefficient. Entries no code prefixes stay 0, which decodes as length 0
    and is reported as a corrupt stream.
    """
    lookup = np.zeros(1 << _HUFFMAN_LOOKUP_BITS, dtype=np.int32)
    code = 0
    index = 0
    for length in range(1, 17):
        for _ in range(counts[length - 1]):
            low = code << (_HUFFMAN_LOOKUP_BITS - length)
            lookup[low : low + (1 << (_HUFFMAN_LOOKUP_BITS - length))] = (length << 8) | symbols[
                index
            ]
            code += 1
            index += 1
        code <<= 1
    # A Python list indexes measurably faster than a numpy array element-wise,
    # and this is the decoder's innermost operation.
    return lookup.tolist()


def _extend(value: int, length: int) -> int:
    """JPEG's ``EXTEND``: sign-extend a ``length``-bit magnitude category."""
    return value if value >= (1 << (length - 1)) else value - (1 << length) + 1


def _entropy_segments(data: bytes, start: int) -> tuple[list[bytes], int]:
    """Split the entropy-coded data at ``start`` into per-restart-interval chunks.

    Byte stuffing (``FF 00``) is undone and fill bytes are dropped here, so
    the bit reader never has to know about them. Returns the chunks and the
    offset of the marker that ended the scan.
    """
    segments: list[bytes] = []
    current = bytearray()
    position = start
    size = len(data)
    while position < size:
        byte = data[position]
        if byte != 0xFF:
            current.append(byte)
            position += 1
            continue
        following = data[position + 1] if position + 1 < size else _EOI
        if following == 0x00:
            current.append(0xFF)
            position += 2
        elif following == 0xFF:  # pragma: no cover - fill byte, rare in practice
            position += 1
        elif 0xD0 <= following <= 0xD7:  # RSTn
            segments.append(bytes(current))
            current = bytearray()
            position += 2
        else:
            break
    segments.append(bytes(current))
    return segments, position


def _decode_block(
    reader: _BitReader,
    dc_lookup: list[int],
    ac_lookup: list[int],
    store: list[int],
    base: int,
    predictor: int,
) -> int:
    """Decode one 8x8 block into ``store[base:base + 64]``; return the new DC predictor."""
    packed = dc_lookup[reader.peek(_HUFFMAN_LOOKUP_BITS)]
    length = packed >> 8
    if length == 0:
        raise UnsupportedJpegError("corrupt JPEG: no Huffman code matches the DC bits")
    reader.skip(length)
    category = packed & 0xFF
    predictor += _extend(reader.read(category), category) if category else 0
    store[base] = predictor

    position = 1
    while position < 64:
        packed = ac_lookup[reader.peek(_HUFFMAN_LOOKUP_BITS)]
        length = packed >> 8
        if length == 0:
            raise UnsupportedJpegError("corrupt JPEG: no Huffman code matches the AC bits")
        reader.skip(length)
        run, category = (packed >> 4) & 0xF, packed & 0xF
        if category == 0:
            if run != 15:  # end of block
                break
            position += 16  # ZRL: sixteen zeroes
            continue
        position += run
        if position > 63:  # pragma: no cover - only reachable in a corrupt stream
            break
        store[base + _NATURAL_ORDER[position]] = _extend(reader.read(category), category)
        position += 1
    return predictor


def _decode_scan(
    segments: list[bytes],
    scan: list[tuple[_Component, list[int], list[int]]],
    mcus_x: int,
    mcus_y: int,
    restart_interval: int,
    grid: tuple[int, int, int, int],
) -> None:
    """Entropy-decode one scan into its components' ``store`` lists."""
    height, width, horizontal_max, vertical_max = grid
    reader = _BitReader(segments[0])
    segment_index = 0
    predictors = [0] * len(scan)

    if len(scan) == 1:
        # A single-component scan is not interleaved: it walks that
        # component's own block grid, which is not padded out to whole MCUs.
        component = scan[0][0]
        columns = -(-(width * component.horizontal) // (8 * horizontal_max))
        units = columns * -(-(height * component.vertical) // (8 * vertical_max))
    else:
        columns = 0  # unused; the interleaved branch walks MCUs instead
        units = mcus_x * mcus_y

    for unit in range(units):
        if restart_interval and unit and unit % restart_interval == 0:
            segment_index += 1
            if segment_index >= len(segments):  # pragma: no cover - truncated stream
                break
            reader = _BitReader(segments[segment_index])
            predictors = [0] * len(scan)

        for index, (component, dc_lookup, ac_lookup) in enumerate(scan):
            blocks: tuple[tuple[int, int], ...]
            if len(scan) == 1:
                row, column = divmod(unit, columns)
                blocks = ((row, column),)
            else:
                mcu_y, mcu_x = divmod(unit, mcus_x)
                blocks = tuple(
                    (mcu_y * component.vertical + by, mcu_x * component.horizontal + bx)
                    for by in range(component.vertical)
                    for bx in range(component.horizontal)
                )
            for row, column in blocks:
                predictors[index] = _decode_block(
                    reader,
                    dc_lookup,
                    ac_lookup,
                    component.store,
                    (row * component.cols + column) * 64,
                    predictors[index],
                )


def read_luma_coefficients(data: bytes) -> JpegCoefficients:
    """Decode ``data``'s quantized luminance DCT coefficients and its luma table.

    Args:
        data: The bytes of a JPEG file.

    Returns:
        A :class:`JpegCoefficients` for the first (luminance) component.

    Raises:
        UnsupportedJpegError: ``data`` is not a JPEG, is progressive,
            arithmetic-coded or lossless, uses more than 8-bit samples, or is
            corrupt enough that the Huffman decoder loses the stream.
    """
    if data[:2] != b"\xff\xd8":
        raise UnsupportedJpegError("not a JPEG file (no SOI marker)")

    quantization: dict[int, np.ndarray] = {}
    dc_tables: dict[int, list[int]] = {}
    ac_tables: dict[int, list[int]] = {}
    components: list[_Component] = []
    height = width = 0
    horizontal_max = vertical_max = 1
    mcus_x = mcus_y = 0
    restart_interval = 0

    position = 2
    size = len(data)
    while position < size - 1:
        if data[position] != 0xFF:
            raise UnsupportedJpegError(f"corrupt JPEG: expected a marker at byte {position}")
        while position < size and data[position] == 0xFF:
            position += 1
        if position >= size:  # pragma: no cover - a stream of fill bytes and nothing else
            break
        marker = data[position]
        position += 1
        if marker == _EOI:
            break
        if marker in _STANDALONE:
            continue

        length = int.from_bytes(data[position : position + 2], "big")
        payload = data[position + 2 : position + length]

        if marker == _DQT:
            offset = 0
            while offset < len(payload):
                precision, table_id = payload[offset] >> 4, payload[offset] & 0xF
                offset += 1
                count = 128 if precision else 64
                raw = payload[offset : offset + count]
                offset += count
                values = np.frombuffer(raw, dtype=">u2" if precision else np.uint8)
                table = np.zeros(64, dtype=np.int32)
                table[list(_NATURAL_ORDER)] = values.astype(np.int32)
                quantization[table_id] = table.reshape(8, 8)
        elif marker == _DHT:
            offset = 0
            while offset < len(payload):
                table_class, table_id = payload[offset] >> 4, payload[offset] & 0xF
                counts = payload[offset + 1 : offset + 17]
                total = sum(counts)
                symbols = payload[offset + 17 : offset + 17 + total]
                offset += 17 + total
                target = ac_tables if table_class else dc_tables
                target[table_id] = _huffman_lookup(counts, symbols)
        elif marker == _DRI:
            restart_interval = int.from_bytes(payload[:2], "big")
        elif marker in _SOF_UNSUPPORTED:
            raise UnsupportedJpegError(
                f"{_SOF_UNSUPPORTED[marker]} JPEGs are not supported; "
                "only baseline and extended-sequential Huffman JPEGs are"
            )
        elif marker in _SOF_BASELINE:
            if payload[0] != 8:
                raise UnsupportedJpegError(
                    f"{payload[0]}-bit JPEG samples are not supported; only 8-bit are"
                )
            height = int.from_bytes(payload[1:3], "big")
            width = int.from_bytes(payload[3:5], "big")
            components = [
                _Component(
                    identifier=payload[6 + index * 3],
                    horizontal=payload[7 + index * 3] >> 4,
                    vertical=payload[7 + index * 3] & 0xF,
                    quant_id=payload[8 + index * 3],
                )
                for index in range(payload[5])
            ]
            horizontal_max = max(component.horizontal for component in components)
            vertical_max = max(component.vertical for component in components)
            mcus_x = -(-width // (8 * horizontal_max))
            mcus_y = -(-height // (8 * vertical_max))
            for component in components:
                component.cols = mcus_x * component.horizontal
                component.rows = mcus_y * component.vertical
                component.store = [0] * (component.rows * component.cols * 64)
        elif marker == _SOS:
            if not components:
                raise UnsupportedJpegError("corrupt JPEG: a scan starts before the frame header")
            by_id = {component.identifier: component for component in components}
            scan = [
                (
                    by_id[payload[1 + index * 2]],
                    dc_tables[payload[2 + index * 2] >> 4],
                    ac_tables[payload[2 + index * 2] & 0xF],
                )
                for index in range(payload[0])
            ]
            segments, position = _entropy_segments(data, position + length)
            _decode_scan(
                segments,
                scan,
                mcus_x,
                mcus_y,
                restart_interval,
                (height, width, horizontal_max, vertical_max),
            )
            continue

        position += length

    if not components:
        raise UnsupportedJpegError("corrupt JPEG: no frame header found")

    luma = components[0]
    blocks = np.asarray(luma.store, dtype=np.int32).reshape(luma.rows, luma.cols, 8, 8)
    spatial = blocks.transpose(0, 2, 1, 3).reshape(luma.rows * 8, luma.cols * 8)
    return JpegCoefficients(
        # Crop the MCU padding away: CAT-Net works on the smallest 8-pixel
        # grid containing the image, not on whole MCUs.
        coefficients=spatial[: -(-height // 8) * 8, : -(-width // 8) * 8].copy(),
        quantization=quantization[luma.quant_id],
        height=height,
        width=width,
    )
