# Vendored from ShieldMnt/invisible-watermark, file imwatermark/maxDct.py
# (the ``method='dwtDct'`` implementation, referenced in that project's
# imwatermark/watermark.py as ``EmbedMaxDct``), commit history at
# https://github.com/ShieldMnt/invisible-watermark .
#
# MIT License
#
# Copyright (c) 2021 ShieldMnt
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ---
#
# Adapted for imgforensics: only imports and typing were changed (added type
# hints, replaced mutable-default-argument parameters with ``None``/tuple
# defaults to satisfy this repo's linter, and the unused ``pprint``/``copy``
# imports were dropped). The DWT/DCT algorithm itself (encode_frame,
# decode_frame, diffuse_dct_matrix, infer_dct_matrix) is unchanged, including
# the upstream project's own quirk of swapping the (h1, v1) sub-bands when
# calling ``pywt.idwt2`` in :meth:`EmbedMaxDct.encode` -- this only affects
# the cosmetic reconstruction of the non-watermarked wavelet sub-bands, not
# the watermark bits themselves (which live in the ``ca1`` low-frequency band
# encoded/decoded identically on both sides), so round-trip decoding is
# unaffected.
"""The ``dwtDct`` invisible-watermark scheme used by Stable Diffusion pipelines:
a watermark payload is embedded into (and recovered from) the low-frequency
DWT sub-band of the Y/U channels via a per-block max-DCT-coefficient
quantization trick. See :mod:`imgforensics.signals.watermark` for the known
Stable Diffusion payloads and the bit-agreement decision rule built on top of
this codec.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import pywt

_DEFAULT_SCALES: tuple[int, int, int] = (0, 36, 36)


class EmbedMaxDct:
    """DWT + per-block max-DCT-coefficient watermark codec (the ``dwtDct`` method).

    Args:
        watermarks: The watermark bits to embed (0/1 ints), only used by
            :meth:`encode`. Ignored by :meth:`decode`.
        wm_len: Number of watermark bits (the payload length); ``decode``
            returns exactly this many bits.
        scales: Per-YUV-channel quantization step (Y, U, V). A channel with
            scale ``<= 0`` is left untouched; the reference encoders leave Y
            untouched and use the same scale for U and V.
        block: Side length of the square blocks the DWT sub-band is split
            into for per-block encoding/decoding.
    """

    def __init__(
        self,
        watermarks: Sequence[int] | None = None,
        wm_len: int = 8,
        scales: tuple[int, int, int] = _DEFAULT_SCALES,
        block: int = 4,
    ) -> None:
        self._watermarks: list[int] = list(watermarks) if watermarks is not None else []
        self._wm_len = wm_len
        self._scales = scales
        self._block = block

    def encode(self, bgr: np.ndarray) -> np.ndarray:
        """Embed the watermark into a uint8 BGR image, returning the encoded copy."""
        row, col, _channels = bgr.shape

        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        for channel in range(2):
            if self._scales[channel] <= 0:
                continue

            ca1, (h1, v1, d1) = pywt.dwt2(yuv[: row // 4 * 4, : col // 4 * 4, channel], "haar")
            self._encode_frame(ca1, self._scales[channel])

            yuv[: row // 4 * 4, : col // 4 * 4, channel] = pywt.idwt2((ca1, (v1, h1, d1)), "haar")

        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def decode(self, bgr: np.ndarray) -> np.ndarray:
        """Recover ``wm_len`` watermark bits from a (possibly unwatermarked) BGR image."""
        row, col, _channels = bgr.shape

        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        scores: list[list[float]] = [[] for _ in range(self._wm_len)]
        for channel in range(2):
            if self._scales[channel] <= 0:
                continue

            ca1, (_h1, _v1, _d1) = pywt.dwt2(yuv[: row // 4 * 4, : col // 4 * 4, channel], "haar")
            scores = self._decode_frame(ca1, self._scales[channel], scores)

        avg_scores = [float(np.mean(s)) for s in scores]
        return np.array(avg_scores) * 255 > 127

    def _decode_frame(
        self, frame: np.ndarray, scale: int, scores: list[list[float]]
    ) -> list[list[float]]:
        row, col = frame.shape
        num = 0

        for i in range(row // self._block):
            for j in range(col // self._block):
                block = frame[
                    i * self._block : i * self._block + self._block,
                    j * self._block : j * self._block + self._block,
                ]

                score = self._infer_dct_matrix(block, scale)
                wm_bit = num % self._wm_len
                scores[wm_bit].append(score)
                num += 1

        return scores

    def _infer_dct_matrix(self, block: np.ndarray, scale: int) -> int:
        pos = int(np.argmax(np.abs(block.flatten()[1:]))) + 1
        i, j = pos // self._block, pos % self._block

        val = block[i][j]
        if val < 0:
            val = abs(val)

        return 1 if (val % scale) > 0.5 * scale else 0

    def _diffuse_dct_matrix(self, block: np.ndarray, wm_bit: int, scale: int) -> np.ndarray:
        pos = int(np.argmax(np.abs(block.flatten()[1:]))) + 1
        i, j = pos // self._block, pos % self._block
        val = block[i][j]
        if val >= 0.0:
            block[i][j] = (val // scale + 0.25 + 0.5 * wm_bit) * scale
        else:
            val = abs(val)
            block[i][j] = -1.0 * (val // scale + 0.25 + 0.5 * wm_bit) * scale
        return block

    def _encode_frame(self, frame: np.ndarray, scale: int) -> None:
        """Encode ``self._watermarks`` into ``frame`` (a DWT sub-band), in place.

        ``frame`` is split into ``self._block`` x ``self._block`` blocks; the
        i-th block (in row-major order) carries ``watermarks[i % wm_len]``.
        """
        row, col = frame.shape
        num = 0
        for i in range(row // self._block):
            for j in range(col // self._block):
                block = frame[
                    i * self._block : i * self._block + self._block,
                    j * self._block : j * self._block + self._block,
                ]
                wm_bit = self._watermarks[num % self._wm_len]

                diffused_block = self._diffuse_dct_matrix(block, wm_bit, scale)
                frame[
                    i * self._block : i * self._block + self._block,
                    j * self._block : j * self._block + self._block,
                ] = diffused_block

                num += 1
