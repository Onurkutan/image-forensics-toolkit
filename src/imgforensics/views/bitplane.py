"""Bit planes: one bit of the luma image, shown on its own.

An 8-bit image is eight one-bit images stacked. The top bits carry the scene,
which is why bit 7 looks like a hard-thresholded photograph. The bottom bits
carry the parts of the pixel value nothing in the scene decides: sensor noise,
quantization, and whatever the last processing step left behind. That makes
the low planes a blunt but honest structure detector -- in an untouched
photograph they look like static, and anything *else* down there (a visible
edge, a block grid, a flat area, banding) was put there by processing rather
than by the camera.

Naming the usual suspects, since the map itself says nothing: a region pasted
from a lower-bit-depth or heavily compressed source, an area painted or
gradient-filled by an editor, and a steganographic payload written into the
least significant bit all show up as structure in a plane that should be
noise. So does an ordinary screenshot, an ordinary PNG export, and an
ordinary sky in a JPEG -- which is exactly why this is a view.
"""

from __future__ import annotations

import numpy as np

from imgforensics.core.image import ForensicImage
from imgforensics.core.parameters import ParameterSpec
from imgforensics.core.registry import register
from imgforensics.core.types import DetectionResult
from imgforensics.views.base import ViewTool, luma

DEFAULT_PLANE = 0

#: Highest bit index of an 8-bit luma; plane 0 is the least significant bit.
MAX_PLANE = 7


def bit_plane_map(plane: np.ndarray, bit: int = DEFAULT_PLANE) -> np.ndarray:
    """Bit ``bit`` of a luma plane rounded to 8 bits, as a float32 map of 0.0 and 1.0."""
    quantized = np.clip(np.rint(plane), 0, 255).astype(np.uint8)
    return ((quantized >> bit) & 1).astype(np.float32)


@register("bit_planes")
class BitPlanesView(ViewTool):
    """One bit of the 8-bit luma image, as a map of 0.0 and 1.0.

    ``fraction_set`` in the details is the share of pixels where the bit is
    1. A plane carrying only noise sits near 0.5; a plane far from it is
    dominated by the scene (the high planes) or by something that flattened
    the low bits (heavy compression, a paint tool, a gradient fill).
    """

    name = "bit_planes"

    def __init__(self, plane: int = DEFAULT_PLANE) -> None:
        self.plane = plane

    @classmethod
    def parameters(cls) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                name="plane",
                kind="int",
                default=DEFAULT_PLANE,
                minimum=0,
                maximum=MAX_PLANE,
                step=1,
                description=(
                    "Which bit of the luma to show, 0 being the least significant. The low "
                    "planes hold noise and processing traces; the high ones hold the scene."
                ),
            )
        ]

    def predict(self, image: ForensicImage) -> DetectionResult:
        heatmap = bit_plane_map(luma(image.rgb), self.plane)
        return self.view_result(
            heatmap,
            plane=self.plane,
            fraction_set=float(heatmap.mean()),
        )
