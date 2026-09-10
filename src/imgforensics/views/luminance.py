"""Luminance gradient: how sharply brightness changes, pixel by pixel.

Edges are where most editing shows. A pasted region carries its own edge
profile -- softer if it was feathered, harder if it was cut out -- and a
region that was painted, blurred or upscaled loses the fine gradient texture
that surrounds it. This view draws the Sobel gradient magnitude of the luma
channel and nothing else, so those differences are visible without any claim
about which of them is suspicious: gradient magnitude is high on every real
edge in every photograph too.

The ``radius`` parameter blurs the luma before the Sobel kernel runs, which
trades the pixel-level noise floor for the shape of larger structures -- the
same reason a reader squints at a print.
"""

from __future__ import annotations

import numpy as np

from imgforensics.core.image import ForensicImage
from imgforensics.core.parameters import ParameterSpec
from imgforensics.core.registry import register
from imgforensics.core.types import DetectionResult
from imgforensics.views.base import ViewTool, luma, neighbourhood, normalize_by_p99

#: Horizontal Sobel kernel; the vertical one is its transpose. Kept in
#: float64 -- see :func:`gradient_magnitude`.
_SOBEL_X = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float64)

DEFAULT_RADIUS = 1

#: Largest pre-blur offered. Radius 3 already averages a 5x5 box, past which
#: the gradient describes the blur more than the image.
_MAX_RADIUS = 3


def _box_blur(plane: np.ndarray, radius: int) -> np.ndarray:
    """``plane`` averaged over a ``(2 * (radius - 1) + 1)`` square; radius 1 is the plane."""
    if radius <= 1:
        return plane
    return neighbourhood(plane, radius - 1).mean(axis=0)


def gradient_magnitude(plane: np.ndarray) -> np.ndarray:
    """The Sobel gradient magnitude of a 2-D plane, at the plane's own shape.

    The two kernel sums are accumulated in float64. A Sobel kernel sums to
    zero, so on a region of constant brightness its two halves must cancel
    exactly -- and in float32 they do not: the running total passes through
    three times a pixel value, which is not always representable, and the
    leftover unit in the last place survives. Since the map is then scaled by
    its own 99th percentile, that leftover is all it would take to paint a
    uniform image bright from edge to edge. In float64 every partial sum of a
    float32 pixel times a small integer is exact, so a flat region cancels to
    a true zero whatever order the sum is taken in.
    """
    stack = neighbourhood(plane, 1).astype(np.float64)
    horizontal = np.tensordot(_SOBEL_X.reshape(-1), stack, axes=(0, 0))
    vertical = np.tensordot(_SOBEL_X.T.reshape(-1), stack, axes=(0, 0))
    return np.sqrt(horizontal * horizontal + vertical * vertical)


def luminance_gradient_map(plane: np.ndarray, radius: int = DEFAULT_RADIUS) -> np.ndarray:
    """The view's map: Sobel magnitude of a box-blurred luma, scaled by its own p99."""
    return normalize_by_p99(gradient_magnitude(_box_blur(plane, radius)))[0]


@register("luminance_gradient")
class LuminanceGradientView(ViewTool):
    """Sobel gradient magnitude of the luma channel, scaled by its 99th percentile.

    Bright means brightness changes fast there. Read it for edges that do not
    belong to the scene: a boundary that is uniformly sharp or uniformly soft
    where the rest of the image is neither, or a region whose fine gradient
    texture stops at a straight line.
    """

    name = "luminance_gradient"

    def __init__(self, radius: int = DEFAULT_RADIUS) -> None:
        self.radius = radius

    @classmethod
    def parameters(cls) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                name="radius",
                kind="int",
                default=DEFAULT_RADIUS,
                minimum=1,
                maximum=_MAX_RADIUS,
                step=1,
                description=(
                    "Box blur applied to the luma before the Sobel kernel. 1 leaves the "
                    "luma untouched; raising it suppresses pixel noise so larger structures "
                    "carry the map."
                ),
            )
        ]

    def predict(self, image: ForensicImage) -> DetectionResult:
        plane = _box_blur(luma(image.rgb), self.radius)
        heatmap, p99 = normalize_by_p99(gradient_magnitude(plane))
        return self.view_result(
            heatmap,
            radius=self.radius,
            p99=p99,
            mean=float(heatmap.mean()),
        )
