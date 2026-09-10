"""What every view shares: the no-verdict contract and three map primitives.

A view answers "what does this image look like under X", never "is this image
fake". :class:`ViewTool` fixes that in one place -- score 0.5, label
``"uncertain"``, ``details["kind"] = "view"`` -- so a view module is left with
nothing but its map.

The three helpers below are the primitives those maps are built from. They are
here rather than in each view because the views are deliberately consistent
with each other: all of them read the same Rec. 601 luma, all of them handle
borders by reflection, and the two continuous ones are scaled by their own
99th percentile so that "bright" means the same thing in both -- the
convention :func:`imgforensics.signals.ela.ela_map` already uses, kept
identical here so a reader who has learned to read one map can read the
others.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from PIL import Image

from imgforensics.core.base import BaseDetector
from imgforensics.core.types import DetectionResult, ToolKind, label_from_score

#: The score every view reports. A view has no verdict to give, and 0.5 is
#: this project's spelling of "no opinion" (the same value a detector uses
#: when it abstains), so a view lands on ``"uncertain"`` and contributes
#: nothing to a fuser that is ever pointed at one.
VIEW_SCORE = 0.5

#: Rec. 601 luma coefficients -- the weights Pillow's "L" conversion uses.
#: Spelled out rather than delegated to ``convert("L")`` so the result stays
#: float, without the rounding to 8-bit steps that would quantize a residual.
_LUMA_WEIGHTS = (0.299, 0.587, 0.114)

#: A 99th percentile at or below this counts as "nothing to see": scaling by
#: it would divide by ~0 and turn sensor rounding into a full-range map.
_FLAT_EPSILON = 1e-6


class ViewTool(BaseDetector):
    """Base class for a tool that shows a map and claims nothing about it.

    Subclasses implement :meth:`~imgforensics.core.base.BaseDetector.predict`
    and build their result with :meth:`view_result`, which fills in the parts
    that are the same for every view. They stay ordinary registered tools --
    ``analyze --detector <name>``, ``--save-heatmaps`` and ``--report-dir``
    treat them like any other -- but they are excluded where a score is the
    point (the default ``analyze`` run, ``benchmark``).
    """

    kind: ClassVar[ToolKind] = "view"

    def view_result(self, heatmap: np.ndarray, **details: Any) -> DetectionResult:
        """A view's result: the map, the no-verdict score, and ``details``.

        Args:
            heatmap: The view's map, float32 and in [0, 1], shaped like the
                image.
            details: Numbers describing the map, merged after
                ``{"kind": "view"}`` -- which marks the result as carrying no
                verdict for any caller reading the JSON rather than the
                registry.
        """
        return DetectionResult(
            detector=self.name,
            score=VIEW_SCORE,
            label=label_from_score(VIEW_SCORE),
            heatmap=heatmap,
            details={"kind": "view", **details},
        )


def luma(rgb: Image.Image) -> np.ndarray:
    """The Rec. 601 luma of an image, as a float32 array in [0, 255].

    Written as three elementwise multiplications rather than a dot product
    over the channel axis, which is the same arithmetic but not the same
    floating point: a dot product may accumulate in a different order for
    different parts of the array, and two identical pixels then come out one
    unit in the last place apart. That is invisible in a photograph and very
    visible in a view -- a gradient map of a *uniform* image would find that
    rounding, scale it by its own 99th percentile, and paint the difference
    over the whole frame.

    Args:
        rgb: Any Pillow image; converted to RGB first.

    Returns:
        A float32 ``(height, width)`` array.
    """
    array = np.asarray(rgb.convert("RGB"), dtype=np.float32)
    red, green, blue = _LUMA_WEIGHTS
    return red * array[..., 0] + green * array[..., 1] + blue * array[..., 2]


def neighbourhood(plane: np.ndarray, radius: int) -> np.ndarray:
    """Every shift of ``plane`` within ``radius``, stacked, reflection-padded.

    This is how the views do their local arithmetic: a filter over a
    ``(2 * radius + 1)`` square becomes one vectorized operation over the
    stacking axis, which keeps them to numpy and Pillow (no SciPy, no
    per-pixel Python) and keeps the result the image's own shape. Reflection
    at the borders means a view never invents a dark frame around the image;
    numpy handles a side shorter than the padding, so a 1x1 image works.

    The stack costs ``(2 * radius + 1) ** 2`` copies of the plane, which is
    what bounds the window sizes the views offer.

    Args:
        plane: A 2-D array.
        radius: Half-width of the window; 0 returns the plane alone.

    Returns:
        A ``((2 * radius + 1) ** 2, height, width)`` array, in row-major
        order of the offsets -- so index ``dy * side + dx`` holds the
        neighbour at ``(dy - radius, dx - radius)``, which is the order a
        flattened convolution kernel is written in.
    """
    height, width = plane.shape
    side = 2 * radius + 1
    padded = np.pad(plane, radius, mode="reflect")
    return np.stack(
        [padded[dy : dy + height, dx : dx + width] for dy in range(side) for dx in range(side)]
    )


def normalize_by_p99(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale ``values`` so its 99th percentile becomes 1.0, clipped to [0, 1].

    The top percentile is given away on purpose: a handful of extreme pixels
    would otherwise set the scale and press everything else into the bottom of
    the range. A flat plane, whose 99th percentile is ~0, yields an all-zero
    map instead of a division by zero.

    Returns:
        The scaled float32 map, and the 99th percentile it was divided by --
        which is the number worth reporting, since the scaled map's own is
        1.0 by construction.
    """
    p99 = float(np.percentile(values, 99))
    scaled = np.zeros_like(values, dtype=np.float32) if p99 <= _FLAT_EPSILON else values / p99
    return np.clip(scaled, 0.0, 1.0).astype(np.float32), p99
