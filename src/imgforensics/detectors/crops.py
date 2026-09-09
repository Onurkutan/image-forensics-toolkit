"""Crop policy for the learned detectors: native-resolution crops, never a resize.

Resizing an image low-passes exactly the high-frequency generator artifacts a
learned detector keys on (``docs/ROADMAP.md``, section 3, "crop, never resize;
augment always"), so every image that reaches a backbone here is *cropped* to
the backbone's input size at its native resolution. Images smaller than that
size are reflection-padded rather than upscaled.

The three modes trade coverage against cost:

- ``center`` -- one crop at the image center; the cheapest, and the only mode
  that is a function of the image alone (no seed, no ordering).
- ``grid`` -- the ``max_crops`` non-overlapping tiles whose centers are
  closest to the image center, nearest first. Deterministic, and biased
  toward the middle of the frame where the subject usually is.
- ``random`` -- ``max_crops`` independently placed crops, seeded from the
  image content itself (see :func:`crops_for`), so the same image always
  yields the same crops while two different images yield different ones.

:func:`crops_for` cuts the crops; :func:`crop_boxes` returns where those same
crops sit in the source image, which is what turns a per-crop score into a
heatmap. Both read their placements from one private helper, so a box can
never disagree with the crop it describes.

:func:`to_array` produces the normalized ``(N, 3, H, W)`` batch a backbone
expects; :func:`to_tensor` is the same thing wrapped in a
:class:`torch.Tensor`. The split exists so the crop policy, including its
normalization, stays importable and testable without ``torch`` installed
(only :func:`to_tensor` needs the optional ``ml`` extra).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel

# Private, but shared deliberately: this is the same reflection-padding the
# evaluation-side crop helpers use, and the two must agree pixel-for-pixel or
# a cached feature would depend on which code path produced the crop.
from imgforensics.eval.preprocess import _pad_to_at_least

if TYPE_CHECKING:  # pragma: no cover - import-time typing only, never at runtime
    import torch

CropMode = Literal["center", "grid", "random"]

_MEAN_STD_LENGTH = 3


class CropPolicy(BaseModel):
    """How an image is turned into a fixed-size batch of native-resolution crops.

    Attributes:
        size: Edge length of every crop, in pixels. Must match the
            backbone's ``input_size``.
        mode: ``"center"``, ``"grid"`` or ``"random"`` (see the module
            docstring).
        max_crops: Upper bound on the number of crops. ``"center"`` always
            returns exactly one crop regardless of this value; ``"grid"``
            returns fewer when the image holds fewer whole tiles.
        seed: Mixed into the per-image seed of ``"random"`` mode; ignored by
            the other two modes.
        min_side_pad: Reflection-pad an image smaller than ``size`` up to
            ``size`` instead of failing. When ``False``, such an image raises
            :class:`ValueError` -- there is no correct crop of it, and
            silently upscaling would violate the crop-never-resize rule.
    """

    size: int = 224
    mode: CropMode = "grid"
    max_crops: int = 4
    seed: int = 0
    min_side_pad: bool = True

    def fingerprint(self) -> str:
        """A short, stable hash of this policy, used as part of a cache key.

        Derived from the model's JSON dump with sorted keys, so it depends
        only on the field *values* -- adding a field with a default changes
        it (features cached under the old policy are then recomputed, which
        is the safe direction), but reordering the class does not.
        """
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _seed_for(policy: CropPolicy, seed_material: bytes) -> int:
    """Derive ``random`` mode's 64-bit crop seed from image content plus the policy seed.

    Mirrors :func:`imgforensics.eval.robustness._seed_for`: hash the material
    together with a discriminator (here the policy's own ``seed``) and take
    the leading 8 bytes. The same image and policy therefore always place
    crops identically, two different images place them differently, and
    bumping ``policy.seed`` reshuffles every image at once.
    """
    digest = hashlib.sha256(seed_material + str(policy.seed).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


@dataclass(frozen=True)
class CropBox:
    """Where one crop sits in the *source* image, in pixels.

    ``height``/``width`` are normally ``policy.size``, but shrink when the
    crop overhangs a reflection-padded edge: a box is reported in the
    un-padded image's coordinates and clipped to it, so it can index the
    source array (and therefore a heatmap of the source's shape) directly. A
    crop that lies entirely in the padding has zero area.
    """

    top: int
    left: int
    height: int
    width: int

    @property
    def is_empty(self) -> bool:
        """Whether the box covers no source pixel at all (entirely in the padding)."""
        return self.height <= 0 or self.width <= 0


def _grid_positions(height: int, width: int, size: int, max_crops: int) -> list[tuple[int, int]]:
    """Top-left corners of the ``max_crops`` non-overlapping tiles closest to the center.

    Tiles are enumerated row-major over the padded array, ranked by the
    Euclidean distance from the tile center to the image center, and returned
    nearest-first. Ties break on ``(top, left)``, so the order is fully
    determined by the image dimensions -- never by dict or set iteration.
    """
    center_y, center_x = height / 2.0, width / 2.0

    ranked: list[tuple[float, int, int]] = []
    for top in range(0, height - size + 1, size):
        for left in range(0, width - size + 1, size):
            tile_y, tile_x = top + size / 2.0, left + size / 2.0
            distance = (tile_y - center_y) ** 2 + (tile_x - center_x) ** 2
            ranked.append((distance, top, left))
    ranked.sort()

    return [(top, left) for _, top, left in ranked[:max_crops]]


def _positions_for(
    padded_shape: tuple[int, int], policy: CropPolicy, seed_material: bytes
) -> list[tuple[int, int]]:
    """Top-left corners of every crop, in the *padded* array's coordinates.

    The single source of truth for crop placement: :func:`crops_for` cuts at
    these positions and :func:`crop_boxes` reports them, so the two can never
    describe different rectangles. The ``center`` and ``random`` formulas
    match :func:`imgforensics.eval.preprocess.center_crop` and
    :func:`~imgforensics.eval.preprocess.random_crops` exactly (same padding,
    same generator, same draw order: ``top`` then ``left`` per crop), which
    ``tests/test_detectors_crops.py`` pins pixel-for-pixel.
    """
    height, width = padded_shape
    size = policy.size

    if policy.mode == "center":
        return [((height - size) // 2, (width - size) // 2)]
    if policy.mode == "grid":
        return _grid_positions(height, width, size, policy.max_crops)

    rng = np.random.default_rng(_seed_for(policy, seed_material))
    return [
        (int(rng.integers(0, height - size + 1)), int(rng.integers(0, width - size + 1)))
        for _ in range(policy.max_crops)
    ]


def _placement(
    image_rgb: Image.Image, policy: CropPolicy, seed_material: bytes | None
) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int]]:
    """Padded RGB array, crop positions in it, and the ``(top, left)`` padding offsets."""
    rgb = image_rgb.convert("RGB")
    size = policy.size
    if not policy.min_side_pad and (rgb.width < size or rgb.height < size):
        raise ValueError(
            f"image is {rgb.width}x{rgb.height}, smaller than the {size}px crop size, "
            "and min_side_pad is disabled (upscaling it would violate the "
            "crop-never-resize rule)"
        )

    source = np.asarray(rgb)
    array = _pad_to_at_least(source, size)
    material = seed_material if seed_material is not None else source.tobytes()
    offsets = (
        max(0, size - source.shape[0]) // 2,
        max(0, size - source.shape[1]) // 2,
    )
    return array, _positions_for(array.shape[:2], policy, material), offsets


def crops_for(
    image_rgb: Image.Image,
    policy: CropPolicy,
    *,
    seed_material: bytes | None = None,
) -> list[Image.Image]:
    """Cut ``image_rgb`` into crops according to ``policy``.

    Args:
        image_rgb: Source image; converted to RGB, never resized.
        policy: The crop policy to apply.
        seed_material: Bytes the ``"random"`` mode seeds itself from,
            normally the image's encoded bytes
            (:attr:`~imgforensics.core.image.ForensicImage.raw`). Defaults to
            the decoded RGB pixel bytes, which is always available and still
            content-derived. Ignored by the other modes.

    Returns:
        A list of ``policy.size`` x ``policy.size`` RGB images, in a
        deterministic order.

    Raises:
        ValueError: ``policy.min_side_pad`` is false and the image is
            smaller than ``policy.size`` in either dimension.
    """
    array, positions, _ = _placement(image_rgb, policy, seed_material)
    size = policy.size
    return [
        Image.fromarray(array[top : top + size, left : left + size], mode="RGB")
        for top, left in positions
    ]


def crop_boxes(
    image_rgb: Image.Image,
    policy: CropPolicy,
    *,
    seed_material: bytes | None = None,
) -> list[CropBox]:
    """Where :func:`crops_for` took its crops from, in ``image_rgb``'s own coordinates.

    Same arguments, same order, same length as :func:`crops_for`: box ``i``
    is the region crop ``i`` was cut from, clipped to the un-padded image
    (see :class:`CropBox`). This is what lets a per-crop score be painted
    back onto an image-shaped heatmap.
    """
    source_height, source_width = image_rgb.height, image_rgb.width
    _, positions, (pad_top, pad_left) = _placement(image_rgb, policy, seed_material)
    size = policy.size

    boxes: list[CropBox] = []
    for top, left in positions:
        clipped_top = min(max(top - pad_top, 0), source_height)
        clipped_left = min(max(left - pad_left, 0), source_width)
        bottom = min(max(top - pad_top + size, 0), source_height)
        right = min(max(left - pad_left + size, 0), source_width)
        boxes.append(
            CropBox(
                top=clipped_top,
                left=clipped_left,
                height=bottom - clipped_top,
                width=right - clipped_left,
            )
        )
    return boxes


def to_array(
    crops: Sequence[Image.Image],
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    """Stack ``crops`` into a normalized ``(N, 3, H, W)`` float32 array.

    Pixels are scaled to ``[0, 1]`` and then normalized channel-wise as
    ``(x - mean) / std`` with the backbone's own pretrained statistics (see
    :func:`imgforensics.detectors.backbones.normalization_for`).

    Raises:
        ValueError: ``crops`` is empty, or ``mean``/``std`` is not a
            three-element sequence.
    """
    if not crops:
        raise ValueError("to_array() requires at least one crop")
    if len(mean) != _MEAN_STD_LENGTH or len(std) != _MEAN_STD_LENGTH:
        raise ValueError(f"mean and std must have 3 elements, got {len(mean)} and {len(std)}")

    stacked = np.stack([np.asarray(crop.convert("RGB"), dtype=np.float32) for crop in crops])
    stacked /= 255.0
    stacked -= np.asarray(mean, dtype=np.float32)
    stacked /= np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(stacked.transpose(0, 3, 1, 2))


def to_tensor(
    crops: Sequence[Image.Image],
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """Normalized ``(N, 3, H, W)`` float32 :class:`torch.Tensor` of ``crops``.

    Thin wrapper over :func:`to_array`; ``torch`` is imported here rather
    than at module scope so the crop policy stays usable without the
    optional ``ml`` extra.
    """
    import torch

    return torch.from_numpy(to_array(crops, mean, std))
