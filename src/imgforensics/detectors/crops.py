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
from typing import TYPE_CHECKING, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel

# Private, but shared deliberately: this is the same reflection-padding the
# evaluation-side crop helpers use, and the two must agree pixel-for-pixel or
# a cached feature would depend on which code path produced the crop.
from imgforensics.eval.preprocess import _pad_to_at_least, center_crop, random_crops

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


def _grid_crops_center_first(rgb: Image.Image, size: int, max_crops: int) -> list[Image.Image]:
    """The ``max_crops`` non-overlapping ``size`` tiles closest to the image center.

    Tiles are enumerated row-major over the padded array, ranked by the
    Euclidean distance from the tile center to the image center, and returned
    nearest-first. Ties break on ``(top, left)``, so the order is fully
    determined by the image dimensions -- never by dict or set iteration.
    """
    array = _pad_to_at_least(np.asarray(rgb.convert("RGB")), size)
    height, width = array.shape[:2]
    center_y, center_x = height / 2.0, width / 2.0

    positions: list[tuple[float, int, int]] = []
    for top in range(0, height - size + 1, size):
        for left in range(0, width - size + 1, size):
            tile_y, tile_x = top + size / 2.0, left + size / 2.0
            distance = (tile_y - center_y) ** 2 + (tile_x - center_x) ** 2
            positions.append((distance, top, left))
    positions.sort()

    return [
        Image.fromarray(array[top : top + size, left : left + size], mode="RGB")
        for _, top, left in positions[:max_crops]
    ]


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
    rgb = image_rgb.convert("RGB")
    size = policy.size
    if not policy.min_side_pad and (rgb.width < size or rgb.height < size):
        raise ValueError(
            f"image is {rgb.width}x{rgb.height}, smaller than the {size}px crop size, "
            "and min_side_pad is disabled (upscaling it would violate the "
            "crop-never-resize rule)"
        )

    if policy.mode == "center":
        return [center_crop(rgb, size)]
    if policy.mode == "grid":
        return _grid_crops_center_first(rgb, size, policy.max_crops)

    material = seed_material if seed_material is not None else np.asarray(rgb).tobytes()
    return random_crops(rgb, size, policy.max_crops, _seed_for(policy, material))


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
