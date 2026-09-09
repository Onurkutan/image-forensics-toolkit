"""Shared crop and training-time augmentation helpers for the learned detectors.

Two independent things live here (``docs/ROADMAP.md``, section 3, "crop, never
resize; augment always"):

- :func:`center_crop`, :func:`random_crops`, :func:`grid_crops` -- native-
  resolution crops used both for training input and for tiled inference,
  never a full-image resize (resizing destroys the high-frequency generator
  artifacts these detectors key on).
- :class:`AugmentationConfig` / :func:`augment` -- the *training-time*
  counterpart of :mod:`imgforensics.eval.robustness`. Both wrap the same
  handful of operations (JPEG/WEBP re-encoding, resize, blur, noise), but
  serve opposite purposes: the robustness suite is a fixed, versioned
  evaluation protocol (same perturbation, every run, so results are
  comparable across models and time), while this module draws randomized
  parameters every call so a training loop sees varied examples. Because of
  that, **this module must never be used at evaluation time** -- doing so
  would let the eval protocol drift between runs and make robustness numbers
  incomparable.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageFilter
from pydantic import BaseModel, Field

_DEFAULT_PROBABILITY = 0.5


def _pad_to_at_least(array: np.ndarray, size: int) -> np.ndarray:
    """Reflection-pad ``array`` (HxWx3) so both spatial dims are >= ``size``.

    Uses ``cv2.copyMakeBorder`` with ``BORDER_REFLECT_101`` rather than
    :func:`numpy.pad`'s ``"reflect"`` mode, which refuses to pad by more than
    ``dim - 1`` pixels in one call -- a real constraint here, since a crop
    ``size`` can be larger than a small source image's side.
    """
    height, width = array.shape[:2]
    pad_h = max(0, size - height)
    pad_w = max(0, size - width)
    if pad_h == 0 and pad_w == 0:
        return array
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    return cv2.copyMakeBorder(array, top, bottom, left, right, cv2.BORDER_REFLECT_101)


def center_crop(rgb: Image.Image, size: int) -> Image.Image:
    """Crop the ``size`` x ``size`` region at the center of ``rgb``.

    Reflection-pads first when ``rgb`` is smaller than ``size`` in either
    dimension, so the result is always exactly ``size`` x ``size``.
    """
    array = _pad_to_at_least(np.asarray(rgb.convert("RGB")), size)
    height, width = array.shape[:2]
    top = (height - size) // 2
    left = (width - size) // 2
    return Image.fromarray(array[top : top + size, left : left + size], mode="RGB")


def random_crops(rgb: Image.Image, size: int, n: int, seed: int) -> list[Image.Image]:
    """``n`` random, independently-placed ``size`` x ``size`` crops of ``rgb``.

    Reflection-pads first when ``rgb`` is smaller than ``size`` (see
    :func:`center_crop`). Crop positions are drawn from a
    :class:`numpy.random.Generator` seeded with ``seed``, so the same
    ``(rgb, size, n, seed)`` always yields the same crop positions.
    """
    array = _pad_to_at_least(np.asarray(rgb.convert("RGB")), size)
    height, width = array.shape[:2]
    rng = np.random.default_rng(seed)
    crops: list[Image.Image] = []
    for _ in range(n):
        top = int(rng.integers(0, height - size + 1))
        left = int(rng.integers(0, width - size + 1))
        crops.append(Image.fromarray(array[top : top + size, left : left + size], mode="RGB"))
    return crops


def grid_crops(rgb: Image.Image, size: int, max_crops: int) -> list[Image.Image]:
    """Non-overlapping ``size`` x ``size`` tiles of ``rgb``, row-major, capped at ``max_crops``.

    Reflection-pads first when ``rgb`` is smaller than ``size`` (see
    :func:`center_crop`). Tiles are taken left-to-right, top-to-bottom,
    stopping as soon as ``max_crops`` tiles have been collected; any
    trailing partial row/column that would not fill a full tile is dropped.
    """
    array = _pad_to_at_least(np.asarray(rgb.convert("RGB")), size)
    height, width = array.shape[:2]
    crops: list[Image.Image] = []
    for top in range(0, height - size + 1, size):
        for left in range(0, width - size + 1, size):
            crops.append(Image.fromarray(array[top : top + size, left : left + size], mode="RGB"))
            if len(crops) >= max_crops:
                return crops
    return crops


class AugmentationConfig(BaseModel):
    """Training-time augmentation config: which perturbations, and their parameter ranges.

    Each of ``jpeg_quality``, ``webp_quality``, ``gaussian_blur_sigma``,
    ``downscale_upscale`` and ``noise_sigma`` is a ``(low, high)`` range that
    a parameter is drawn uniformly from when that augmentation fires;
    ``None`` disables it. ``cutout`` is ``{"size": int, "count": int}`` (edge
    length and number of squares to blank out), or ``None`` to disable.
    Every enabled augmentation is applied independently with probability
    ``p`` on each call to :func:`augment` (so, with several enabled, more
    than one can fire on the same image), unless ``probabilities`` overrides
    that rate for a given field.
    """

    jpeg_quality: tuple[int, int] | None = None
    webp_quality: tuple[int, int] | None = None
    gaussian_blur_sigma: tuple[float, float] | None = None
    downscale_upscale: tuple[float, float] | None = None
    noise_sigma: tuple[float, float] | None = None
    cutout: dict[str, int] | None = None
    p: float = _DEFAULT_PROBABILITY
    probabilities: dict[str, float] = Field(default_factory=dict)
    """Per-augmentation firing rates, keyed by this model's own field names.

    A field not listed here fires at ``p``. Real capture pipelines do not
    apply their degradations at one shared rate -- almost everything on the
    web has been JPEG-compressed at least once, while heavy blur is rare --
    so the packaged ``configs/augment_default.yaml`` sets a different rate
    per operation. An unknown key is ignored rather than rejected, so a
    config written for a future field still loads.
    """

    @classmethod
    def from_yaml(cls, path: str | Path) -> AugmentationConfig:
        """Load a config from a YAML mapping of this model's fields."""
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)

    def probability_for(self, field_name: str) -> float:
        """Firing rate of one augmentation: its ``probabilities`` entry, else ``p``."""
        return float(self.probabilities.get(field_name, self.p))

    def fingerprint(self) -> str:
        """A short, stable hash of this config, used as part of a feature cache key.

        Same construction as
        :meth:`imgforensics.detectors.crops.CropPolicy.fingerprint` -- the
        canonical JSON dump with sorted keys -- so features extracted under
        two different augmentation policies never collide in the cache.
        """
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _apply_downscale_upscale(
    rgb: Image.Image, bounds: tuple[float, float], rng: np.random.Generator
) -> Image.Image:
    scale = float(rng.uniform(*bounds))
    original_size = rgb.size
    width, height = original_size
    down_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    down = rgb.resize(down_size, Image.Resampling.BILINEAR)
    return down.resize(original_size, Image.Resampling.BILINEAR)


def _apply_blur(
    rgb: Image.Image, bounds: tuple[float, float], rng: np.random.Generator
) -> Image.Image:
    sigma = float(rng.uniform(*bounds))
    return rgb.filter(ImageFilter.GaussianBlur(radius=sigma))


def _apply_noise(
    rgb: Image.Image, bounds: tuple[float, float], rng: np.random.Generator
) -> Image.Image:
    sigma = float(rng.uniform(*bounds))
    array = np.asarray(rgb, dtype=np.float32)
    noisy = np.clip(array + rng.normal(0.0, sigma, array.shape), 0.0, 255.0)
    return Image.fromarray(noisy.astype(np.uint8), mode="RGB")


def _apply_jpeg(rgb: Image.Image, bounds: tuple[int, int], rng: np.random.Generator) -> Image.Image:
    quality = int(rng.integers(bounds[0], bounds[1] + 1))
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def _apply_webp(rgb: Image.Image, bounds: tuple[int, int], rng: np.random.Generator) -> Image.Image:
    quality = int(rng.integers(bounds[0], bounds[1] + 1))
    buffer = io.BytesIO()
    rgb.save(buffer, format="WEBP", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def _apply_cutout(
    rgb: Image.Image, cutout: dict[str, int], rng: np.random.Generator
) -> Image.Image:
    size = int(cutout.get("size", 32))
    count = int(cutout.get("count", 1))
    array = np.asarray(rgb, dtype=np.uint8).copy()
    height, width = array.shape[:2]
    for _ in range(count):
        half = size // 2
        cy = int(rng.integers(0, height))
        cx = int(rng.integers(0, width))
        top, bottom = max(0, cy - half), min(height, cy + half)
        left, right = max(0, cx - half), min(width, cx + half)
        array[top:bottom, left:right] = 0
    return Image.fromarray(array, mode="RGB")


def augment(rgb: Image.Image, config: AugmentationConfig, rng: np.random.Generator) -> Image.Image:
    """Apply ``config``'s enabled augmentations to ``rgb``, each at its own probability.

    An augmentation fires with probability
    :meth:`AugmentationConfig.probability_for` -- its ``probabilities``
    entry when it has one, otherwise the shared ``config.p``.

    Draws all randomness (which augmentations fire, and their sampled
    parameters) from ``rng``, so a caller that wants reproducible training
    batches passes a seeded :class:`numpy.random.Generator`. Order of
    application: downscale/upscale, blur, noise, JPEG, WEBP, cutout. One
    ``rng.random()`` draw is spent per *enabled* augmentation, in that
    order, so the stream a given ``rng`` produces depends on which
    augmentations are enabled but not on which of them fired.

    This is the training-time counterpart of
    :mod:`imgforensics.eval.robustness` -- see the module docstring for why
    it must not be used to build evaluation data.
    """
    result = rgb.convert("RGB")
    if config.downscale_upscale is not None and rng.random() < config.probability_for(
        "downscale_upscale"
    ):
        result = _apply_downscale_upscale(result, config.downscale_upscale, rng)
    if config.gaussian_blur_sigma is not None and rng.random() < config.probability_for(
        "gaussian_blur_sigma"
    ):
        result = _apply_blur(result, config.gaussian_blur_sigma, rng)
    if config.noise_sigma is not None and rng.random() < config.probability_for("noise_sigma"):
        result = _apply_noise(result, config.noise_sigma, rng)
    if config.jpeg_quality is not None and rng.random() < config.probability_for("jpeg_quality"):
        result = _apply_jpeg(result, config.jpeg_quality, rng)
    if config.webp_quality is not None and rng.random() < config.probability_for("webp_quality"):
        result = _apply_webp(result, config.webp_quality, rng)
    if config.cutout is not None and rng.random() < config.probability_for("cutout"):
        result = _apply_cutout(result, config.cutout, rng)
    return result
