"""Deterministic robustness suite: JPEG/WEBP re-encoding, resize, crop, noise, and
a "social" re-share pipeline, applied to a :class:`~imgforensics.core.image.ForensicImage`.

A robustness table (``docs/ROADMAP.md``, section 3, "honest evaluation") is only
meaningful if the perturbations it is built from are reproducible: running the
same suite over the same image twice must yield byte-identical results. The
only perturbation here with an actual random component is ``gaussian_noise``;
its seed is derived from ``sha256(image bytes + perturbation name)`` (see
:func:`_seed_for`), so the same image and level always draw the same noise,
two different images draw different noise, and two different levels applied
to the same image draw independent noise.

Every non-``clean`` perturbation returns a new :class:`ForensicImage` built
from freshly encoded bytes (so ``.raw`` and ``.format`` stay populated):
``jpeg``/``webp``/``social`` re-encode losslessly-decoded pixels with Pillow's
default settings and no ``exif=`` argument, which -- verified against this
project's Pillow version -- never carries EXIF over from the source image, so
no extra metadata-stripping step is needed. ``resize``, ``resize_roundtrip``,
``center_crop`` and ``gaussian_noise`` encode to PNG (lossless), so the
pixel-domain transform itself is the only thing that changes the bytes.
"""

from __future__ import annotations

import hashlib
import io
from importlib import resources
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from PIL import Image
from pydantic import BaseModel, Field

from imgforensics.core.image import ForensicImage
from imgforensics.utils.image_io import to_numpy

PerturbationKind = Literal[
    "clean",
    "jpeg",
    "webp",
    "resize",
    "resize_roundtrip",
    "center_crop",
    "gaussian_noise",
    "social",
]

_DEFAULT_SOCIAL_MAX_SIDE = 1080
_DEFAULT_SOCIAL_QUALITY = 80


class Perturbation(BaseModel):
    """A single named robustness-suite level.

    ``params`` holds the kind-specific arguments (see the module docstring
    and :meth:`RobustnessSuite.apply`), e.g. ``{"quality": 85}`` for
    ``kind="jpeg"`` or ``{"scale": 0.5}`` for ``kind="resize"``.
    """

    name: str
    kind: PerturbationKind
    params: dict[str, float | int | str] = Field(default_factory=dict)


def _seed_for(image: ForensicImage, perturbation: Perturbation) -> int:
    """Deterministic 64-bit seed derived from the image content and the level name.

    Uses ``image.raw`` (the original encoded bytes) when available, falling
    back to the decoded RGB pixel bytes for a :class:`ForensicImage` built
    from a bare PIL image. Including the perturbation's name means every
    level draws independent noise even on the same image.
    """
    basis = image.raw if image.raw is not None else to_numpy(image.rgb).tobytes()
    digest = hashlib.sha256(basis + perturbation.name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def _encode(rgb: Image.Image, fmt: str, **save_kwargs: object) -> bytes:
    buffer = io.BytesIO()
    rgb.convert("RGB").save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()


def _resized(rgb: Image.Image, scale: float) -> Image.Image:
    width, height = rgb.size
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return rgb.resize(new_size, Image.Resampling.BILINEAR)


def _center_cropped(rgb: Image.Image, fraction: float) -> Image.Image:
    width, height = rgb.size
    crop_w = max(1, round(width * fraction))
    crop_h = max(1, round(height * fraction))
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    return rgb.crop((left, top, left + crop_w, top + crop_h))


def _with_gaussian_noise(rgb: Image.Image, sigma: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = np.asarray(rgb.convert("RGB"), dtype=np.float32)
    noisy = np.clip(array + rng.normal(0.0, sigma, array.shape), 0.0, 255.0)
    return Image.fromarray(noisy.astype(np.uint8), mode="RGB")


class RobustnessSuite(BaseModel):
    """An ordered list of :class:`Perturbation` levels plus a config version."""

    levels: list[Perturbation] = Field(default_factory=list)
    version: int = 1

    @classmethod
    def _from_yaml_text(cls, text: str) -> RobustnessSuite:
        raw = yaml.safe_load(text) or {}
        levels = [Perturbation.model_validate(item) for item in raw.get("levels", [])]
        return cls(levels=levels, version=int(raw.get("version", 1)))

    @classmethod
    def from_yaml(cls, path: str | Path) -> RobustnessSuite:
        """Load a suite from a YAML file with a top-level ``version`` and ``levels`` list."""
        return cls._from_yaml_text(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def default(cls) -> RobustnessSuite:
        """Load the packaged default suite (``eval/robustness_default.yaml``)."""
        text = (
            resources.files("imgforensics.eval")
            .joinpath("robustness_default.yaml")
            .read_text(encoding="utf-8")
        )
        return cls._from_yaml_text(text)

    def apply(self, perturbation: Perturbation, image: ForensicImage) -> ForensicImage:
        """Apply one perturbation to ``image``, returning a new :class:`ForensicImage`.

        See the module docstring for the byte-identical determinism guarantee
        and which levels round-trip through JPEG/WEBP/PNG.

        Raises:
            ValueError: for an unrecognized ``perturbation.kind`` (should not
                happen given :class:`Perturbation`'s ``Literal`` validation,
                but keeps this function exhaustive for direct callers).
        """
        kind = perturbation.kind
        params = perturbation.params

        if kind == "clean":
            return image

        if kind == "jpeg":
            quality = int(params["quality"])
            data = _encode(image.rgb, "JPEG", quality=quality)
            return ForensicImage.from_bytes(data, path=image.path)

        if kind == "webp":
            quality = int(params["quality"])
            data = _encode(image.rgb, "WEBP", quality=quality)
            return ForensicImage.from_bytes(data, path=image.path)

        if kind == "resize":
            scale = float(params["scale"])
            resized = _resized(image.rgb, scale)
            data = _encode(resized, "PNG")
            return ForensicImage.from_bytes(data, path=image.path)

        if kind == "resize_roundtrip":
            scale = float(params["scale"])
            original_size = image.rgb.size
            down = _resized(image.rgb, scale)
            back_up = down.resize(original_size, Image.Resampling.BILINEAR)
            data = _encode(back_up, "PNG")
            return ForensicImage.from_bytes(data, path=image.path)

        if kind == "center_crop":
            fraction = float(params["fraction"])
            cropped = _center_cropped(image.rgb, fraction)
            data = _encode(cropped, "PNG")
            return ForensicImage.from_bytes(data, path=image.path)

        if kind == "gaussian_noise":
            sigma = float(params["sigma"])
            seed = _seed_for(image, perturbation)
            noisy = _with_gaussian_noise(image.rgb, sigma, seed)
            data = _encode(noisy, "PNG")
            return ForensicImage.from_bytes(data, path=image.path)

        if kind == "social":
            max_side = int(params.get("max_side", _DEFAULT_SOCIAL_MAX_SIDE))
            quality = int(params.get("quality", _DEFAULT_SOCIAL_QUALITY))
            width, height = image.rgb.size
            longer = max(width, height)
            resized = _resized(image.rgb, max_side / longer) if longer > max_side else image.rgb
            data = _encode(resized, "JPEG", quality=quality)
            return ForensicImage.from_bytes(data, path=image.path)

        raise ValueError(f"Unknown perturbation kind: {kind!r}")
