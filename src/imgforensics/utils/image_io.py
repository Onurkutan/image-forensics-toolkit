"""Helpers for loading images and computing lightweight metadata."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_image(source: str | Path | bytes | Image.Image) -> Image.Image:
    """Load an image from a path, raw bytes, or an existing PIL image as RGB.

    EXIF orientation is applied via :func:`PIL.ImageOps.exif_transpose` so the
    returned image is upright regardless of camera orientation metadata.
    """
    if isinstance(source, Image.Image):
        image = source
    elif isinstance(source, bytes):
        image = Image.open(io.BytesIO(source))
    else:
        image = Image.open(Path(source))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def to_numpy(image: Image.Image) -> np.ndarray:
    """Convert an RGB PIL image to a uint8 HxWx3 numpy array."""
    return np.array(image.convert("RGB"), dtype=np.uint8)


def image_hash(image: Image.Image) -> str:
    """Return the sha256 hex digest of the image's raw RGB pixel bytes."""
    array = to_numpy(image)
    return hashlib.sha256(array.tobytes()).hexdigest()
