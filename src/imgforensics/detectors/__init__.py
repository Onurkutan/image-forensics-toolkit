"""Image-level detectors that classify a whole image as real or AI-generated.

The learned detectors are built on a frozen ViT backbone plus a small head
(``docs/ROADMAP.md``, Phase 3). Their heavy dependencies -- ``torch``, ``timm``
-- live in the optional ``ml`` extra, so this package is written to import
cleanly *without* them: :mod:`~imgforensics.detectors.crops` (the crop policy)
and :mod:`~imgforensics.detectors.features` (the feature cache) are pure
numpy/PIL at import time, and every ``torch`` import sits inside the function
that needs it. Callers that need to know whether the extra is present ask
:func:`is_ml_available` rather than catching :class:`ImportError`.

Importing this package registers the learned detector ``dinov2_head`` with
:mod:`imgforensics.core.registry` -- but only when the ``ml`` extra is
present, so ``imgforensics analyze`` on a torch-free machine offers exactly
the detectors it did before. :mod:`imgforensics.detectors.learned` itself
imports no torch at module scope, so the registration costs nothing at
startup even where the extra *is* installed.
"""

from __future__ import annotations

from importlib.util import find_spec

from imgforensics.detectors.backbones import (
    BACKBONES,
    BackboneSpec,
    get_backbone,
    normalization_for,
    resolve_device,
)
from imgforensics.detectors.crops import (
    CropBox,
    CropPolicy,
    crop_boxes,
    crops_for,
    to_array,
    to_tensor,
)
from imgforensics.detectors.features import (
    CacheKey,
    FeatureCache,
    FeatureExtractor,
    extract_to_cache,
)

#: Modules the optional ``ml`` extra installs; all of them must be importable.
_ML_MODULES = ("torch", "timm")


def is_ml_available() -> bool:
    """Whether the optional ``ml`` extra (torch, timm) is installed.

    Checks for the modules' import *specs* rather than importing them, so
    asking the question costs no time and pulls no CUDA context into the
    process.
    """
    try:
        return all(find_spec(name) is not None for name in _ML_MODULES)
    except (ImportError, ValueError):  # pragma: no cover - broken/partial install
        return False


if is_ml_available():  # pragma: no cover - the branch taken depends on the install
    # Side effect: registers the "dinov2_head" learned detector. Imported here
    # rather than at the top so the statement above decides whether it happens.
    from imgforensics.detectors import learned  # noqa: F401

__all__ = [
    "BACKBONES",
    "BackboneSpec",
    "CacheKey",
    "CropBox",
    "CropPolicy",
    "FeatureCache",
    "FeatureExtractor",
    "crop_boxes",
    "crops_for",
    "extract_to_cache",
    "get_backbone",
    "is_ml_available",
    "normalization_for",
    "resolve_device",
    "to_array",
    "to_tensor",
]
