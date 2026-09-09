"""Image-level detectors that classify a whole image as real or AI-generated.

The learned detectors are built on a frozen ViT backbone plus a small head
(``docs/ROADMAP.md``, Phase 3). Their heavy dependencies -- ``torch``, ``timm``
-- live in the optional ``ml`` extra, so this package is written to import
cleanly *without* them: :mod:`~imgforensics.detectors.crops` (the crop policy)
and :mod:`~imgforensics.detectors.features` (the feature cache) are pure
numpy/PIL at import time, and every ``torch`` import sits inside the function
that needs it. Callers that need to know whether the extra is present ask
:func:`is_ml_available` rather than catching :class:`ImportError`.
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
from imgforensics.detectors.crops import CropPolicy, crops_for, to_array, to_tensor
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


__all__ = [
    "BACKBONES",
    "BackboneSpec",
    "CacheKey",
    "CropPolicy",
    "FeatureCache",
    "FeatureExtractor",
    "crops_for",
    "extract_to_cache",
    "get_backbone",
    "is_ml_available",
    "normalization_for",
    "resolve_device",
    "to_array",
    "to_tensor",
]
