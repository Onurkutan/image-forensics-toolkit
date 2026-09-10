"""Pixel-level localizers that highlight manipulated or AI-inpainted regions.

Where :mod:`imgforensics.detectors` answers "how generated does this image
look", this package answers "*where*": every localizer returns a
:class:`~imgforensics.core.types.DetectionResult` whose ``heatmap`` is the
primary output and whose ``score`` is derived from it.

:mod:`imgforensics.localization.weights` is the license gate for the
pretrained weights these need, and is deliberately torch-free -- ``weights
list`` and ``weights fetch`` work on a machine that cannot yet run the model.

Importing this package registers the ``iml_vit`` localizer with
:mod:`imgforensics.core.registry`, but only when the optional ``ml`` extra is
installed, so ``imgforensics analyze`` on a torch-free machine offers exactly
the detectors it did before. :mod:`imgforensics.localization.iml_vit` itself
imports no torch at module scope, so the registration costs nothing at
startup even where the extra *is* installed. This mirrors
:mod:`imgforensics.detectors`' handling of ``dinov2_head``.
"""

from __future__ import annotations

from imgforensics.detectors import is_ml_available
from imgforensics.localization.weights import (
    WEIGHTS,
    FetchWeightsReport,
    WeightSpec,
    fetch_weights,
    get_weight_spec,
    resolve_weights_dir,
    weights_file,
)

if is_ml_available():  # pragma: no cover - the branch taken depends on the install
    # Side effect: registers the "iml_vit" localizer. Imported here rather
    # than at the top so the statement above decides whether it happens.
    from imgforensics.localization import iml_vit  # noqa: F401

__all__ = [
    "WEIGHTS",
    "FetchWeightsReport",
    "WeightSpec",
    "fetch_weights",
    "get_weight_spec",
    "resolve_weights_dir",
    "weights_file",
]
