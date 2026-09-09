"""Classical low-level signal analyzers (noise, ELA, JPEG artifacts, and similar).

Importing this package registers every signal detector (currently
``metadata``, ``ela``, ``c2pa``, ``sd_watermark``, ``copy_move``,
``jpeg_ghost`` and ``double_jpeg``) with :mod:`imgforensics.core.registry`
as a side effect.

:data:`SIGNAL_NAMES` names exactly those detectors. It exists because the
registry also holds detectors that are *not* signals -- the learned
``dinov2_head`` registers itself when the optional ``ml`` extra is installed
-- so "every registered detector" and "every signal" stopped being the same
set. Anything that means the second (the ``signals_mean`` baseline, the
benchmark runner's ``--all-signals``) reads this tuple.

This package imports pure numpy/PIL/opencv only: no ``torch``, directly or
transitively, so a signals-only run never pays for the ``ml`` extra.
"""

from imgforensics.signals import (
    copymove,
    double_jpeg,
    ela,
    jpeg_ghost,
    metadata,
    provenance,
    watermark,
)

#: Registry names of the classical signal detectors, sorted.
SIGNAL_NAMES: tuple[str, ...] = (
    "c2pa",
    "copy_move",
    "double_jpeg",
    "ela",
    "jpeg_ghost",
    "metadata",
    "sd_watermark",
)

__all__ = [
    "SIGNAL_NAMES",
    "copymove",
    "double_jpeg",
    "ela",
    "jpeg_ghost",
    "metadata",
    "provenance",
    "watermark",
]
