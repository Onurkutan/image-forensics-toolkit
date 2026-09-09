"""Classical low-level signal analyzers (noise, ELA, JPEG artifacts, and similar).

Importing this package registers every signal detector (currently
``metadata``, ``ela``, ``c2pa``, ``sd_watermark``, ``copy_move``,
``jpeg_ghost`` and ``double_jpeg``) with :mod:`imgforensics.core.registry`
as a side effect.
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

__all__ = [
    "copymove",
    "double_jpeg",
    "ela",
    "jpeg_ghost",
    "metadata",
    "provenance",
    "watermark",
]
