"""Classical low-level signal analyzers (noise, ELA, JPEG artifacts, and similar).

Importing this package registers every signal detector (currently
``metadata``, ``ela``, ``c2pa`` and ``sd_watermark``) with
:mod:`imgforensics.core.registry` as a side effect.
"""

from imgforensics.signals import ela, metadata, provenance, watermark

__all__ = ["ela", "metadata", "provenance", "watermark"]
