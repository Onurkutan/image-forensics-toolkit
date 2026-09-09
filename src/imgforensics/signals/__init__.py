"""Classical low-level signal analyzers (noise, ELA, JPEG artifacts, and similar).

Importing this package registers every signal detector (currently ``metadata``
and ``ela``) with :mod:`imgforensics.core.registry` as a side effect.
"""

from imgforensics.signals import ela, metadata

__all__ = ["ela", "metadata"]
