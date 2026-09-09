"""Dataset manifests, an external-dataset registry, and a bias audit tool.

- :mod:`imgforensics.data.manifest` -- build/save/load a JSON-Lines manifest
  of labeled images (path, label, source, sha256, resolution, JPEG quality).
- :mod:`imgforensics.data.registry` -- catalogue of external datasets
  (``datasets.yaml``), with license and ``commercial_ok`` metadata.
- :mod:`imgforensics.data.audit` -- compares the real/fake halves of a
  manifest for format/resolution/JPEG-quality bias and duplicate images.

See ``docs/ROADMAP.md``, section 4 (target architecture) and Phase 2, for
how these fit into the rest of the toolkit.
"""

from imgforensics.data.audit import AuditReport, BiasError, audit_manifest
from imgforensics.data.manifest import (
    Manifest,
    ManifestEntry,
    ManifestMeta,
    build_manifest,
    label_from_parent_folder,
)
from imgforensics.data.registry import (
    DatasetInfo,
    get_dataset,
    load_registry,
    registry_table_markdown,
)

__all__ = [
    "AuditReport",
    "BiasError",
    "DatasetInfo",
    "Manifest",
    "ManifestEntry",
    "ManifestMeta",
    "audit_manifest",
    "build_manifest",
    "get_dataset",
    "label_from_parent_folder",
    "load_registry",
    "registry_table_markdown",
]
