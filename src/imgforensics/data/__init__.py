"""Dataset manifests, an external-dataset registry, and a bias audit tool.

- :mod:`imgforensics.data.manifest` -- build/save/load a JSON-Lines manifest
  of labeled images (path, label, source, sha256, resolution, JPEG quality),
  plus :func:`~imgforensics.data.manifest.sample` (stratified subsampling),
  :func:`~imgforensics.data.manifest.merge` (combine manifests), and
  :func:`~imgforensics.data.manifest.crop_entries` (native-resolution
  crops, to equalize resolution between classes without resampling).
- :mod:`imgforensics.data.registry` -- catalogue of external datasets
  (``datasets.yaml``), with license and ``commercial_ok`` metadata.
- :mod:`imgforensics.data.audit` -- compares the real/fake halves of a
  manifest for format/resolution/JPEG-quality bias and duplicate images.
- :mod:`imgforensics.data.acquire` -- license-gated dataset downloads
  (``acquire.yaml``) for the registry's datasets.
- :mod:`imgforensics.data.layouts` -- turns a downloaded dataset folder
  into a manifest via a declarative folder-shape description
  (``layouts.yaml``).

See ``docs/ROADMAP.md``, section 4 (target architecture) and Phase 2, for
how these fit into the rest of the toolkit.
"""

from imgforensics.data.acquire import (
    AcquireRecipe,
    AcquireStep,
    ChecksumMismatchError,
    FetchReport,
    LicenseNotAcceptedError,
    fetch,
    get_recipe,
    load_acquire_recipes,
)
from imgforensics.data.audit import AuditReport, BiasError, audit_manifest
from imgforensics.data.layouts import (
    Layout,
    MaterializeReport,
    get_layout,
    load_layouts,
    materialize_parquet,
    prepare,
)
from imgforensics.data.manifest import (
    CropReport,
    Manifest,
    ManifestEntry,
    ManifestMeta,
    build_manifest,
    crop_entries,
    label_from_parent_folder,
    merge,
    sample,
    split_by_group,
)
from imgforensics.data.registry import (
    DatasetInfo,
    get_dataset,
    load_registry,
    registry_table_markdown,
)

__all__ = [
    "AcquireRecipe",
    "AcquireStep",
    "AuditReport",
    "BiasError",
    "ChecksumMismatchError",
    "CropReport",
    "DatasetInfo",
    "FetchReport",
    "Layout",
    "LicenseNotAcceptedError",
    "Manifest",
    "ManifestEntry",
    "ManifestMeta",
    "MaterializeReport",
    "audit_manifest",
    "build_manifest",
    "crop_entries",
    "fetch",
    "get_dataset",
    "get_layout",
    "get_recipe",
    "label_from_parent_folder",
    "load_acquire_recipes",
    "load_layouts",
    "load_registry",
    "materialize_parquet",
    "merge",
    "prepare",
    "registry_table_markdown",
    "sample",
    "split_by_group",
]
