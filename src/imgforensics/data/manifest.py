"""Dataset manifests: build, save, load, and summarize labeled image lists.

A manifest is the reproducibility unit for a dataset slice: a JSON Lines file
of :class:`ManifestEntry` records (one per image, sorted by path) plus a
sidecar ``*.meta.json`` describing the dataset as a whole. Every entry stores
a sha256 of the file's bytes so a manifest can be diffed or re-verified
without re-reading every image, and a ``commercial_ok`` flag at the dataset
level so a downstream training config can filter on it (see
``docs/ROADMAP.md``, sections 3 and 7).
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable, Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, Field

from imgforensics.signals.metadata import _jpeg_quality_info

Label = Literal["real", "fake"]
Split = Literal["train", "val", "test"]

#: Default set of file extensions considered by :func:`build_manifest`.
DEFAULT_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")

# Parent-folder name -> label, used by label_from_parent_folder. Keys are
# lower-cased before lookup so the match is case-insensitive.
_FOLDER_LABELS: dict[str, Label] = {
    "real": "real",
    "fake": "fake",
    "0_real": "real",
    "1_fake": "fake",
    "authentic": "real",
    "tampered": "fake",
    "nature": "real",
    "ai": "fake",
}


class ManifestEntry(BaseModel):
    """A single labeled image within a dataset manifest.

    ``path`` (and ``mask_path``, when present) are POSIX-style (forward
    slashes) and relative to the manifest's ``root`` directory, so a
    manifest built on one machine stays valid after the dataset is moved or
    copied elsewhere.
    """

    path: str
    label: Label
    source: str
    generator: str | None = None
    split: Split | None = None
    mask_path: str | None = None
    sha256: str
    width: int
    height: int
    format: str
    jpeg_quality: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ManifestMeta(BaseModel):
    """Dataset-level metadata attached to a :class:`Manifest`."""

    manifest_version: int = 1
    dataset: str
    root: str
    license: str | None = None
    commercial_ok: bool | None = None
    created: str
    notes: str | None = None


class Manifest(BaseModel):
    """A dataset manifest: its metadata plus every labeled entry."""

    meta: ManifestMeta
    entries: list[ManifestEntry] = Field(default_factory=list)

    @staticmethod
    def _meta_path_for(path: Path) -> Path:
        return path.parent / f"{path.stem}.meta.json"

    def save(self, path: str | Path) -> None:
        """Write the manifest as JSON Lines to ``path``, sorted by entry path.

        Also writes the sidecar metadata file ``<path stem>.meta.json`` next
        to it. Sorting by path makes the output deterministic (stable byte
        content, and thus a stable diff/hash) regardless of the order
        entries were built or discovered in.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sorted_entries = sorted(self.entries, key=lambda entry: entry.path)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for entry in sorted_entries:
                handle.write(entry.model_dump_json())
                handle.write("\n")
        meta_path = self._meta_path_for(path)
        meta_path.write_text(self.meta.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        """Read a manifest previously written by :meth:`save`."""
        path = Path(path)
        meta_path = cls._meta_path_for(path)
        meta = ManifestMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        entries: list[ManifestEntry] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entries.append(ManifestEntry.model_validate_json(line))
        return cls(meta=meta, entries=entries)

    def summary(self) -> dict[str, dict[str, int]]:
        """Return entry counts grouped by label, source, generator, split, and format.

        Entries with no generator/split (``None``) are counted under the key
        ``"none"``.
        """
        counts: dict[str, dict[str, int]] = {
            "label": {},
            "source": {},
            "generator": {},
            "split": {},
            "format": {},
        }
        for entry in self.entries:
            counts["label"][entry.label] = counts["label"].get(entry.label, 0) + 1
            counts["source"][entry.source] = counts["source"].get(entry.source, 0) + 1
            generator_key = entry.generator or "none"
            counts["generator"][generator_key] = counts["generator"].get(generator_key, 0) + 1
            split_key = entry.split or "none"
            counts["split"][split_key] = counts["split"].get(split_key, 0) + 1
            counts["format"][entry.format] = counts["format"].get(entry.format, 0) + 1
        return counts


def label_from_parent_folder(path: Path) -> Label | None:
    """Infer a label from the image's immediate parent folder name.

    Recognizes (case-insensitively) ``real``/``fake``, ``0_real``/``1_fake``,
    ``authentic``/``tampered``, and ``nature``/``ai``. Returns ``None`` when
    the parent folder name does not match any of these.
    """
    return _FOLDER_LABELS.get(path.parent.name.lower())


def _relative_posix(path: Path, root: Path) -> str:
    candidate = path if path.is_absolute() else (root / path)
    return candidate.resolve().relative_to(root.resolve()).as_posix()


def build_manifest(
    root: str | Path,
    *,
    dataset: str,
    label_of: Callable[[Path], Label | None],
    generator_of: Callable[[Path], str | None] | None = None,
    split_of: Callable[[Path], Split | None] | None = None,
    mask_of: Callable[[Path], Path | None] | None = None,
    license: str | None = None,
    commercial_ok: bool | None = None,
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
    progress: bool = True,
) -> tuple[Manifest, list[str]]:
    """Walk ``root`` recursively and build a :class:`Manifest` from its images.

    Files are visited in sorted (POSIX-path) order for determinism. For each
    file, ``label_of`` decides whether (and how) it is included: a ``None``
    return skips the file entirely (it is not counted as an error). Included
    files get a sha256 of their raw bytes, width/height/format read via
    :func:`PIL.Image.open` (lazy -- only the header is parsed, pixels are
    never decoded), and, for JPEGs, a quality estimate from the embedded
    quantization tables (:func:`imgforensics.signals.metadata._jpeg_quality_info`).

    A file that cannot be opened or hashed is recorded in the returned
    ``skipped`` list (as ``"<path>: <error>"``) rather than raising, so one
    corrupt file does not abort a large build.

    Args:
        root: Directory to walk.
        dataset: Value stored as each entry's ``source`` and the manifest's
            ``meta.dataset``.
        label_of: Maps a file path to a label, or ``None`` to skip it.
        generator_of: Optional map to a generator name.
        split_of: Optional map to a train/val/test split.
        mask_of: Optional map to a mask file path (absolute, or relative to
            ``root``); stored as a root-relative POSIX path.
        license: Recorded in ``meta.license``.
        commercial_ok: Recorded in ``meta.commercial_ok``.
        extensions: Case-insensitive file extensions to consider.
        progress: When true, print a one-line progress count as files are
            processed (no extra dependency; plain ``print``, easy to
            suppress in tests with ``progress=False``).

    Returns:
        A ``(manifest, skipped)`` tuple.
    """
    root = Path(root).resolve()
    extensions_lower = {ext.lower() for ext in extensions}
    files = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions_lower),
        key=lambda p: p.as_posix(),
    )

    entries: list[ManifestEntry] = []
    skipped: list[str] = []

    for index, file_path in enumerate(files, start=1):
        if progress and (index % 500 == 0 or index == len(files)):
            print(f"[manifest] {index}/{len(files)} files scanned")

        label = label_of(file_path)
        if label is None:
            continue

        try:
            data = file_path.read_bytes()
            sha256 = hashlib.sha256(data).hexdigest()
            with Image.open(file_path) as img:
                width, height = img.size
                fmt = img.format or file_path.suffix.lstrip(".").upper()
                jpeg_quality: int | None = None
                if fmt == "JPEG":
                    jpeg_quality = _jpeg_quality_info(img)["jpeg_quality_estimate"]
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the build
            skipped.append(f"{file_path}: {exc}")
            continue

        generator = generator_of(file_path) if generator_of else None
        split = split_of(file_path) if split_of else None
        mask_path: str | None = None
        if mask_of is not None:
            mask_result = mask_of(file_path)
            if mask_result is not None:
                mask_path = _relative_posix(mask_result, root)

        entries.append(
            ManifestEntry(
                path=_relative_posix(file_path, root),
                label=label,
                source=dataset,
                generator=generator,
                split=split,
                mask_path=mask_path,
                sha256=sha256,
                width=width,
                height=height,
                format=fmt,
                jpeg_quality=jpeg_quality,
            )
        )

    meta = ManifestMeta(
        dataset=dataset,
        root=str(root),
        license=license,
        commercial_ok=commercial_ok,
        created=date.today().isoformat(),
    )
    return Manifest(meta=meta, entries=entries), skipped


def _stratum_value(entry: ManifestEntry, field_name: str) -> str:
    value = getattr(entry, field_name)
    return "none" if value is None else str(value)


def _stratum_seed(seed: int, key: tuple[str, ...]) -> int:
    """A 64-bit integer seed derived from ``seed`` and a stratum key.

    ``random.Random`` only guarantees reproducible seeding for
    ``None``/``int``/``float``/``str``/``bytes``/``bytearray`` -- seeding it
    with a tuple falls back to Python's (hash-randomized, so
    process-dependent) ``hash()``, which would silently break
    :func:`sample`'s determinism guarantee across runs. Hashing an explicit
    string encoding instead keeps the seed stable everywhere.
    """
    basis = f"{seed}:{'/'.join(key)}".encode()
    digest = hashlib.sha256(basis).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def sample(
    manifest: Manifest,
    n: int,
    *,
    seed: int = 0,
    stratify_by: Sequence[str] = ("label", "generator"),
) -> Manifest:
    """Deterministically draw a proportional stratified sample of ``n`` entries.

    Entries are grouped by the tuple of ``getattr(entry, field)`` for each
    field in ``stratify_by`` (missing/``None`` values group under
    ``"none"``, matching :meth:`Manifest.summary`). ``n`` is allocated
    across strata proportionally to each stratum's share of the manifest,
    using the largest-remainder method so the allocation sums to exactly
    ``min(n, len(manifest.entries))`` -- never more entries than exist, and
    never more than a stratum actually has. Selection within a stratum is
    seeded from ``(seed, stratum key)`` via :class:`random.Random`, so the
    same manifest, ``n``, ``seed`` and ``stratify_by`` always produce the
    same sample.

    The returned manifest's ``meta`` is a copy of ``manifest.meta`` with a
    note appended: ``"sampled <k> of <N> with seed <seed>"``.

    Args:
        manifest: Source manifest.
        n: Target sample size (clamped to ``len(manifest.entries)``).
        seed: Seed for the deterministic per-stratum shuffle.
        stratify_by: :class:`ManifestEntry` field names to stratify by.

    Returns:
        A new :class:`Manifest` with the sampled entries.
    """
    entries = manifest.entries
    total = len(entries)
    target_total = max(0, min(n, total))

    groups: dict[tuple[str, ...], list[ManifestEntry]] = {}
    for entry in entries:
        key = tuple(_stratum_value(entry, field_name) for field_name in stratify_by)
        groups.setdefault(key, []).append(entry)

    keys = sorted(groups.keys())
    counts = {key: len(groups[key]) for key in keys}
    raw_shares = {key: (target_total * counts[key] / total) if total else 0.0 for key in keys}
    allocation = {key: min(counts[key], int(math.floor(raw_shares[key]))) for key in keys}
    remainder = target_total - sum(allocation.values())

    # Largest-remainder method: give the leftover seats to the strata with the
    # biggest fractional share first (ties broken by sorted key, for determinism).
    by_remainder = sorted(
        keys, key=lambda key: (-(raw_shares[key] - math.floor(raw_shares[key])), key)
    )
    index = 0
    while remainder > 0 and by_remainder:
        key = by_remainder[index % len(by_remainder)]
        if allocation[key] < counts[key]:
            allocation[key] += 1
            remainder -= 1
        index += 1

    selected: list[ManifestEntry] = []
    for key in keys:
        take = allocation[key]
        if take <= 0:
            continue
        stratum_entries = sorted(groups[key], key=lambda entry: entry.path)
        indices = list(range(len(stratum_entries)))
        random.Random(_stratum_seed(seed, key)).shuffle(indices)
        selected.extend(stratum_entries[i] for i in indices[:take])

    note = f"sampled {len(selected)} of {total} with seed {seed}"
    combined_notes = f"{manifest.meta.notes}; {note}" if manifest.meta.notes else note
    new_meta = manifest.meta.model_copy(update={"notes": combined_notes})
    return Manifest(meta=new_meta, entries=selected)


def merge(manifests: Sequence[Manifest]) -> Manifest:
    """Concatenate several manifests' entries into one.

    When every input manifest shares the same ``meta.root``, entry paths
    (already root-relative) are kept as-is and the merged manifest keeps
    that root. When roots differ, every entry's path (and ``mask_path``,
    when set) is rewritten to an absolute POSIX path -- ``<that
    manifest's root> / <entry path>`` -- so the merged manifest is still
    unambiguous, and a note recording this is appended to ``meta.notes``.

    ``meta.dataset`` becomes the sorted, ``"+"``-joined set of source
    dataset names. ``meta.license`` is kept only when every input agrees;
    otherwise ``None`` with a note. ``meta.commercial_ok`` is ``True`` only
    when every input is ``True``, ``False`` when any input is ``False``
    (the conservative choice for a downstream commercial-use filter), and
    ``None`` otherwise (e.g. all unknown, or a mix of ``True``/``None``
    with no ``False``).

    Raises:
        ValueError: ``manifests`` is empty.
    """
    if not manifests:
        raise ValueError("merge() requires at least one manifest")

    roots = {m.meta.root for m in manifests}
    common_root = next(iter(roots)) if len(roots) == 1 else None

    notes: list[str] = []
    entries: list[ManifestEntry] = []
    if common_root is not None:
        for m in manifests:
            entries.extend(m.entries)
    else:
        notes.append("merged manifests had different roots; paths stored as absolute")
        for m in manifests:
            root_path = Path(m.meta.root)
            for entry in m.entries:
                abs_path = (root_path / entry.path).as_posix()
                abs_mask = (root_path / entry.mask_path).as_posix() if entry.mask_path else None
                entries.append(entry.model_copy(update={"path": abs_path, "mask_path": abs_mask}))

    dataset_names = sorted({m.meta.dataset for m in manifests})
    licenses = {m.meta.license for m in manifests}
    if len(licenses) == 1:
        license_ = licenses.pop()
    else:
        license_ = None
        notes.append("source manifests disagreed on license; left unset")

    if all(m.meta.commercial_ok is True for m in manifests):
        commercial_ok: bool | None = True
    elif any(m.meta.commercial_ok is False for m in manifests):
        commercial_ok = False
    else:
        commercial_ok = None

    for m in manifests:
        if m.meta.notes:
            notes.append(f"[{m.meta.dataset}] {m.meta.notes}")

    meta = ManifestMeta(
        dataset="+".join(dataset_names),
        root=common_root or "",
        license=license_,
        commercial_ok=commercial_ok,
        created=date.today().isoformat(),
        notes="; ".join(notes) or None,
    )
    return Manifest(meta=meta, entries=entries)
