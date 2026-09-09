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
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, Field

from imgforensics.signals.metadata import _jpeg_quality_info
from imgforensics.utils.image_io import load_image

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


def _file_metadata(path: Path) -> tuple[str, int, int, str, int | None]:
    """Read an on-disk image's ``(sha256, width, height, format, jpeg_quality)``.

    Shared by :func:`build_manifest` (scanning a dataset's existing files)
    and :func:`crop_entries` (reading back a file it just wrote or copied),
    so every manifest entry's these five fields are computed by exactly one
    code path regardless of which caller produced the file. Width/height/
    format come from :func:`PIL.Image.open`, which only parses the header
    (pixels are never decoded here); ``jpeg_quality`` is ``None`` for every
    non-JPEG format.
    """
    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    with Image.open(path) as img:
        width, height = img.size
        fmt = img.format or path.suffix.lstrip(".").upper()
        jpeg_quality: int | None = None
        if fmt == "JPEG":
            jpeg_quality = _jpeg_quality_info(img)["jpeg_quality_estimate"]
    return sha256, width, height, fmt, jpeg_quality


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
            sha256, width, height, fmt, jpeg_quality = _file_metadata(file_path)
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


def _combine_notes(existing: str | None, addition: str) -> str:
    return f"{existing}; {addition}" if existing else addition


def split_by_group(
    manifest: Manifest,
    *,
    group_field: str = "generator",
    val_fraction: float = 0.2,
    seed: int = 0,
    holdout: list[str] | None = None,
) -> tuple[Manifest, Manifest]:
    """Split ``manifest`` into two group-disjoint halves (train, val).

    Every entry's group is ``getattr(entry, group_field)`` (``generator`` by
    default). Entries whose group is ``None`` -- typically real images,
    which usually carry no generator -- never determine the group
    assignment below; instead they are shuffled (seeded from ``seed``) and
    split proportionally to ``val_fraction`` at random, so *both* returned
    manifests contain some of them.

    For every other (non-``None``) group, exactly one of the two returned
    manifests gets *all* of that group's entries -- no generator straddles
    the train/val boundary:

    - When ``holdout`` is given, every group named in it goes entirely to
      the second (val) manifest, and every other group entirely to the
      first (train) manifest.
    - Otherwise, groups are deterministically shuffled (seeded from
      ``seed``) and then greedily assigned largest-first: a group is added
      to val (in that shuffled, size-descending order) until val's running
      entry count reaches ``val_fraction`` of the total grouped entries, so
      the val group set ends up holding *about* -- rarely exactly, since
      whole groups cannot be split -- that fraction.

    Every returned entry has ``split`` set to ``"train"`` or ``"val"``
    accordingly (via :meth:`ManifestEntry.model_copy`, so the input
    manifest's entries are untouched), and both manifests' ``meta.notes``
    gets a note describing how the split was made.

    Args:
        manifest: Source manifest.
        group_field: :class:`ManifestEntry` field name to group by.
        val_fraction: Target fraction of non-``None``-group entries (and,
            independently, of ``None``-group entries) assigned to val.
        seed: Seed for the deterministic group shuffle/assignment and the
            random split of ``None``-group entries.
        holdout: When given, exact group names to force entirely into val,
            instead of the size-based greedy assignment.

    Returns:
        ``(train_manifest, val_manifest)``.
    """
    grouped: dict[str, list[ManifestEntry]] = {}
    ungrouped: list[ManifestEntry] = []
    for entry in manifest.entries:
        value = getattr(entry, group_field)
        if value is None:
            ungrouped.append(entry)
        else:
            grouped.setdefault(str(value), []).append(entry)

    total_grouped = sum(len(v) for v in grouped.values())
    if holdout is not None:
        holdout_set = set(holdout)
        val_groups = {name for name in grouped if name in holdout_set}
        val_grouped_count = sum(len(grouped[name]) for name in val_groups)
        note = (
            f"split_by_group: field={group_field!r}, holdout={sorted(holdout_set)} "
            f"-> {len(val_groups)} group(s), {val_grouped_count} of {total_grouped} "
            f"grouped entries into val"
        )
    else:
        group_names = sorted(grouped)
        rng = random.Random(_stratum_seed(seed, (group_field, "group-shuffle")))
        rng.shuffle(group_names)
        ordered = sorted(group_names, key=lambda name: -len(grouped[name]))
        target = round(val_fraction * total_grouped)
        val_groups = set()
        running = 0
        for name in ordered:
            if running >= target:
                break
            val_groups.add(name)
            running += len(grouped[name])
        note = (
            f"split_by_group: field={group_field!r}, val_fraction={val_fraction}, seed={seed} "
            f"-> {len(val_groups)} group(s), {running} of {total_grouped} grouped entries into val "
            "(whole groups only, largest-first greedy assignment)"
        )

    train_entries: list[ManifestEntry] = []
    val_entries: list[ManifestEntry] = []
    for name, entries in grouped.items():
        (val_entries if name in val_groups else train_entries).extend(entries)

    ungrouped_sorted = sorted(ungrouped, key=lambda entry: entry.path)
    indices = list(range(len(ungrouped_sorted)))
    random.Random(_stratum_seed(seed, (group_field, "ungrouped-split"))).shuffle(indices)
    ungrouped_val_count = round(val_fraction * len(ungrouped_sorted))
    val_positions = set(indices[:ungrouped_val_count])
    for position, entry in enumerate(ungrouped_sorted):
        (val_entries if position in val_positions else train_entries).append(entry)

    train_entries = [e.model_copy(update={"split": "train"}) for e in train_entries]
    val_entries = [e.model_copy(update={"split": "val"}) for e in val_entries]

    train_meta = manifest.meta.model_copy(
        update={"notes": _combine_notes(manifest.meta.notes, note)}
    )
    val_meta = manifest.meta.model_copy(update={"notes": _combine_notes(manifest.meta.notes, note)})

    return (
        Manifest(meta=train_meta, entries=train_entries),
        Manifest(meta=val_meta, entries=val_entries),
    )


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


@dataclass
class CropReport:
    """Result of :func:`crop_entries`.

    ``cropped`` counts source entries that were selected and large enough
    to crop -- one per *entry*, regardless of mode. ``tiles_written``
    additionally counts the individual tile files produced in
    ``mode="tiles"`` (always ``0`` in ``mode="center"``, since a center
    crop is not a tile). ``passthrough`` counts selected entries too small
    to crop (copied through unchanged instead), ``copied`` counts entries
    excluded by ``labels`` (also copied through unchanged), and
    ``skipped_unreadable`` counts entries whose source file could not be
    read at all -- one bad file does not abort the run, mirroring
    :func:`build_manifest`. ``bytes_written`` totals the size of files
    actually written to disk during this call; a destination file already
    present from an earlier, interrupted run is left untouched (see the
    idempotence note on :func:`crop_entries`) and does not add to it.
    """

    cropped: int = 0
    tiles_written: int = 0
    passthrough: int = 0
    copied: int = 0
    skipped_unreadable: int = 0
    bytes_written: int = 0


def _copy_through(source_path: Path, dest_path: Path) -> bool:
    """Byte-copy ``source_path`` to ``dest_path`` unless it is already there.

    Returns whether bytes were actually written -- ``False`` when
    ``dest_path`` already existed, which is what keeps :func:`crop_entries`
    idempotent across repeated calls.
    """
    if dest_path.exists():
        return False
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(source_path.read_bytes())
    return True


def _save_crop(image: Image.Image, dest_path: Path) -> bool:
    """Save ``image`` as a lossless PNG at ``dest_path`` unless already there.

    Returns whether the file was actually written, for the same
    idempotence reason as :func:`_copy_through`.
    """
    if dest_path.exists():
        return False
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest_path, format="PNG")
    return True


def _cropped_entry(
    source_entry: ManifestEntry,
    dest_path: Path,
    *,
    new_path: str,
    crop_mode: str | None,
    crop_size: int | None,
    tile_index: int | None,
    original_width: int,
    original_height: int,
) -> ManifestEntry:
    """Build one output-manifest entry from a file :func:`crop_entries` just wrote or copied.

    ``sha256``/``width``/``height``/``format``/``jpeg_quality`` all come
    from re-reading ``dest_path`` via :func:`_file_metadata` -- the same
    helper :func:`build_manifest` uses -- rather than from arithmetic that
    could drift from what actually ended up on disk.
    """
    sha256, width, height, fmt, jpeg_quality = _file_metadata(dest_path)
    return ManifestEntry(
        path=new_path,
        label=source_entry.label,
        source=source_entry.source,
        generator=source_entry.generator,
        split=source_entry.split,
        mask_path=None,
        sha256=sha256,
        width=width,
        height=height,
        format=fmt,
        jpeg_quality=jpeg_quality,
        extra={
            "source_path": source_entry.path,
            "crop_mode": crop_mode,
            "crop_size": crop_size,
            "tile_index": tile_index,
            "original_width": original_width,
            "original_height": original_height,
        },
    )


def crop_entries(
    manifest: Manifest,
    out_dir: str | Path,
    *,
    size: int,
    mode: Literal["center", "tiles"],
    labels: Sequence[Label] | None = None,
    min_side: int | None = None,
    progress: bool = True,
) -> tuple[Manifest, CropReport]:
    """Crop selected entries to native-resolution ``size`` x ``size`` squares.

    This closes a resolution gap between classes without resampling. A
    dataset whose real images run at 1024 px and whose fakes run at 512 px
    lets a detector learn scene scale instead of generation artifacts (see
    ``imgforensics.data.audit``'s resolution-bucket check, which is exactly
    what this is meant to make disappear). Cutting the larger class down to
    native-resolution square crops of the smaller class's size removes that
    gap without resampling -- resizing would itself leave a detectable
    low-pass trace, trading one shortcut for another.

    For every entry whose ``label`` is in ``labels`` (every entry, when
    ``labels`` is ``None``):

    - ``mode="center"``: one ``size`` x ``size`` crop at the image's
      center.
    - ``mode="tiles"``: every non-overlapping ``size`` x ``size`` tile,
      row-major (``_t00``, ``_t01``, ...); a trailing partial row/column
      that does not fill a whole tile is dropped.

    Eligibility is read from the *source* manifest's own recorded
    ``width``/``height`` -- no need to open a file just to decide whether to
    crop it -- against ``size``, or, when given, ``max(size, min_side)``.
    ``min_side`` is a stricter floor than ``size`` itself, useful when a
    crop taken right at the size boundary would sit too close to the
    image's edge to trust (e.g. compression artifacts concentrated there).
    An entry below that floor is copied through unchanged instead and
    counted as ``passthrough`` in the returned report, exactly like an
    entry outside ``labels`` (counted as ``copied``) -- either way the
    output manifest stays a complete dataset, just with some entries
    untouched.

    Pixels are never resampled. EXIF orientation is applied once, via
    :func:`imgforensics.utils.image_io.load_image`, before the crop is
    taken (so an entry's post-orientation size can differ from its source
    manifest's recorded, pre-orientation ``width``/``height`` in the rare
    case of a rotated image; the eligibility check above still uses the
    recorded values, since it must decide whether to open the file at
    all). Every cropped output is saved as a lossless PNG regardless of the
    input's format -- this tool is meant to be run on one label at a time
    (or on classes that already share a format), and a format mismatch it
    introduces itself is exactly what
    :func:`imgforensics.data.audit.audit_manifest` will flag on the result.

    Output tree, under ``out_dir``: a crop is written to ``<source path
    without its extension>[_tNN].png`` (no suffix in ``mode="center"``); a
    passthrough or not-selected entry keeps its original relative path
    unchanged. (Two source entries that differ only in extension, e.g.
    ``a.jpg`` and ``a.png``, would collide at that stripped-extension path
    -- an edge case this function does not guard against.)

    Every output entry keeps its source entry's ``label``, ``source``,
    ``generator`` and ``split``; ``mask_path`` is always ``None`` (this
    tool does not crop masks), and ``meta.notes`` gets a note when any
    source entry had one, since the association would otherwise silently
    go stale. ``extra`` records ``source_path`` (the original relative
    path) and ``original_width``/``original_height``; for entries actually
    subject to this crop job -- selected by ``labels``, whether cropped or
    passed through for being too small -- it also records ``crop_mode`` and
    ``crop_size``, and ``tile_index`` (the tile's position in
    ``mode="tiles"``, else ``None``). An entry excluded by ``labels`` never
    had cropping attempted at all, so ``crop_mode``/``crop_size`` are
    ``None`` for it.

    Idempotent: a destination file that already exists is left untouched
    (not re-decoded, re-cropped, or re-copied), so an interrupted run can
    be resumed by calling this again with the same ``out_dir``.
    Deterministic: entries are processed in source-path order, and a
    source image's tiles are written in row-major order, so the returned
    manifest's entry order (and thus its saved byte content, since
    :meth:`Manifest.save` sorts by path) never depends on filesystem
    iteration order.

    Args:
        manifest: Source manifest; its ``meta.root`` locates every entry's
            source file.
        out_dir: Destination directory for the new image tree.
        size: Crop edge length in pixels.
        mode: ``"center"`` or ``"tiles"``.
        labels: Only these labels are cropped; every other entry is copied
            through unchanged. ``None`` (the default) selects every label.
        min_side: Optional stricter floor than ``size`` for crop
            eligibility (see above).
        progress: When true, print a one-line progress count as entries are
            processed, matching :func:`build_manifest`'s convention.

    Returns:
        A ``(manifest, report)`` tuple. The returned manifest's
        ``meta.root`` is ``out_dir``; ``meta.notes`` gets
        ``"cropped <labels> to <size> (<mode>) from <dataset>"``.

    Raises:
        ValueError: ``size`` is not positive, or ``mode`` is neither
            ``"center"`` nor ``"tiles"``.
    """
    if mode not in ("center", "tiles"):
        raise ValueError(f"mode must be 'center' or 'tiles', got {mode!r}")
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")

    # Imported here, not at module scope: imgforensics.eval (via
    # eval.runner) imports imgforensics.data.manifest, so importing
    # eval.preprocess at module scope would be a circular import.
    from imgforensics.eval.preprocess import center_crop, grid_crops

    out_dir = Path(out_dir).resolve()
    root = Path(manifest.meta.root)
    label_set = set(labels) if labels is not None else None
    threshold = size if min_side is None else max(size, min_side)

    entries_sorted = sorted(manifest.entries, key=lambda entry: entry.path)
    report = CropReport()
    new_entries: list[ManifestEntry] = []
    had_mask = False
    total = len(entries_sorted)

    for index, entry in enumerate(entries_sorted, start=1):
        if progress and (index % 500 == 0 or index == total):
            print(f"[crop] {index}/{total} entries processed")
        if entry.mask_path is not None:
            had_mask = True

        source_path = root / entry.path
        selected = label_set is None or entry.label in label_set

        if not selected:
            dest_path = out_dir / entry.path
            try:
                wrote = _copy_through(source_path, dest_path)
            except OSError:
                report.skipped_unreadable += 1
                continue
            if wrote:
                report.bytes_written += dest_path.stat().st_size
            report.copied += 1
            new_entries.append(
                _cropped_entry(
                    entry,
                    dest_path,
                    new_path=entry.path,
                    crop_mode=None,
                    crop_size=None,
                    tile_index=None,
                    original_width=entry.width,
                    original_height=entry.height,
                )
            )
            continue

        if min(entry.width, entry.height) < threshold:
            dest_path = out_dir / entry.path
            try:
                wrote = _copy_through(source_path, dest_path)
            except OSError:
                report.skipped_unreadable += 1
                continue
            if wrote:
                report.bytes_written += dest_path.stat().st_size
            report.passthrough += 1
            new_entries.append(
                _cropped_entry(
                    entry,
                    dest_path,
                    new_path=entry.path,
                    crop_mode=mode,
                    crop_size=size,
                    tile_index=None,
                    original_width=entry.width,
                    original_height=entry.height,
                )
            )
            continue

        try:
            image = load_image(source_path)
        except Exception:  # noqa: BLE001 - one bad file must not abort the run
            report.skipped_unreadable += 1
            continue
        actual_width, actual_height = image.size

        tiles: list[tuple[int | None, Image.Image]]
        if mode == "center":
            tiles = [(None, center_crop(image, size))]
        else:
            tiles_x = actual_width // size
            tiles_y = actual_height // size
            grid = grid_crops(image, size, max_crops=max(1, tiles_x * tiles_y))
            tiles = list(enumerate(grid))

        stem = PurePosixPath(entry.path).with_suffix("").as_posix()
        report.cropped += 1
        for tile_index, cropped_image in tiles:
            new_path = f"{stem}.png" if tile_index is None else f"{stem}_t{tile_index:02d}.png"
            dest_path = out_dir / new_path
            wrote = _save_crop(cropped_image, dest_path)
            if wrote:
                report.bytes_written += dest_path.stat().st_size
            if tile_index is not None:
                report.tiles_written += 1
            new_entries.append(
                _cropped_entry(
                    entry,
                    dest_path,
                    new_path=new_path,
                    crop_mode=mode,
                    crop_size=size,
                    tile_index=tile_index,
                    original_width=actual_width,
                    original_height=actual_height,
                )
            )

    labels_desc = ",".join(labels) if labels else "all"
    note = f"cropped {labels_desc} to {size} ({mode}) from {manifest.meta.dataset}"
    if had_mask:
        note += "; source manifest had mask(s), not carried over (masks are not cropped)"
    new_meta = manifest.meta.model_copy(
        update={"root": str(out_dir), "notes": _combine_notes(manifest.meta.notes, note)}
    )
    return Manifest(meta=new_meta, entries=new_entries), report
