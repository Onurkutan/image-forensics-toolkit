"""Layout adapters: turn a downloaded dataset folder into a manifest.

A dataset acquired with :func:`imgforensics.data.acquire.fetch` lands on
disk in whatever folder shape its own distribution uses -- per-generator
subfolders, ``Au``/``Tp``/split trees, ``0_real``/``1_fake`` label
folders, and so on. A :class:`Layout` (from the packaged ``layouts.yaml``)
describes that shape declaratively -- which globs are real vs fake, how to
read off a generator name, a split, and a mask path -- so
:func:`prepare` can call
:func:`imgforensics.data.manifest.build_manifest` with the right
``label_of``/``generator_of``/``split_of``/``mask_of`` callables without a
bespoke script per dataset.

Glob patterns here are POSIX-style, relative to the dataset root, and
support ``**`` as "zero or more path segments" (unlike :mod:`fnmatch`'s
plain ``*``, which already crosses ``/`` on its own -- see
:func:`_compile_glob` for the exact translation).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from imgforensics.data.audit import AuditReport, audit_manifest
from imgforensics.data.manifest import (
    Label,
    Manifest,
    Split,
    build_manifest,
    label_from_parent_folder,
)
from imgforensics.data.registry import get_dataset

GeneratorFrom = Literal["parent", "grandparent", "regex", "none"]
SplitFrom = Literal["parent", "top", "regex", "none"]

_SPLIT_VALUES = ("train", "val", "test")


class Layout(BaseModel):
    """Declarative description of one dataset's on-disk folder shape.

    ``real_globs``/``fake_globs`` select which files count as real/fake (a
    file matching neither, or matching ``exclude_globs``, is skipped
    entirely -- the same "return None to skip" contract as
    :func:`~imgforensics.data.manifest.build_manifest`'s ``label_of``).

    ``generator_from``/``split_from`` pick how to read off a generator
    name / split from a matched file's relative path:

    - ``"parent"``: the file's immediate parent folder name.
    - ``"grandparent"`` (generator only): the parent's parent folder name.
    - ``"top"`` (split only): the first path segment.
    - ``"regex"``: apply ``generator_regex``/``split_regex`` (a pattern with
      a named group ``gen`` or ``split`` respectively) to the file's
      root-relative POSIX path.
    - ``"none"``: always ``None``.

    A value derived this way is looked up in ``split_map`` (e.g.
    ``{"training": "train", "validation": "val"}``); if it is not a key
    there but is already one of ``"train"``/``"val"``/``"test"``, it is
    used as-is, otherwise the split is ``None``.

    ``mask_template``, when set, is a format string with ``{stem}``
    (filename without extension), ``{name}`` (filename with extension) and
    ``{parent}`` (the file's root-relative parent directory, POSIX-style,
    or ``"."`` at the root) placeholders, resolved relative to the dataset
    root -- e.g. ``"masks/{stem}.png"`` or, to reach a sibling of the
    image's own folder, ``"{parent}/../gt/{stem}_gt.png"``. A template that
    resolves to a nonexistent file yields ``mask_path=None`` rather than an
    error.
    """

    dataset: str
    real_globs: list[str] = Field(default_factory=list)
    fake_globs: list[str] = Field(default_factory=list)
    generator_from: GeneratorFrom = "none"
    generator_regex: str | None = None
    split_from: SplitFrom = "none"
    split_regex: str | None = None
    split_map: dict[str, str] = Field(default_factory=dict)
    mask_template: str | None = None
    exclude_globs: list[str] = Field(default_factory=list)
    notes: str | None = None


def load_layouts() -> list[Layout]:
    """Load and validate every layout from the packaged ``layouts.yaml``."""
    text = resources.files("imgforensics.data").joinpath("layouts.yaml").read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    return [Layout.model_validate(item) for item in raw.get("layouts", [])]


def get_layout(dataset: str) -> Layout:
    """Return the layout for ``dataset`` (exact, case-sensitive match).

    Raises:
        KeyError: if no layout is registered for ``dataset``, listing the
            available names in the error message.
    """
    for layout in load_layouts():
        if layout.dataset == dataset:
            return layout
    known = ", ".join(sorted(layout.dataset for layout in load_layouts()))
    raise KeyError(f"No layout registered for {dataset!r}. Available: {known or '<none>'}")


def _compile_glob(pattern: str) -> re.Pattern[str]:
    """Translate a POSIX relative glob (``*``, ``?``, ``**``) into an anchored regex.

    ``**/`` becomes an optional "zero or more directories" prefix (so
    ``"**/0_real/**"`` matches both ``"0_real/x.jpg"`` and
    ``"train/0_real/x.jpg"``), a bare ``**`` becomes ``.*``, ``*`` matches
    within one path segment (``[^/]*``), and ``?`` matches one character
    within a segment (``[^/]``). Everything else is a literal.
    """
    placeholder = "\0"
    tmp = pattern.replace("**/", placeholder + "GS" + placeholder)
    tmp = tmp.replace("**", placeholder + "GG" + placeholder)
    tmp = tmp.replace("*", placeholder + "ST" + placeholder)
    tmp = tmp.replace("?", placeholder + "QM" + placeholder)
    escaped = re.escape(tmp)
    escaped = escaped.replace(re.escape(placeholder + "GS" + placeholder), "(?:.*/)?")
    escaped = escaped.replace(re.escape(placeholder + "GG" + placeholder), ".*")
    escaped = escaped.replace(re.escape(placeholder + "ST" + placeholder), "[^/]*")
    escaped = escaped.replace(re.escape(placeholder + "QM" + placeholder), "[^/]")
    return re.compile(f"(?s:{escaped})\\Z")


def _compile_all(patterns: list[str]) -> list[re.Pattern[str]]:
    return [_compile_glob(p) for p in patterns]


def _matches(rel_posix: str, compiled: list[re.Pattern[str]]) -> bool:
    return any(pattern.match(rel_posix) for pattern in compiled)


def _relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _generator_of(rel_posix: str, layout: Layout) -> str | None:
    if layout.generator_from == "none":
        return None
    parts = rel_posix.split("/")
    if layout.generator_from == "parent":
        return parts[-2] if len(parts) >= 2 else None
    if layout.generator_from == "grandparent":
        return parts[-3] if len(parts) >= 3 else None
    if layout.generator_from == "regex" and layout.generator_regex:
        match = re.search(layout.generator_regex, rel_posix)
        if match and "gen" in match.groupdict():
            return match.group("gen")
        return None
    return None


def _normalize_split(raw: str | None, layout: Layout) -> Split | None:
    if raw is None:
        return None
    mapped = layout.split_map.get(raw, raw)
    return mapped if mapped in _SPLIT_VALUES else None  # type: ignore[return-value]


def _split_of(rel_posix: str, layout: Layout) -> Split | None:
    if layout.split_from == "none":
        return None
    parts = rel_posix.split("/")
    if layout.split_from == "top":
        return _normalize_split(parts[0] if parts else None, layout)
    if layout.split_from == "parent":
        return _normalize_split(parts[-2] if len(parts) >= 2 else None, layout)
    if layout.split_from == "regex" and layout.split_regex:
        match = re.search(layout.split_regex, rel_posix)
        raw = match.group("split") if match and "split" in match.groupdict() else None
        return _normalize_split(raw, layout)
    return None


def _mask_of(file_path: Path, root: Path, layout: Layout) -> Path | None:
    if not layout.mask_template:
        return None
    rel = file_path.resolve().relative_to(root.resolve())
    stem = rel.stem
    name = rel.name
    parent = rel.parent.as_posix()
    candidate_str = layout.mask_template.format(stem=stem, name=name, parent=parent)
    candidate = (root / candidate_str).resolve()
    return candidate if candidate.is_file() else None


def _callables_from_layout(
    layout: Layout, root: Path
) -> tuple[
    Callable[[Path], Label | None],
    Callable[[Path], str | None],
    Callable[[Path], Split | None],
    Callable[[Path], Path | None],
]:
    real_re = _compile_all(layout.real_globs)
    fake_re = _compile_all(layout.fake_globs)
    exclude_re = _compile_all(layout.exclude_globs)

    def label_of(path: Path) -> Label | None:
        rel = _relative_posix(path, root)
        if _matches(rel, exclude_re):
            return None
        if _matches(rel, real_re):
            return "real"
        if _matches(rel, fake_re):
            return "fake"
        return None

    def generator_of(path: Path) -> str | None:
        return _generator_of(_relative_posix(path, root), layout)

    def split_of(path: Path) -> Split | None:
        return _split_of(_relative_posix(path, root), layout)

    def mask_of(path: Path) -> Path | None:
        return _mask_of(path, root, layout)

    return label_of, generator_of, split_of, mask_of


def prepare(
    dataset: str,
    src_root: str | Path,
    out_path: str | Path,
    *,
    license: str | None = None,  # noqa: A002 -- matches the registry field name
    commercial_ok: bool | None = None,
) -> tuple[Manifest, list[str], AuditReport]:
    """Build, save, and audit a manifest for ``dataset`` rooted at ``src_root``.

    Looks up a :class:`Layout` for ``dataset`` in the packaged
    ``layouts.yaml``; when none is registered, falls back to
    :func:`~imgforensics.data.manifest.label_from_parent_folder` (a
    generic ``real``/``fake``-style folder tree) and prints a note saying
    so, rather than failing.

    ``license``/``commercial_ok`` default to the
    :mod:`imgforensics.data.registry` entry for ``dataset`` when it exists
    and the argument is left ``None``; when ``dataset`` is not in the
    registry either, they stay ``None`` unless passed explicitly.

    The resulting manifest is saved to ``out_path`` before this returns.

    Returns:
        ``(manifest, skipped, audit_report)`` -- ``skipped`` is the list of
        unreadable files from ``build_manifest``, and ``audit_report`` is
        always computed non-strict (:func:`~imgforensics.data.audit.audit_manifest`
        with ``strict=False``); callers that want a hard failure on bias
        problems check ``audit_report.ok`` themselves (see the CLI's
        ``--strict-audit``).
    """
    resolved_license, resolved_commercial_ok = license, commercial_ok
    try:
        info = get_dataset(dataset)
    except KeyError:
        pass
    else:
        if resolved_license is None:
            resolved_license = info.license
        if resolved_commercial_ok is None:
            resolved_commercial_ok = info.commercial_ok

    root = Path(src_root).resolve()
    try:
        layout = get_layout(dataset)
    except KeyError:
        print(
            f"[prepare] no layout registered for {dataset!r}; "
            "falling back to label_from_parent_folder"
        )
        label_of: Callable[[Path], Label | None] = label_from_parent_folder
        generator_of: Callable[[Path], str | None] | None = None
        split_of: Callable[[Path], Split | None] | None = None
        mask_of: Callable[[Path], Path | None] | None = None
    else:
        label_of, generator_of, split_of, mask_of = _callables_from_layout(layout, root)

    attributes = _load_attributes(root)
    if attributes:
        generator_of, split_of = _wrap_with_attributes(root, attributes, generator_of, split_of)

    manifest, skipped = build_manifest(
        root,
        dataset=dataset,
        label_of=label_of,
        generator_of=generator_of,
        split_of=split_of,
        mask_of=mask_of,
        license=resolved_license,
        commercial_ok=resolved_commercial_ok,
        progress=False,
    )

    manifest.save(out_path)
    report = audit_manifest(manifest, strict=False)
    return manifest, skipped, report


#: Name of the sidecar file :func:`materialize_parquet` writes next to the
#: image tree, and that :func:`prepare` looks for at ``<src_root>/attributes.jsonl``.
_ATTRIBUTES_FILENAME = "attributes.jsonl"


def _load_attributes(src_root: Path) -> dict[str, dict[str, Any]]:
    """Load ``<src_root>/attributes.jsonl`` (see :func:`materialize_parquet`), if present.

    Returns a ``{root-relative POSIX path: record}`` mapping, or an empty
    dict when the file does not exist. A line that fails to parse as JSON or
    carries no ``"path"`` key is skipped rather than aborting the whole load.
    """
    path = src_root / _ATTRIBUTES_FILENAME
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            rel_path = record.get("path")
            if isinstance(rel_path, str):
                records[rel_path] = record
    return records


def _wrap_with_attributes(
    root: Path,
    attributes: dict[str, dict[str, Any]],
    generator_of: Callable[[Path], str | None] | None,
    split_of: Callable[[Path], Split | None] | None,
) -> tuple[Callable[[Path], str | None], Callable[[Path], Split | None]]:
    """Prefer ``attributes.jsonl``'s ``model_name``/``split`` over the folder-derived values.

    Falls back to ``generator_of``/``split_of`` (the layout- or
    ``label_from_parent_folder``-derived callables, which may themselves be
    ``None``) for any path the sidecar has no record for, or whose record
    leaves the field unset -- so a partially-covered sidecar never loses
    information the folder structure would otherwise have provided.
    """

    def wrapped_generator_of(path: Path) -> str | None:
        record = attributes.get(_relative_posix(path, root))
        if record is not None:
            model_name = record.get("model_name")
            if model_name:
                return str(model_name)
        return generator_of(path) if generator_of is not None else None

    def wrapped_split_of(path: Path) -> Split | None:
        record = attributes.get(_relative_posix(path, root))
        if record is not None:
            split_value = record.get("split")
            if split_value in _SPLIT_VALUES:
                return split_value  # type: ignore[return-value]
        return split_of(path) if split_of is not None else None

    return wrapped_generator_of, wrapped_split_of


# --- Community Forensics Small: Parquet materialization -------------------

#: Column names verified against the OwensLab/CommunityForensics-Small
#: dataset-server schema (2026-09-09): image_name, format, resolution, mode,
#: image_data, model_name, nsfw_flag, prompt, real_source, subset, split,
#: label, architecture. ``resolution``/``mode`` are not needed to
#: materialize a row and are not read.
_PARQUET_NAME_COLUMN = "image_name"
_PARQUET_FORMAT_COLUMN = "format"
_PARQUET_IMAGE_COLUMN = "image_data"
_PARQUET_GENERATOR_COLUMN = "model_name"
_PARQUET_NSFW_COLUMN = "nsfw_flag"
_PARQUET_PROMPT_COLUMN = "prompt"
_PARQUET_REAL_SOURCE_COLUMN = "real_source"
_PARQUET_SUBSET_COLUMN = "subset"
_PARQUET_SPLIT_COLUMN = "split"
_PARQUET_LABEL_COLUMN = "label"
_PARQUET_ARCHITECTURE_COLUMN = "architecture"

#: Columns actually read from each shard (a strict subset of the schema
#: above keeps the per-batch memory footprint down).
_PARQUET_COLUMNS = (
    _PARQUET_NAME_COLUMN,
    _PARQUET_FORMAT_COLUMN,
    _PARQUET_IMAGE_COLUMN,
    _PARQUET_GENERATOR_COLUMN,
    _PARQUET_NSFW_COLUMN,
    _PARQUET_PROMPT_COLUMN,
    _PARQUET_REAL_SOURCE_COLUMN,
    _PARQUET_SUBSET_COLUMN,
    _PARQUET_SPLIT_COLUMN,
    _PARQUET_LABEL_COLUMN,
    _PARQUET_ARCHITECTURE_COLUMN,
)
#: Columns without which a row cannot be materialized at all.
_PARQUET_REQUIRED_COLUMNS = (
    _PARQUET_NAME_COLUMN,
    _PARQUET_FORMAT_COLUMN,
    _PARQUET_IMAGE_COLUMN,
    _PARQUET_LABEL_COLUMN,
)

#: Row-group batch size for :func:`materialize_parquet`'s streaming read --
#: small enough that a whole shard is never resident in memory at once.
_PARQUET_BATCH_SIZE = 512
#: Extension to write when a row's ``format`` value isn't a recognized name.
_UNKNOWN_FORMAT_EXTENSION = "bin"
#: Longest a stored ``prompt`` attribute may be (characters).
_PROMPT_TRUNCATE_LENGTH = 200

_FORMAT_EXTENSIONS: dict[str, str] = {
    "PNG": "png",
    "JPEG": "jpg",
    "JPG": "jpg",
    "WEBP": "webp",
    "BMP": "bmp",
    "TIFF": "tiff",
    "TIF": "tiff",
    "GIF": "gif",
}


@dataclass
class MaterializeReport:
    """Outcome of one :func:`materialize_parquet` call.

    ``skipped`` counts rows that were read but not written, keyed by reason
    (``"nsfw"``, ``"missing_image_data"``, ``"already_exists"``).
    ``by_label``/``by_generator``/``formats`` count every row that was (or
    already had been) materialized -- i.e. everything not skipped -- keyed
    by the manifest ``label`` it maps to, the folder-name generator it was
    written under, and its raw ``format`` column value, respectively.
    """

    rows_read: int = 0
    written: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    by_label: dict[str, int] = field(default_factory=dict)
    by_generator: dict[str, int] = field(default_factory=dict)
    formats: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def _bump(counts: dict[str, int], key: str) -> None:
        counts[key] = counts.get(key, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_read": self.rows_read,
            "written": self.written,
            "skipped": self.skipped,
            "by_label": self.by_label,
            "by_generator": self.by_generator,
            "formats": self.formats,
        }


def _extension_for_format(fmt: str | None) -> str:
    """The lower-case file extension to write for a Parquet ``format`` value.

    Falls back to the lower-cased raw value for a format not in the known
    table (still deterministic, still a valid extension for most strings),
    or :data:`_UNKNOWN_FORMAT_EXTENSION` when ``fmt`` is empty/``None``.
    """
    if not fmt:
        return _UNKNOWN_FORMAT_EXTENSION
    key = fmt.strip().upper()
    return _FORMAT_EXTENSIONS.get(key, key.lower())


def _label_to_dir(value: Any) -> Label:
    """Map a raw ``label`` column value to ``"real"``/``"fake"``.

    Community Forensics Small's own dataset-server schema uses ``1`` =
    generated/fake, ``0`` = real, but a Parquet shard from a different
    export could carry ``"real"``/``"fake"`` strings or booleans instead --
    all three are handled. ``bool`` is checked before ``int`` because
    Python's ``bool`` is an ``int`` subclass and would otherwise match the
    integer branch first (harmlessly here, since ``True``/``False`` compare
    equal to ``1``/``0``, but the explicit check keeps the mapping obvious).

    Raises:
        ValueError: ``value`` is not a recognized label encoding, naming
            the offending value and its type -- this fails loudly rather
            than silently guessing a label.
    """
    if isinstance(value, bool):
        return "fake" if value else "real"
    if isinstance(value, int):
        if value == 1:
            return "fake"
        if value == 0:
            return "real"
        raise ValueError(f"Unrecognized integer label value {value!r} (expected 0 or 1)")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "fake":
            return "fake"
        if lowered == "real":
            return "real"
        raise ValueError(f"Unrecognized string label value {value!r} (expected 'real' or 'fake')")
    raise ValueError(
        f"Unrecognized label value {value!r} of type {type(value).__name__} "
        "(expected an int 0/1, a bool, or a 'real'/'fake' string)"
    )


def materialize_parquet(
    src_dir: str | Path,
    out_dir: str | Path,
    *,
    max_rows: int | None = None,
    progress: bool = True,
) -> MaterializeReport:
    """Materialize Community Forensics Small's Parquet shards into a real/fake image tree.

    Community Forensics Small ships as Parquet shards (``data/*.parquet``)
    rather than a real/fake image tree, so :func:`prepare` cannot walk it
    directly. This reads every ``*.parquet`` file in ``src_dir`` one row
    group at a time (via :class:`pyarrow.parquet.ParquetFile`, never
    ``read_table`` on a whole shard) and, for each row:

    - Skips it (counted under ``skipped["nsfw"]``) when ``nsfw_flag`` is true.
    - Skips it (``skipped["missing_image_data"]``) when ``image_data`` is
      ``None``.
    - Maps ``label`` to ``real``/``fake`` with :func:`_label_to_dir` --
      raising loudly on an unrecognized value rather than guessing.
    - Writes the row's raw ``image_data`` bytes unchanged (no re-encoding)
      to ``<out_dir>/<real|fake>/<generator>/<split>/<image_name>.<ext>``,
      where ``generator`` is ``model_name`` for a fake row and
      ``real_source`` (or ``"real"`` when unset) for a real row, ``split``
      falls back to ``"none"`` when unset, and ``ext`` comes from the
      ``format`` column via :func:`_extension_for_format`. That shape
      matches the packaged ``layouts.yaml`` entry for "Community Forensics"
      (``generator_from: grandparent``, ``split_from: parent``), so
      ``prepare("Community Forensics", out_dir, ...)`` walks the result
      directly.
    - Is skipped (``skipped["already_exists"]``) instead of rewritten when
      the target file already exists -- re-running this on a partially (or
      fully) materialized ``out_dir`` is safe and cheap.

    Also writes ``<out_dir>/attributes.jsonl``, one JSON line per row that
    maps to a file present in ``out_dir`` after this call (whether just
    written or already there), recording that file's root-relative path
    plus ``model_name``, ``architecture``, ``subset``, ``split`` and a
    ``prompt`` truncated to :data:`_PROMPT_TRUNCATE_LENGTH` characters --
    see :func:`prepare`, which prefers this sidecar's ``model_name``/
    ``split`` over the folder-derived ones when it exists. And
    ``<out_dir>/materialize.json``, a JSON dump of the returned
    :class:`MaterializeReport`.

    Requires ``pyarrow`` (the optional ``data`` extra -- deliberately kept
    out of the base dependency set, see ``docs/ROADMAP.md`` section 3's
    small-dependency principle). When it is not installed, this prints
    manual instructions and returns an all-zero :class:`MaterializeReport`
    instead of raising, matching this project's pattern of degrading to
    instructions rather than failing hard on an optional extra (see
    ``imgforensics.signals.c2pa``).

    Args:
        src_dir: Directory containing the downloaded ``*.parquet`` shards.
        out_dir: Destination folder tree (created if missing).
        max_rows: Stop after reading this many rows total, across every
            shard (sorted by filename); ``None`` reads every row of every
            shard.
        progress: Print a running row count every :data:`_PARQUET_BATCH_SIZE`
            rows.

    Raises:
        ValueError: a shard is missing one of the columns in
            :data:`_PARQUET_REQUIRED_COLUMNS`, or a row's ``label`` value is
            not recognized by :func:`_label_to_dir`.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print(
            "[materialize_parquet] pyarrow is not installed; install it with "
            'pip install "imgforensics[data]" (or `pip install pyarrow` directly), '
            "or use the Hugging Face `datasets` library to read the Parquet shards "
            "yourself, then write <out_dir>/<real|fake>/<generator>/<split>/<name> "
            "per row."
        )
        return MaterializeReport()

    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = MaterializeReport()
    attribute_lines: list[str] = []
    rows_seen = 0

    for shard_path in sorted(src_dir.glob("*.parquet")):
        if max_rows is not None and rows_seen >= max_rows:
            break

        parquet_file = pq.ParquetFile(shard_path)
        available = set(parquet_file.schema_arrow.names)
        missing_required = [c for c in _PARQUET_REQUIRED_COLUMNS if c not in available]
        if missing_required:
            raise ValueError(f"{shard_path}: missing required column(s) {missing_required}")
        read_columns = [c for c in _PARQUET_COLUMNS if c in available]

        for batch in parquet_file.iter_batches(
            batch_size=_PARQUET_BATCH_SIZE, columns=read_columns
        ):
            if max_rows is not None and rows_seen >= max_rows:
                break
            for row in batch.to_pylist():
                if max_rows is not None and rows_seen >= max_rows:
                    break
                rows_seen += 1
                report.rows_read += 1
                if progress and report.rows_read % _PARQUET_BATCH_SIZE == 0:
                    print(f"[materialize_parquet] {report.rows_read} rows read")

                if bool(row.get(_PARQUET_NSFW_COLUMN)):
                    report._bump(report.skipped, "nsfw")
                    continue

                image_bytes = row.get(_PARQUET_IMAGE_COLUMN)
                if image_bytes is None:
                    report._bump(report.skipped, "missing_image_data")
                    continue

                label = _label_to_dir(row.get(_PARQUET_LABEL_COLUMN))
                model_name = row.get(_PARQUET_GENERATOR_COLUMN)
                real_source = row.get(_PARQUET_REAL_SOURCE_COLUMN)
                generator = str(model_name if label == "fake" else (real_source or "real"))

                split_value = row.get(_PARQUET_SPLIT_COLUMN)
                split_dir = str(split_value) if split_value else "none"

                fmt_value = row.get(_PARQUET_FORMAT_COLUMN)
                report._bump(report.formats, str(fmt_value) if fmt_value else "unknown")
                extension = _extension_for_format(fmt_value)

                name_value = row[_PARQUET_NAME_COLUMN]
                target_dir = out_dir / label / generator / split_dir
                target_path = target_dir / f"{name_value}.{extension}"

                report._bump(report.by_label, label)
                report._bump(report.by_generator, generator)

                if target_path.exists():
                    report._bump(report.skipped, "already_exists")
                else:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_path.write_bytes(image_bytes)
                    report.written += 1

                prompt_value = row.get(_PARQUET_PROMPT_COLUMN)
                if isinstance(prompt_value, str):
                    prompt_value = prompt_value[:_PROMPT_TRUNCATE_LENGTH]
                attribute_lines.append(
                    json.dumps(
                        {
                            "path": target_path.relative_to(out_dir).as_posix(),
                            "model_name": model_name,
                            "architecture": row.get(_PARQUET_ARCHITECTURE_COLUMN),
                            "subset": row.get(_PARQUET_SUBSET_COLUMN),
                            "split": split_value,
                            "prompt": prompt_value,
                        }
                    )
                )

    attributes_text = "".join(f"{line}\n" for line in attribute_lines)
    (out_dir / "attributes.jsonl").write_text(attributes_text, encoding="utf-8")
    (out_dir / "materialize.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    return report
