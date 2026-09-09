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

import re
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Literal

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
        manifest, skipped = build_manifest(
            root,
            dataset=dataset,
            label_of=label_from_parent_folder,
            license=resolved_license,
            commercial_ok=resolved_commercial_ok,
            progress=False,
        )
    else:
        label_of, generator_of, split_of, mask_of = _callables_from_layout(layout, root)
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


# --- Community Forensics Small: Parquet materialization -------------------

#: Column names verified against the OwensLab/CommunityForensics-Small
#: dataset-server schema (2026-09-09): image bytes, integer label (1 =
#: generated/fake, 0 = real, matching this dataset's own convention), the
#: generator name, and the train/val/test split, all present per row so no
#: folder-name inference is needed once materialized.
_PARQUET_IMAGE_COLUMN = "image_data"
_PARQUET_LABEL_COLUMN = "label"
_PARQUET_GENERATOR_COLUMN = "model_name"
_PARQUET_SPLIT_COLUMN = "split"
_PARQUET_NAME_COLUMN = "image_name"


def materialize_parquet(
    parquet_dir: str | Path,
    out_dir: str | Path,
    *,
    image_column: str = _PARQUET_IMAGE_COLUMN,
    label_column: str = _PARQUET_LABEL_COLUMN,
    generator_column: str = _PARQUET_GENERATOR_COLUMN,
    split_column: str = _PARQUET_SPLIT_COLUMN,
    name_column: str = _PARQUET_NAME_COLUMN,
    fake_label_value: int = 1,
) -> int:
    """Materialize Parquet-shard rows (Community Forensics Small's format) into a folder tree.

    Community Forensics Small ships as Parquet shards
    (``data/*.parquet``) rather than a real/fake image tree, so
    :func:`prepare` cannot walk it directly. This helper reads every
    ``*.parquet`` file in ``parquet_dir`` and writes each row's image bytes
    to ``<out_dir>/<real|fake>/<generator or "unknown">/<split or "none">/<image_name>``.
    That shape matches the packaged ``layouts.yaml`` entry for "Community
    Forensics" (``real_globs``/``fake_globs`` of ``real/**``/``fake/**``,
    ``generator_from: grandparent``, ``split_from: parent``) -- so after
    materializing, ``prepare("Community Forensics", out_dir, ...)`` walks
    the result directly.

    Requires ``pyarrow`` (not an imgforensics dependency -- deliberately
    excluded, see ``docs/ROADMAP.md`` section 3's small-dependency
    principle). When it is not installed, this prints manual instructions
    and returns ``0`` instead of raising, matching this project's pattern
    of degrading to instructions rather than failing hard on an optional
    extra (see ``imgforensics.signals.c2pa``).

    Args:
        parquet_dir: Directory containing the downloaded ``*.parquet`` shards.
        out_dir: Destination folder tree.
        image_column: Column holding raw image bytes.
        label_column: Column whose value equals ``fake_label_value`` for a
            generated image, anything else for real.
        generator_column: Column holding the generator/model name.
        split_column: Column holding the train/val/test split name.
        name_column: Column holding the output file's base name.
        fake_label_value: The label value that means "fake"/generated.

    Returns:
        Number of images written (``0`` when pyarrow is unavailable).
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print(
            "[materialize_parquet] pyarrow is not installed; install it manually "
            "(pip install pyarrow) or use the Hugging Face `datasets` library to "
            "read the Parquet shards yourself, then write "
            "<out_dir>/<real|fake>/<generator>/<name> per row."
        )
        return 0

    parquet_dir = Path(parquet_dir)
    out_dir = Path(out_dir)
    written = 0
    for shard_path in sorted(parquet_dir.glob("*.parquet")):
        table = pq.read_table(shard_path)
        columns = {
            image_column: table.column(image_column),
            label_column: table.column(label_column),
            generator_column: table.column(generator_column)
            if generator_column in table.column_names
            else None,
            split_column: table.column(split_column)
            if split_column in table.column_names
            else None,
            name_column: table.column(name_column) if name_column in table.column_names else None,
        }
        num_rows = table.num_rows
        for row in range(num_rows):
            image_bytes = columns[image_column][row].as_py()
            if image_bytes is None:
                continue
            label_value = columns[label_column][row].as_py()
            label_dir = "fake" if label_value == fake_label_value else "real"
            generator = (
                columns[generator_column][row].as_py()
                if columns[generator_column] is not None
                else None
            )
            generator_dir = generator or "unknown"
            name = (
                columns[name_column][row].as_py()
                if columns[name_column] is not None
                else f"{shard_path.stem}_{row}.png"
            )
            split_value = (
                columns[split_column][row].as_py() if columns[split_column] is not None else None
            )
            # Always emit a split segment (default "none") so every image sits at the
            # same path depth -- the layout's fixed generator_from="grandparent" /
            # split_from="parent" would otherwise misread a variable-depth path.
            split_dir = split_value or "none"
            target_dir = out_dir / label_dir / generator_dir / split_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / name).write_bytes(image_bytes)
            written += 1
    return written
