"""Dataset acquisition: license-gated download recipes for the registry's datasets.

A dataset in ``imgforensics.data.registry`` is a catalogue entry, not a
download button: nothing in this repository is redistributed (see
``docs/ROADMAP.md``, section 7). This module turns a registry entry into an
actual download by pairing it with an :class:`AcquireRecipe` from the
packaged ``acquire.yaml`` and running its steps with :func:`fetch`.

Every recipe step is one of four methods:

- ``http`` -- a plain streaming download (stdlib ``urllib``, no extra
  dependency) with ``.part``-file resume via an HTTP ``Range`` request,
  optional sha256 verification, and optional zip/tar unpacking.
- ``hf`` -- ``huggingface_hub.snapshot_download`` (optional ``data`` extra),
  imported lazily so the base package never requires it. When the step sets
  ``max_files``, a bounded path is used instead: list the repo's files,
  filter with ``allow_patterns``, keep the first ``max_files`` (sorted), and
  download just those via ``huggingface_hub.hf_hub_download`` -- for a repo
  whose full contents exceed this project's disk budget.
- ``gdrive`` -- ``gdown.download`` (optional ``data`` extra), also lazy.
- ``manual`` -- prints instructions and does nothing else; used whenever a
  dataset has no stable, scriptable download (a signed/expiring URL, a
  request form, or credentials this project never handles).

:func:`fetch` never accepts or stores credentials: a dataset that needs a
Kaggle token, a Hugging Face access token, or a login only ever reaches this
module through a ``manual`` step.

The license gate is unconditional: :func:`fetch` always prints the
registry's license text and ``commercial_ok`` flag before doing anything,
and refuses to download (raising :class:`LicenseNotAcceptedError`) unless
the caller passes ``accept_license=True`` -- except in ``dry_run`` mode,
which only prints the plan and downloads nothing.
"""

from __future__ import annotations

import fnmatch
import hashlib
import posixpath
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import date
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from imgforensics.data.registry import DatasetInfo, get_dataset

#: Bytes read per HTTP chunk (also the resume/progress granularity).
_CHUNK_SIZE = 1 << 16  # 64 KiB
#: Print an in-progress line roughly every this many bytes (see _http_download).
_PROGRESS_STRIDE = _CHUNK_SIZE * 32


class LicenseNotAcceptedError(RuntimeError):
    """Raised by :func:`fetch` when the license was not accepted and this is not a dry run."""


class ChecksumMismatchError(RuntimeError):
    """Raised when a downloaded file's sha256 does not match the recipe's expected value.

    The partially/fully downloaded file is deleted before this is raised, so
    a corrupt or tampered download never lingers on disk as if it were good.
    """


class AcquireStep(BaseModel):
    """One step of an :class:`AcquireRecipe`: a single download or a manual instruction.

    Only the fields relevant to ``method`` need be set in YAML; the rest
    default to ``None``/empty. A :func:`~pydantic.model_validator` checks
    that the fields required by each method are present when the recipe is
    loaded, so a malformed ``acquire.yaml`` entry fails at load time rather
    than mid-download.
    """

    method: Literal["http", "hf", "gdrive", "manual"]

    # method == "http"
    url: str | None = None
    filename: str | None = None
    sha256: str | None = None
    unpack: bool = False

    # method == "hf"
    repo_id: str | None = None
    repo_type: Literal["dataset"] | None = None
    allow_patterns: list[str] = Field(default_factory=list)
    revision: str | None = None
    #: When set, bound the download to the first ``max_files`` files (sorted,
    #: after filtering by ``allow_patterns``) instead of the whole repo --
    #: see the module docstring's ``hf`` bullet.
    max_files: int | None = None

    # method == "gdrive" (also uses filename/unpack above)
    file_id: str | None = None

    # method == "manual"
    instructions: str | None = None

    @model_validator(mode="after")
    def _check_required_fields(self) -> AcquireStep:
        if self.method == "http":
            if not self.url or not self.filename:
                raise ValueError("an 'http' step requires 'url' and 'filename'")
        elif self.method == "hf":
            if not self.repo_id:
                raise ValueError("an 'hf' step requires 'repo_id'")
            if self.repo_type is None:
                self.repo_type = "dataset"
        elif self.method == "gdrive":
            if not self.file_id or not self.filename:
                raise ValueError("a 'gdrive' step requires 'file_id' and 'filename'")
        elif self.method == "manual" and not self.instructions:
            raise ValueError("a 'manual' step requires 'instructions'")
        return self


class AcquireRecipe(BaseModel):
    """The download recipe for one registry dataset.

    ``dataset`` must match a name in ``imgforensics.data.registry``.
    ``variants`` names alternative step lists (e.g. ``"val_only"``,
    ``"train"``, ``"small"``) selectable via :func:`fetch`'s ``variant``
    argument; ``steps`` is what runs when no variant is requested.
    """

    dataset: str
    steps: list[AcquireStep]
    subset_note: str | None = None
    variants: dict[str, list[AcquireStep]] | None = None


@dataclass
class StepReport:
    """Outcome of running one :class:`AcquireStep`."""

    method: str
    description: str
    status: Literal["done", "skipped", "pending"]
    bytes_downloaded: int = 0


@dataclass
class FetchReport:
    """Outcome of a full :func:`fetch` call: every step's :class:`StepReport`."""

    dataset: str
    dest: Path
    accepted_license: bool
    dry_run: bool
    steps: list[StepReport] = field(default_factory=list)


def load_acquire_recipes() -> list[AcquireRecipe]:
    """Load and validate every recipe from the packaged ``acquire.yaml``."""
    text = resources.files("imgforensics.data").joinpath("acquire.yaml").read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    return [AcquireRecipe.model_validate(item) for item in raw.get("recipes", [])]


def get_recipe(dataset: str) -> AcquireRecipe:
    """Return the acquisition recipe for ``dataset`` (exact, case-sensitive match).

    Raises:
        KeyError: if no recipe is registered for ``dataset``, listing the
            available names in the error message.
    """
    for recipe in load_acquire_recipes():
        if recipe.dataset == dataset:
            return recipe
    known = ", ".join(sorted(r.dataset for r in load_acquire_recipes()))
    raise KeyError(
        f"No acquisition recipe registered for {dataset!r}. Available: {known or '<none>'}"
    )


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _http_download(url: str, target: Path, *, sha256: str | None, progress: bool) -> int:
    """Stream ``url`` to ``target`` via a ``<target>.part`` file, resuming if it exists.

    Sends ``Range: bytes=<already-downloaded>-`` when a ``.part`` file is
    present. If the server honors it (HTTP 206), download appends; if the
    server ignores it and returns a fresh 200 response, the ``.part`` file
    is truncated and the download restarts from zero rather than corrupting
    it with a byte-offset mismatch. On success the ``.part`` file is renamed
    to ``target``; on a checksum mismatch it is deleted.

    Returns:
        Total bytes written to the file (including any resumed prefix).
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    part_path = target.with_name(target.name + ".part")
    resume_from = part_path.stat().st_size if part_path.exists() else 0

    request = urllib.request.Request(url)
    if resume_from:
        request.add_header("Range", f"bytes={resume_from}-")

    with urllib.request.urlopen(request) as response:  # noqa: S310 -- recipe URLs are curated, not user input
        resumed = resume_from > 0 and response.status == 206
        mode = "ab" if resumed else "wb"
        written = resume_from if resumed else 0
        with part_path.open(mode) as handle:
            next_report = written + _PROGRESS_STRIDE
            while True:
                chunk = response.read(_CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                if progress and written >= next_report:
                    print(f"[fetch] {target.name}: {written} bytes")
                    next_report = written + _PROGRESS_STRIDE

    if sha256 is not None:
        digest = _sha256_of_file(part_path)
        if digest != sha256:
            part_path.unlink(missing_ok=True)
            raise ChecksumMismatchError(
                f"{target.name}: sha256 mismatch (expected {sha256}, got {digest}); file deleted"
            )

    part_path.replace(target)
    return written


def _check_safe_member(member_name: str, dest_dir: Path) -> None:
    """Raise ``ValueError`` if ``member_name`` would extract outside ``dest_dir``.

    Rejects an absolute path and any path that still contains a ``..``
    segment after normalization (e.g. ``sub/../../evil.txt``), then -- as a
    second, independent check -- rejects anything whose resolved location
    falls outside ``dest_dir.resolve()``. Used for both zip and tar members,
    on every Python version, before any extraction happens.
    """
    normalized = posixpath.normpath(member_name.replace("\\", "/"))
    if (
        PurePosixPath(normalized).is_absolute()
        or normalized == ".."
        or normalized.startswith("../")
    ):
        raise ValueError(f"refusing to extract unsafe archive member path: {member_name!r}")

    dest_resolved = dest_dir.resolve()
    target_resolved = (dest_dir / normalized).resolve()
    if target_resolved != dest_resolved and dest_resolved not in target_resolved.parents:
        raise ValueError(f"refusing to extract unsafe archive member path: {member_name!r}")


def _unpack(archive: Path, dest_dir: Path) -> None:
    """Extract ``archive`` (zip or tar, any tar compression) into ``dest_dir``.

    Every member's path is checked with :func:`_check_safe_member` before
    any file is extracted, rejecting an absolute path or a ``..`` escape out
    of ``dest_dir`` (raising ``ValueError`` naming the offending member).
    This runs on every Python version; the tarfile ``filter="data"``
    extraction filter (PEP 706, Python 3.12+) is applied on top of it as an
    additional layer where available, not as a replacement for it.
    """
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                _check_safe_member(info.filename, dest_dir)
            zf.extractall(dest_dir)
        return
    if name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")):
        with tarfile.open(archive) as tf:
            members = tf.getmembers()
            for member in members:
                _check_safe_member(member.name, dest_dir)
            if hasattr(tarfile, "data_filter"):  # Python 3.12+ extraction filter (PEP 706)
                tf.extractall(dest_dir, filter="data")
            else:
                tf.extractall(dest_dir)  # noqa: S202 -- members were already path-checked above
        return
    raise ValueError(
        f"Don't know how to unpack {archive.name!r} (expected .zip or a .tar* archive)"
    )


def _hf_snapshot_download(
    *,
    repo_id: str,
    repo_type: str,
    allow_patterns: list[str] | None,
    local_dir: Path,
    revision: str | None,
) -> str:
    """Thin, lazily-importing wrapper around ``huggingface_hub.snapshot_download``.

    Kept as a standalone function (rather than inlined) so tests can
    monkeypatch it directly without installing ``huggingface_hub``.
    """
    from huggingface_hub import snapshot_download  # optional 'data' extra

    return str(
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            allow_patterns=allow_patterns,
            local_dir=str(local_dir),
            revision=revision,
        )
    )


def _hf_list_repo_files(*, repo_id: str, repo_type: str, revision: str | None) -> list[str]:
    """Thin, lazily-importing wrapper around ``huggingface_hub.list_repo_files``.

    Kept as a standalone function (rather than inlined) so tests can
    monkeypatch it directly without installing ``huggingface_hub``. Used by
    the bounded ``max_files`` download path to list a repo's files before
    picking which ones to fetch.
    """
    from huggingface_hub import list_repo_files  # optional 'data' extra

    return list(list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=revision))


def _hf_hub_download(
    *, repo_id: str, repo_type: str, filename: str, local_dir: Path, revision: str | None
) -> str:
    """Thin, lazily-importing wrapper around ``huggingface_hub.hf_hub_download``.

    Kept as a standalone function (rather than inlined) so tests can
    monkeypatch it directly without installing ``huggingface_hub``. Used by
    the bounded ``max_files`` download path to fetch one file at a time,
    instead of ``snapshot_download``'s whole-repo (filtered) pull.
    """
    from huggingface_hub import hf_hub_download  # optional 'data' extra

    return str(
        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            local_dir=str(local_dir),
            revision=revision,
        )
    )


def _gdown_download(*, file_id: str, output: Path) -> str | None:
    """Thin, lazily-importing wrapper around ``gdown.download``.

    Kept as a standalone function (rather than inlined) so tests can
    monkeypatch it directly without installing ``gdown``. Returns ``None``
    on failure (gdown's own convention), which :func:`fetch` reports as a
    ``"skipped"`` step rather than raising.
    """
    import gdown  # optional 'data' extra

    result = gdown.download(id=file_id, output=str(output), quiet=True)
    return result if isinstance(result, str) else None


def _describe_step(step: AcquireStep) -> str:
    if step.method == "http":
        return f"http: {step.url} -> {step.filename}"
    if step.method == "hf":
        patterns = ", ".join(step.allow_patterns) if step.allow_patterns else "*"
        max_files_part = f", max_files={step.max_files}" if step.max_files is not None else ""
        return (
            f"hf: {step.repo_id} (revision={step.revision or 'default'}, "
            f"patterns=[{patterns}]{max_files_part})"
        )
    if step.method == "gdrive":
        return f"gdrive: {step.file_id} -> {step.filename}"
    first_line = (step.instructions or "").strip().splitlines()[0] if step.instructions else ""
    return f"manual: {first_line}"


def _run_step(step: AcquireStep, dataset_dir: Path, *, progress: bool) -> StepReport:
    description = _describe_step(step)

    if step.method == "manual":
        print(step.instructions)
        return StepReport(method="manual", description=description, status="pending")

    if step.method == "http":
        assert step.url is not None and step.filename is not None
        target = dataset_dir / step.filename
        downloaded = _http_download(step.url, target, sha256=step.sha256, progress=progress)
        if step.unpack:
            _unpack(target, dataset_dir)
        return StepReport(
            method="http", description=description, status="done", bytes_downloaded=downloaded
        )

    if step.method == "hf":
        assert step.repo_id is not None
        repo_type = step.repo_type or "dataset"
        if step.max_files is not None:
            all_files = _hf_list_repo_files(
                repo_id=step.repo_id, repo_type=repo_type, revision=step.revision
            )
            patterns = step.allow_patterns or ["*"]
            matched = sorted(
                name for name in all_files if any(fnmatch.fnmatch(name, pat) for pat in patterns)
            )
            selected = matched[: step.max_files]
            for filename in selected:
                _hf_hub_download(
                    repo_id=step.repo_id,
                    repo_type=repo_type,
                    filename=filename,
                    local_dir=dataset_dir,
                    revision=step.revision,
                )
            total_bytes = sum(
                (dataset_dir / filename).stat().st_size
                for filename in selected
                if (dataset_dir / filename).exists()
            )
            return StepReport(
                method="hf",
                description=f"{description} -> {len(selected)} file(s) selected",
                status="done",
                bytes_downloaded=total_bytes,
            )
        _hf_snapshot_download(
            repo_id=step.repo_id,
            repo_type=repo_type,
            allow_patterns=step.allow_patterns or None,
            local_dir=dataset_dir,
            revision=step.revision,
        )
        return StepReport(method="hf", description=description, status="done")

    if step.method == "gdrive":
        assert step.file_id is not None and step.filename is not None
        target = dataset_dir / step.filename
        result = _gdown_download(file_id=step.file_id, output=target)
        if result is None:
            return StepReport(method="gdrive", description=description, status="skipped")
        if step.unpack and target.exists():
            _unpack(target, dataset_dir)
        size = target.stat().st_size if target.exists() else 0
        return StepReport(
            method="gdrive", description=description, status="done", bytes_downloaded=size
        )

    raise ValueError(f"Unknown acquisition method: {step.method!r}")  # pragma: no cover


def _write_license_acceptance(dataset_dir: Path, info: DatasetInfo) -> Path:
    path = dataset_dir / "LICENSE_ACCEPTED.txt"
    lines = [
        f"Dataset: {info.name}",
        f"License: {info.license}",
        f"Commercial OK: {info.commercial_ok}",
        f"Accepted: {date.today().isoformat()}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _resolve_steps(recipe: AcquireRecipe, variant: str | None) -> list[AcquireStep]:
    if variant is None:
        return recipe.steps
    variants = recipe.variants or {}
    if variant not in variants:
        known = ", ".join(sorted(variants))
        raise KeyError(
            f"No variant {variant!r} for {recipe.dataset!r}. Available: {known or '<none>'}"
        )
    return variants[variant]


def fetch(
    dataset: str,
    dest: str | Path,
    *,
    variant: str | None = None,
    accept_license: bool = False,
    dry_run: bool = False,
    progress: bool = True,
) -> FetchReport:
    """Download ``dataset`` into ``<dest>/<dataset>/`` per its packaged recipe.

    Always prints the dataset's name, license, ``commercial_ok`` flag, and
    homepage first (from the :mod:`imgforensics.data.registry` entry, never
    from the recipe). Unless ``dry_run`` is true, refuses to proceed with
    :class:`LicenseNotAcceptedError` unless ``accept_license`` is true, and
    -- before running any step -- writes ``<dest>/<dataset>/LICENSE_ACCEPTED.txt``
    recording the dataset name, license text, ``commercial_ok``, and today's
    date. ``dry_run=True`` only prints the plan (dataset info, recipe note,
    and every step's description) and downloads nothing; it does not require
    ``accept_license`` and does not write the acceptance file.

    Args:
        dataset: A name present in both the dataset registry and
            ``acquire.yaml`` (mismatches raise :class:`KeyError`).
        dest: Destination directory; the dataset is written to
            ``<dest>/<dataset>/``.
        variant: Selects ``recipe.variants[variant]`` instead of
            ``recipe.steps``.
        accept_license: Must be true (outside ``dry_run``) to proceed past
            the license gate.
        dry_run: Print the plan only; no network access, no files written.
        progress: Print periodic byte-count lines during ``http`` downloads.

    Returns:
        A :class:`FetchReport` listing every step's outcome.

    Raises:
        KeyError: ``dataset`` (or ``variant``) is not registered.
        LicenseNotAcceptedError: ``accept_license`` is false and this is not
            a dry run.
        ChecksumMismatchError: an ``http`` step's downloaded file does not
            match its recorded sha256.
    """
    info = get_dataset(dataset)
    recipe = get_recipe(dataset)
    steps = _resolve_steps(recipe, variant)

    print(f"Dataset: {info.name}")
    print(f"License: {info.license}")
    print(f"Commercial OK: {info.commercial_ok}")
    print(f"Homepage: {info.homepage or '-'}")
    if recipe.subset_note:
        print(f"Note: {recipe.subset_note}")

    dataset_dir = Path(dest) / dataset

    if dry_run:
        print("Dry run: no network access, no files written.")
        for step in steps:
            print(f"  - {_describe_step(step)}")
        step_reports = [
            StepReport(method=step.method, description=_describe_step(step), status="pending")
            for step in steps
        ]
        return FetchReport(
            dataset=dataset,
            dest=dataset_dir,
            accepted_license=False,
            dry_run=True,
            steps=step_reports,
        )

    if not accept_license:
        raise LicenseNotAcceptedError(
            f"license not accepted for {dataset!r}: re-run with accept_license=True "
            "(CLI: --accept-license) after reading the license printed above."
        )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    _write_license_acceptance(dataset_dir, info)

    step_reports = [_run_step(step, dataset_dir, progress=progress) for step in steps]
    return FetchReport(
        dataset=dataset, dest=dataset_dir, accepted_license=True, dry_run=False, steps=step_reports
    )
