"""License-gated download of the pretrained localizer weights.

The dataset half of this project already refuses to download anything until
its license has been read and accepted (:mod:`imgforensics.data.acquire`).
Model weights get the same treatment and for the same reason
(``docs/ROADMAP.md``, section 7): nothing third-party is committed to this
repository, users fetch it from the original source, and the license is
printed every time before a byte moves.

This module is the weights counterpart of that: a small registry of
:class:`WeightSpec` entries and :func:`fetch_weights`, which reuses
:mod:`imgforensics.data.acquire`'s ``http`` and ``gdrive` download helpers
rather than growing a second downloader. It has no torch dependency, so
``imgforensics weights list`` and ``imgforensics weights fetch`` work on a
machine without the optional ``ml`` extra installed -- fetching the weights
and being able to run them are separate problems.

**Where files land**, in order: the ``dest`` argument, else
``$IMGFORENSICS_WEIGHTS_DIR/<name>/``, else ``weights/<name>/`` relative to
the working directory. That is the same environment variable the frozen
backbones use for their Hugging Face cache
(:data:`imgforensics.detectors.backbones.WEIGHTS_DIR_ENV`), so one setting
keeps every downloaded weight inside the project's gitignored ``weights/``
directory.

:func:`fetch_weights` is idempotent: a file already present with the expected
sha256 is left alone and reported as ``"cached"`` without touching the
network, so re-running the command in a script costs nothing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from imgforensics.data.acquire import (
    ChecksumMismatchError,
    LicenseNotAcceptedError,
    _gdown_download,
    _http_download,
    _sha256_of_file,
)
from imgforensics.detectors.backbones import WEIGHTS_DIR_ENV

__all__ = [
    "WEIGHTS",
    "FetchWeightsReport",
    "WeightSpec",
    "fetch_weights",
    "get_weight_spec",
    "resolve_weights_dir",
    "weights_file",
]

#: Fallback weights directory, relative to the working directory (gitignored).
DEFAULT_WEIGHTS_DIR = Path("weights")


class WeightSpec(BaseModel):
    """One downloadable set of pretrained weights.

    Attributes:
        name: Registry key, and the subdirectory the file is written to.
        method: ``"http"`` for a direct URL, ``"gdrive"`` for a Google Drive
            file id (downloaded through ``gdown``, the optional ``data``
            extra).
        url_or_gdrive_id: The URL (``http``) or the Drive file id
            (``gdrive``).
        filename: File name to write inside the weights directory.
        sha256: Expected digest of the downloaded file, verified after every
            download and re-checked on every later call. ``None`` disables
            verification, which :func:`fetch_weights` reports loudly.
        size_mb: Approximate download size, printed before the download so
            the license gate is also a disk-budget gate.
        license: License of the *weights*, printed before every download.
        commercial_ok: Whether that license permits commercial use, mirroring
            the dataset registry's flag.
        source: Where the file comes from, for ``THIRD_PARTY_NOTICES.md`` and
            for the printed gate.
        model: Human-readable model name.
    """

    name: str
    method: Literal["http", "gdrive"]
    url_or_gdrive_id: str
    filename: str
    sha256: str | None = None
    size_mb: float | None = None
    license: str = "unknown"
    commercial_ok: bool | None = None
    source: str = ""
    model: str = ""


#: The registered localizer weights, keyed by the name used on the CLI and in
#: the detector registry.
WEIGHTS: dict[str, WeightSpec] = {
    "iml_vit": WeightSpec(
        name="iml_vit",
        model="IML-ViT (CASIAv2-trained release)",
        method="gdrive",
        url_or_gdrive_id="1xXJGJPW1i5j9Pc1JKd7fJmIAQkvt9jY7",
        filename="iml-vit_checkpoint.pth",
        sha256="7631fe852e139d904ad96fd766a37120d703c966cb80dfad70ffae4ff426cce4",
        size_mb=350.2,
        license="MIT",
        commercial_ok=True,
        source=(
            "SunnyHaze/IML-ViT, checkpoints/ckpt_download_page.md -> Google Drive "
            "file 1xXJGJPW1i5j9Pc1JKd7fJmIAQkvt9jY7 (iml-vit_checkpoint.pth); "
            "downloaded on first use, never committed"
        ),
    ),
    "catnet_v2": WeightSpec(
        name="catnet_v2",
        model="CAT-Net v2 (CAT_full, RGB + DCT streams)",
        method="gdrive",
        url_or_gdrive_id="1tyOKVdx6UMys2OcNpUj9r6scxNIpcoLE",
        filename="CAT_full_v2.pth.tar",
        sha256="f82aaafdd1142775231feedcea0bb7027f7370561d9e8d107465454001865989",
        size_mb=873.1,
        license="CC-BY-4.0",
        commercial_ok=True,
        source=(
            "mjkwon2021/CAT-Net, README.md -> Google Drive folder "
            "14uNqj46505MQc3swBQgbaiPVAWtNChbz ('trained weight') -> "
            "CAT_full_v2.pth.tar. That folder entry is a Drive *shortcut* "
            "(id 1anexqI_JlkO41wx7MgkRLf34VnpIzPw), which gdown cannot "
            "follow -- it reports the file as owner-only -- so the id above "
            "is the shortcut's target, read from the redirect its /view page "
            "issues. The weights are CC-BY-4.0, so using them requires "
            "attributing CAT-Net; downloaded on first use, never committed"
        ),
    ),
}


def get_weight_spec(name: str) -> WeightSpec:
    """Return the :class:`WeightSpec` registered under ``name``.

    Raises:
        KeyError: if nothing is registered under ``name``, listing the known
            names in the error message.
    """
    try:
        return WEIGHTS[name]
    except KeyError as exc:
        known = ", ".join(sorted(WEIGHTS)) or "<none>"
        raise KeyError(f"No weights registered as {name!r}. Available: {known}") from exc


def resolve_weights_dir(dest: str | Path | None = None) -> Path:
    """The base directory weights are stored under: argument, env var, then default."""
    if dest is not None:
        return Path(dest)
    from_env = os.environ.get(WEIGHTS_DIR_ENV)
    if from_env:
        return Path(from_env).expanduser()
    return DEFAULT_WEIGHTS_DIR


def weights_file(name: str, dest: str | Path | None = None) -> Path:
    """Where ``name``'s weight file lives (whether or not it has been downloaded)."""
    spec = get_weight_spec(name)
    return resolve_weights_dir(dest) / spec.name / spec.filename


@dataclass
class FetchWeightsReport:
    """Outcome of one :func:`fetch_weights` call.

    ``status`` is ``"cached"`` when the file was already present and passed
    (or skipped) verification, ``"downloaded"`` when it was fetched now, and
    ``"failed"`` when a ``gdrive`` download returned nothing (gdown's own
    convention for a quota/permission failure, which is not an exception).
    """

    name: str
    path: Path
    status: Literal["cached", "downloaded", "failed"]
    bytes_downloaded: int = 0
    sha256: str | None = None


def _print_license(spec: WeightSpec) -> None:
    size = f"{spec.size_mb:.1f} MB" if spec.size_mb is not None else "unknown"
    print(f"Weights: {spec.name} -- {spec.model or spec.name}")
    print(f"License: {spec.license}")
    print(f"Commercial OK: {spec.commercial_ok}")
    print(f"Approx. size: {size}")
    print(f"Source: {spec.source or '-'}")


def fetch_weights(
    name: str,
    dest: str | Path | None = None,
    *,
    accept_license: bool = False,
    progress: bool = True,
) -> FetchWeightsReport:
    """Download ``name``'s pretrained weights, after printing and gating on the license.

    Always prints the weights' model name, license, ``commercial_ok`` flag,
    approximate size and source first. Refuses to download with
    :class:`~imgforensics.data.acquire.LicenseNotAcceptedError` unless
    ``accept_license`` is true. A file already on disk is verified against
    the spec's sha256 and, when it matches, returned untouched -- so calling
    this twice downloads once.

    Args:
        name: A key of :data:`WEIGHTS`.
        dest: Base directory to write ``<dest>/<name>/<filename>`` into.
            ``None`` falls back to ``$IMGFORENSICS_WEIGHTS_DIR`` and then
            ``weights/``.
        accept_license: Must be true to proceed past the license gate.
        progress: Print periodic byte-count lines during an ``http``
            download (``gdrive`` downloads print nothing either way).

    Returns:
        A :class:`FetchWeightsReport`.

    Raises:
        KeyError: ``name`` is not registered.
        LicenseNotAcceptedError: ``accept_license`` is false.
        ChecksumMismatchError: the file on disk, or the one just downloaded,
            does not match the spec's sha256. A freshly downloaded file is
            deleted before this is raised; an existing one is left in place
            with an error naming it, since deleting a file the caller may
            have put there deliberately is not this function's decision.
    """
    spec = get_weight_spec(name)
    _print_license(spec)

    target = resolve_weights_dir(dest) / spec.name / spec.filename

    if target.is_file():
        if spec.sha256 is None:
            print(f"[weights] {target} already present (no sha256 recorded to verify it against)")
            return FetchWeightsReport(name=name, path=target, status="cached")
        digest = _sha256_of_file(target)
        if digest == spec.sha256:
            print(f"[weights] {target} already present and verified")
            return FetchWeightsReport(name=name, path=target, status="cached", sha256=digest)
        raise ChecksumMismatchError(
            f"{target}: sha256 mismatch (expected {spec.sha256}, got {digest}); "
            "delete the file and re-run to download it again"
        )

    if not accept_license:
        raise LicenseNotAcceptedError(
            f"license not accepted for weights {name!r}: re-run with accept_license=True "
            "(CLI: --accept-license) after reading the license printed above."
        )

    target.parent.mkdir(parents=True, exist_ok=True)

    if spec.method == "http":
        written = _http_download(
            spec.url_or_gdrive_id, target, sha256=spec.sha256, progress=progress
        )
        digest = _sha256_of_file(target)
        return FetchWeightsReport(
            name=name,
            path=target,
            status="downloaded",
            bytes_downloaded=written,
            sha256=digest,
        )

    result = _gdown_download(file_id=spec.url_or_gdrive_id, output=target)
    if result is None or not target.is_file():
        print(
            f"[weights] the Google Drive download of {spec.filename} returned nothing. "
            "Drive rejects automated downloads once a file's daily quota is exhausted; "
            "download it by hand from "
            f"https://drive.google.com/file/d/{spec.url_or_gdrive_id}/view "
            f"and place it at {target}."
        )
        return FetchWeightsReport(name=name, path=target, status="failed")

    digest = _sha256_of_file(target)
    if spec.sha256 is not None and digest != spec.sha256:
        target.unlink(missing_ok=True)
        raise ChecksumMismatchError(
            f"{spec.filename}: sha256 mismatch (expected {spec.sha256}, got {digest}); file deleted"
        )
    return FetchWeightsReport(
        name=name,
        path=target,
        status="downloaded",
        bytes_downloaded=target.stat().st_size,
        sha256=digest,
    )
