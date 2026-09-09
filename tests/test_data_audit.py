"""Tests for imgforensics.data.audit: bias detection over a manifest."""

from __future__ import annotations

import pytest

from imgforensics.data.audit import AuditReport, BiasError, audit_manifest
from imgforensics.data.manifest import Manifest, ManifestEntry, ManifestMeta


def _entry(
    path: str,
    label: str,
    *,
    fmt: str = "JPEG",
    width: int = 800,
    height: int = 600,
    jpeg_quality: int | None = 85,
    sha256: str | None = None,
) -> ManifestEntry:
    return ManifestEntry(
        path=path,
        label=label,  # type: ignore[arg-type]
        source="unit-test",
        sha256=sha256 or ("0" * 63 + path[-1] if path else "0" * 64),
        width=width,
        height=height,
        format=fmt,
        jpeg_quality=jpeg_quality,
    )


def _manifest(entries: list[ManifestEntry]) -> Manifest:
    meta = ManifestMeta(dataset="unit-test", root="/root", created="2026-01-01")
    return Manifest(meta=meta, entries=entries)


def _distinct_hashes(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i:060d}" for i in range(n)]


def test_audit_balanced_set_has_no_problems() -> None:
    hashes = _distinct_hashes("r", 20) + _distinct_hashes("f", 20)
    entries = []
    for i in range(20):
        entries.append(
            _entry(
                f"real/{i}.jpg", "real", fmt="JPEG", jpeg_quality=80 + (i % 10), sha256=hashes[i]
            )
        )
    for i in range(20):
        entries.append(
            _entry(
                f"fake/{i}.jpg",
                "fake",
                fmt="JPEG",
                jpeg_quality=80 + (i % 10),
                sha256=hashes[20 + i],
            )
        )
    manifest = _manifest(entries)

    report = audit_manifest(manifest)

    assert isinstance(report, AuditReport)
    assert report.ok is True
    assert report.problems == []


def test_audit_detects_format_bias() -> None:
    entries = []
    for i in range(20):
        entries.append(_entry(f"real/{i}.jpg", "real", fmt="JPEG", sha256=f"r{i:063d}"))
    for i in range(20):
        entries.append(
            _entry(f"fake/{i}.png", "fake", fmt="PNG", jpeg_quality=None, sha256=f"f{i:063d}")
        )
    manifest = _manifest(entries)

    report = audit_manifest(manifest)

    assert report.ok is False
    assert any("Format distribution" in p for p in report.problems)
    assert report.distributions["format"]["real"] == {"JPEG": 20}
    assert report.distributions["format"]["fake"] == {"PNG": 20}


def test_audit_strict_raises_bias_error() -> None:
    entries = [_entry(f"real/{i}.jpg", "real", fmt="JPEG", sha256=f"r{i:063d}") for i in range(20)]
    entries += [
        _entry(f"fake/{i}.png", "fake", fmt="PNG", jpeg_quality=None, sha256=f"f{i:063d}")
        for i in range(20)
    ]
    manifest = _manifest(entries)

    with pytest.raises(BiasError) as exc_info:
        audit_manifest(manifest, strict=True)
    assert exc_info.value.problems
    assert any("Format distribution" in p for p in exc_info.value.problems)


def test_audit_no_problems_does_not_raise_even_in_strict_mode() -> None:
    hashes = _distinct_hashes("r", 20) + _distinct_hashes("f", 20)
    entries = [
        _entry(f"real/{i}.jpg", "real", jpeg_quality=80 + (i % 10), sha256=hashes[i])
        for i in range(20)
    ]
    entries += [
        _entry(f"fake/{i}.jpg", "fake", jpeg_quality=80 + (i % 10), sha256=hashes[20 + i])
        for i in range(20)
    ]
    manifest = _manifest(entries)

    report = audit_manifest(manifest, strict=True)
    assert report.ok is True


def test_audit_detects_duplicates_within_a_label() -> None:
    entries = [
        _entry("real/a.jpg", "real", sha256="d" * 64),
        _entry("real/b.jpg", "real", sha256="d" * 64),
        _entry("real/c.jpg", "real", sha256="e" * 64),
        _entry("fake/a.jpg", "fake", sha256="f" * 64),
        _entry("fake/b.jpg", "fake", sha256="g" * 64),
        _entry("fake/c.jpg", "fake", sha256="h" * 64),
    ]
    manifest = _manifest(entries)

    report = audit_manifest(manifest)

    assert any("duplicate image" in p and "'real'" in p for p in report.problems)


def test_audit_detects_duplicates_across_labels() -> None:
    shared_hash = "a" * 64
    entries = [
        _entry("real/a.jpg", "real", sha256=shared_hash),
        _entry("real/b.jpg", "real", sha256="b" * 64),
        _entry("real/c.jpg", "real", sha256="c" * 64),
        _entry("fake/a.jpg", "fake", sha256=shared_hash),
        _entry("fake/b.jpg", "fake", sha256="e" * 64),
        _entry("fake/c.jpg", "fake", sha256="f" * 64),
    ]
    manifest = _manifest(entries)

    report = audit_manifest(manifest)

    assert any("same sha256 under both" in p or "labelled both ways" in p for p in report.problems)


def test_audit_detects_class_imbalance() -> None:
    entries = [_entry(f"real/{i}.jpg", "real", sha256=f"r{i:063d}") for i in range(30)]
    entries += [_entry(f"fake/{i}.jpg", "fake", sha256=f"f{i:063d}") for i in range(3)]
    manifest = _manifest(entries)

    report = audit_manifest(manifest)

    assert any("imbalance" in p.lower() for p in report.problems)


def test_audit_report_to_markdown_contains_problems_and_tables() -> None:
    entries = [_entry(f"real/{i}.jpg", "real", fmt="JPEG", sha256=f"r{i:063d}") for i in range(20)]
    entries += [
        _entry(f"fake/{i}.png", "fake", fmt="PNG", jpeg_quality=None, sha256=f"f{i:063d}")
        for i in range(20)
    ]
    manifest = _manifest(entries)

    report = audit_manifest(manifest)
    markdown = report.to_markdown()

    assert "# Bias audit report" in markdown
    assert "problem(s) found" in markdown
    assert "## format" in markdown
    assert "JPEG" in markdown and "PNG" in markdown
