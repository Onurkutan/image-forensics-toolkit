"""Tests for imgforensics.data.manifest: build/save/load round trip."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from imgforensics.data.manifest import (
    Manifest,
    ManifestEntry,
    ManifestMeta,
    build_manifest,
    label_from_parent_folder,
)


def _make_tree(root: Path) -> None:
    (root / "real").mkdir(parents=True)
    (root / "fake").mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (32 + i, 24)).save(root / "real" / f"r{i}.jpg", format="JPEG", quality=85)
    for i in range(2):
        Image.new("RGB", (40, 30)).save(root / "fake" / f"f{i}.png", format="PNG")


@pytest.mark.parametrize(
    ("folder", "expected"),
    [
        ("real", "real"),
        ("Real", "real"),
        ("fake", "fake"),
        ("FAKE", "fake"),
        ("0_real", "real"),
        ("1_fake", "fake"),
        ("authentic", "real"),
        ("tampered", "fake"),
        ("nature", "real"),
        ("ai", "fake"),
        ("something_else", None),
    ],
)
def test_label_from_parent_folder(folder: str, expected: str | None) -> None:
    assert label_from_parent_folder(Path(f"/dataset/{folder}/img.jpg")) == expected


def test_build_manifest_walks_tree_and_extracts_fields(tmp_path: Path) -> None:
    _make_tree(tmp_path)

    manifest, skipped = build_manifest(
        tmp_path,
        dataset="unit-test",
        label_of=label_from_parent_folder,
        license="MIT",
        commercial_ok=True,
        progress=False,
    )

    assert skipped == []
    assert len(manifest.entries) == 5
    assert manifest.meta.dataset == "unit-test"
    assert manifest.meta.license == "MIT"
    assert manifest.meta.commercial_ok is True

    by_path = {entry.path: entry for entry in manifest.entries}
    assert by_path["real/r0.jpg"].label == "real"
    assert by_path["real/r0.jpg"].source == "unit-test"
    assert by_path["real/r0.jpg"].format == "JPEG"
    assert by_path["real/r0.jpg"].jpeg_quality is not None
    assert by_path["real/r0.jpg"].width == 32
    assert by_path["real/r0.jpg"].height == 24
    assert len(by_path["real/r0.jpg"].sha256) == 64

    assert by_path["fake/f0.png"].label == "fake"
    assert by_path["fake/f0.png"].format == "PNG"
    assert by_path["fake/f0.png"].jpeg_quality is None

    # Paths are always POSIX-style, even on Windows.
    assert all("\\" not in entry.path for entry in manifest.entries)


def test_build_manifest_skips_files_with_no_label(tmp_path: Path) -> None:
    (tmp_path / "unlabeled").mkdir()
    Image.new("RGB", (10, 10)).save(tmp_path / "unlabeled" / "x.png", format="PNG")
    (tmp_path / "real").mkdir()
    Image.new("RGB", (10, 10)).save(tmp_path / "real" / "y.png", format="PNG")

    manifest, skipped = build_manifest(
        tmp_path, dataset="ds", label_of=label_from_parent_folder, progress=False
    )

    assert skipped == []
    assert len(manifest.entries) == 1
    assert manifest.entries[0].path == "real/y.png"


def test_build_manifest_records_unreadable_files_as_skipped(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "broken.jpg").write_bytes(b"not an image")
    Image.new("RGB", (10, 10)).save(tmp_path / "real" / "ok.png", format="PNG")

    manifest, skipped = build_manifest(
        tmp_path, dataset="ds", label_of=label_from_parent_folder, progress=False
    )

    assert len(manifest.entries) == 1
    assert len(skipped) == 1
    assert "broken.jpg" in skipped[0]


def test_build_manifest_uses_generator_split_mask_callables(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "masks").mkdir()
    image_path = tmp_path / "real" / "a.png"
    Image.new("RGB", (10, 10)).save(image_path, format="PNG")
    mask_path = tmp_path / "masks" / "a_mask.png"
    Image.new("L", (10, 10)).save(mask_path, format="PNG")

    manifest, _ = build_manifest(
        tmp_path,
        dataset="ds",
        label_of=label_from_parent_folder,
        generator_of=lambda p: "sdxl",
        split_of=lambda p: "train",
        mask_of=lambda p: mask_path,
        progress=False,
    )

    entry = manifest.entries[0]
    assert entry.generator == "sdxl"
    assert entry.split == "train"
    assert entry.mask_path == "masks/a_mask.png"


def test_manifest_save_load_round_trip(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    manifest, _ = build_manifest(
        tmp_path,
        dataset="roundtrip",
        label_of=label_from_parent_folder,
        license="CC-BY-4.0",
        commercial_ok=False,
        progress=False,
    )

    out_path = tmp_path / "out" / "manifest.jsonl"
    manifest.save(out_path)

    assert out_path.exists()
    meta_path = tmp_path / "out" / "manifest.meta.json"
    assert meta_path.exists()

    loaded = Manifest.load(out_path)
    assert loaded.meta == manifest.meta
    assert len(loaded.entries) == len(manifest.entries)
    assert [e.path for e in loaded.entries] == sorted(e.path for e in manifest.entries)
    original_by_path = {e.path: e for e in manifest.entries}
    for entry in loaded.entries:
        assert entry == original_by_path[entry.path]


def test_manifest_save_is_sorted_by_path(tmp_path: Path) -> None:
    meta = ManifestMeta(dataset="d", root="/root", created="2026-01-01")
    entries = [
        ManifestEntry(
            path="zzz.jpg",
            label="real",
            source="d",
            sha256="0" * 64,
            width=1,
            height=1,
            format="JPEG",
        ),
        ManifestEntry(
            path="aaa.jpg",
            label="fake",
            source="d",
            sha256="1" * 64,
            width=1,
            height=1,
            format="JPEG",
        ),
    ]
    manifest = Manifest(meta=meta, entries=entries)
    out_path = tmp_path / "manifest.jsonl"
    manifest.save(out_path)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert '"path":"aaa.jpg"' in lines[0]
    assert '"path":"zzz.jpg"' in lines[1]


def test_manifest_summary(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    manifest, _ = build_manifest(
        tmp_path, dataset="ds", label_of=label_from_parent_folder, progress=False
    )
    summary = manifest.summary()
    assert summary["label"] == {"real": 3, "fake": 2}
    assert summary["source"] == {"ds": 5}
    assert summary["generator"] == {"none": 5}
    assert summary["split"] == {"none": 5}
    assert summary["format"] == {"JPEG": 3, "PNG": 2}
