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
    merge,
    sample,
    split_by_group,
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


# --- sample() ----------------------------------------------------------------


def _entry(
    path: str, label: str, *, generator: str | None = None, sha: str | None = None
) -> ManifestEntry:
    return ManifestEntry(
        path=path,
        label=label,  # type: ignore[arg-type]
        source="ds",
        generator=generator,
        sha256=sha or ("a" * 63 + "0"),
        width=10,
        height=10,
        format="JPEG",
    )


def _stratified_manifest() -> Manifest:
    entries = [_entry(f"real/{i}.jpg", "real", sha=f"{i:064d}") for i in range(60)]
    entries += [
        _entry(f"fake/sdxl/{i}.jpg", "fake", generator="sdxl", sha=f"{1000 + i:064d}")
        for i in range(20)
    ]
    entries += [
        _entry(f"fake/dalle3/{i}.jpg", "fake", generator="dalle3", sha=f"{2000 + i:064d}")
        for i in range(20)
    ]
    meta = ManifestMeta(dataset="ds", root="/root", created="2026-01-01")
    return Manifest(meta=meta, entries=entries)


def test_sample_is_deterministic_for_the_same_seed() -> None:
    manifest = _stratified_manifest()
    first = sample(manifest, 20, seed=0)
    second = sample(manifest, 20, seed=0)
    assert [e.path for e in first.entries] == [e.path for e in second.entries]


def test_sample_differs_across_seeds() -> None:
    manifest = _stratified_manifest()
    first = sample(manifest, 20, seed=0)
    second = sample(manifest, 20, seed=1)
    assert [e.path for e in first.entries] != [e.path for e in second.entries]


def test_sample_is_proportionally_stratified() -> None:
    manifest = _stratified_manifest()
    sampled = sample(manifest, 20, seed=0, stratify_by=("label", "generator"))
    counts: dict[tuple[str, str], int] = {}
    for entry in sampled.entries:
        key = (entry.label, entry.generator or "none")
        counts[key] = counts.get(key, 0) + 1
    # 60/20/20 split of 100 -> proportional shares of 20 are 12/4/4.
    assert counts == {("real", "none"): 12, ("fake", "sdxl"): 4, ("fake", "dalle3"): 4}


def test_sample_never_exceeds_available_entries() -> None:
    manifest = _stratified_manifest()
    sampled = sample(manifest, 10_000, seed=0)
    assert len(sampled.entries) == len(manifest.entries)


def test_sample_never_exceeds_a_strata_available_count() -> None:
    entries = [_entry("real/0.jpg", "real", sha="0" * 64)]
    entries += [_entry(f"fake/{i}.jpg", "fake", sha=f"{i + 1:064d}") for i in range(99)]
    meta = ManifestMeta(dataset="ds", root="/root", created="2026-01-01")
    manifest = Manifest(meta=meta, entries=entries)

    sampled = sample(manifest, 50, seed=0, stratify_by=("label",))
    by_label: dict[str, int] = {}
    for entry in sampled.entries:
        by_label[entry.label] = by_label.get(entry.label, 0) + 1
    assert by_label.get("real", 0) <= 1
    assert len(sampled.entries) == 50


def test_sample_meta_records_a_note() -> None:
    manifest = _stratified_manifest()
    sampled = sample(manifest, 20, seed=7)
    assert sampled.meta.notes == "sampled 20 of 100 with seed 7"
    # Everything else about meta is preserved.
    assert sampled.meta.dataset == manifest.meta.dataset
    assert sampled.meta.root == manifest.meta.root


def test_sample_appends_to_existing_notes() -> None:
    manifest = _stratified_manifest()
    manifest.meta.notes = "original note"
    sampled = sample(manifest, 5, seed=0)
    assert sampled.meta.notes == "original note; sampled 5 of 100 with seed 0"


# --- merge() -------------------------------------------------------------


def test_merge_requires_at_least_one_manifest() -> None:
    with pytest.raises(ValueError, match="at least one"):
        merge([])


def test_merge_same_root_keeps_relative_paths() -> None:
    meta_a = ManifestMeta(
        dataset="a", root="/data/root", created="2026-01-01", license="MIT", commercial_ok=True
    )
    meta_b = ManifestMeta(
        dataset="b", root="/data/root", created="2026-01-01", license="MIT", commercial_ok=True
    )
    manifest_a = Manifest(meta=meta_a, entries=[_entry("x/1.jpg", "real", sha="1" * 64)])
    manifest_b = Manifest(meta=meta_b, entries=[_entry("y/2.jpg", "fake", sha="2" * 64)])

    merged = merge([manifest_a, manifest_b])

    assert len(merged.entries) == 2
    assert {e.path for e in merged.entries} == {"x/1.jpg", "y/2.jpg"}
    assert merged.meta.root == "/data/root"
    assert merged.meta.dataset == "a+b"
    assert merged.meta.license == "MIT"
    assert merged.meta.commercial_ok is True
    assert merged.meta.notes is None


def test_merge_different_roots_stores_absolute_paths_and_notes() -> None:
    meta_a = ManifestMeta(dataset="a", root="/data/root_a", created="2026-01-01")
    meta_b = ManifestMeta(dataset="b", root="/data/root_b", created="2026-01-01")
    manifest_a = Manifest(meta=meta_a, entries=[_entry("x/1.jpg", "real", sha="1" * 64)])
    manifest_b = Manifest(
        meta=meta_b,
        entries=[
            ManifestEntry(
                path="y/2.jpg",
                label="fake",
                source="b",
                sha256="2" * 64,
                width=10,
                height=10,
                format="JPEG",
                mask_path="masks/2.png",
            )
        ],
    )

    merged = merge([manifest_a, manifest_b])

    paths = {e.path for e in merged.entries}
    assert paths == {"/data/root_a/x/1.jpg", "/data/root_b/y/2.jpg"}
    mask_entry = next(e for e in merged.entries if e.path == "/data/root_b/y/2.jpg")
    assert mask_entry.mask_path == "/data/root_b/masks/2.png"
    assert "different roots" in merged.meta.notes


def test_merge_commercial_ok_is_false_if_any_input_is_false() -> None:
    meta_a = ManifestMeta(dataset="a", root="/r", created="2026-01-01", commercial_ok=True)
    meta_b = ManifestMeta(dataset="b", root="/r", created="2026-01-01", commercial_ok=False)
    manifest_a = Manifest(meta=meta_a, entries=[])
    manifest_b = Manifest(meta=meta_b, entries=[])
    merged = merge([manifest_a, manifest_b])
    assert merged.meta.commercial_ok is False


def test_merge_commercial_ok_is_true_only_if_all_true() -> None:
    meta_a = ManifestMeta(dataset="a", root="/r", created="2026-01-01", commercial_ok=True)
    meta_b = ManifestMeta(dataset="b", root="/r", created="2026-01-01", commercial_ok=True)
    merged = merge([Manifest(meta=meta_a, entries=[]), Manifest(meta=meta_b, entries=[])])
    assert merged.meta.commercial_ok is True


def test_merge_commercial_ok_is_none_when_mixed_true_and_unknown() -> None:
    meta_a = ManifestMeta(dataset="a", root="/r", created="2026-01-01", commercial_ok=True)
    meta_b = ManifestMeta(dataset="b", root="/r", created="2026-01-01", commercial_ok=None)
    merged = merge([Manifest(meta=meta_a, entries=[]), Manifest(meta=meta_b, entries=[])])
    assert merged.meta.commercial_ok is None


def test_merge_disagreeing_licenses_are_left_unset_with_a_note() -> None:
    meta_a = ManifestMeta(dataset="a", root="/r", created="2026-01-01", license="MIT")
    meta_b = ManifestMeta(dataset="b", root="/r", created="2026-01-01", license="CC-BY-4.0")
    merged = merge([Manifest(meta=meta_a, entries=[]), Manifest(meta=meta_b, entries=[])])
    assert merged.meta.license is None
    assert "disagreed on license" in merged.meta.notes


def test_merge_result_round_trips_through_save_load(tmp_path: Path) -> None:
    meta_a = ManifestMeta(dataset="a", root="/r", created="2026-01-01")
    meta_b = ManifestMeta(dataset="b", root="/r", created="2026-01-01")
    manifest_a = Manifest(meta=meta_a, entries=[_entry("x/1.jpg", "real", sha="1" * 64)])
    manifest_b = Manifest(meta=meta_b, entries=[_entry("y/2.jpg", "fake", sha="2" * 64)])
    merged = merge([manifest_a, manifest_b])

    out_path = tmp_path / "merged.jsonl"
    merged.save(out_path)
    loaded = Manifest.load(out_path)
    assert len(loaded.entries) == 2


# --- split_by_group() --------------------------------------------------------


def _grouped_manifest() -> Manifest:
    entries: list[ManifestEntry] = []
    counter = 0
    for generator, count in {"A": 10, "B": 8, "C": 6, "D": 4}.items():
        for i in range(count):
            entries.append(
                _entry(
                    f"fake/{generator}/{i}.jpg",
                    "fake",
                    generator=generator,
                    sha=f"{counter:064d}",
                )
            )
            counter += 1
    for i in range(20):
        entries.append(_entry(f"real/{i}.jpg", "real", sha=f"{counter:064d}"))
        counter += 1
    meta = ManifestMeta(dataset="ds", root="/root", created="2026-01-01")
    return Manifest(meta=meta, entries=entries)


def test_split_by_group_is_generator_disjoint() -> None:
    manifest = _grouped_manifest()
    train, val = split_by_group(manifest, seed=0)

    train_generators = {e.generator for e in train.entries if e.generator is not None}
    val_generators = {e.generator for e in val.entries if e.generator is not None}
    assert train_generators & val_generators == set()
    assert train_generators | val_generators == {"A", "B", "C", "D"}


def test_split_by_group_keeps_reals_in_both_halves() -> None:
    manifest = _grouped_manifest()
    train, val = split_by_group(manifest, seed=0)

    train_reals = [e for e in train.entries if e.generator is None]
    val_reals = [e for e in val.entries if e.generator is None]
    assert train_reals
    assert val_reals
    assert len(train_reals) + len(val_reals) == 20


def test_split_by_group_sets_split_field_on_every_entry() -> None:
    manifest = _grouped_manifest()
    train, val = split_by_group(manifest, seed=0)

    assert train.entries and val.entries
    assert all(e.split == "train" for e in train.entries)
    assert all(e.split == "val" for e in val.entries)


def test_split_by_group_notes_describe_the_split() -> None:
    manifest = _grouped_manifest()
    train, val = split_by_group(manifest, seed=0)

    assert train.meta.notes is not None
    assert "split_by_group" in train.meta.notes
    assert train.meta.notes == val.meta.notes


def test_split_by_group_holdout_forces_named_groups_into_val() -> None:
    manifest = _grouped_manifest()
    train, val = split_by_group(manifest, holdout=["B"])

    val_generators = {e.generator for e in val.entries if e.generator is not None}
    train_generators = {e.generator for e in train.entries if e.generator is not None}
    assert val_generators == {"B"}
    assert train_generators == {"A", "C", "D"}


def test_split_by_group_is_deterministic_for_the_same_seed() -> None:
    manifest = _grouped_manifest()
    train1, val1 = split_by_group(manifest, seed=3)
    train2, val2 = split_by_group(manifest, seed=3)

    assert [e.path for e in train1.entries] == [e.path for e in train2.entries]
    assert [e.path for e in val1.entries] == [e.path for e in val2.entries]


def test_split_by_group_ungrouped_split_differs_across_seeds() -> None:
    manifest = _grouped_manifest()
    _, val1 = split_by_group(manifest, seed=0)
    _, val2 = split_by_group(manifest, seed=1)

    val_reals_1 = {e.path for e in val1.entries if e.generator is None}
    val_reals_2 = {e.path for e in val2.entries if e.generator is None}
    assert val_reals_1 != val_reals_2


def test_split_by_group_never_splits_a_group_across_both_halves() -> None:
    manifest = _grouped_manifest()
    train, val = split_by_group(manifest, val_fraction=0.35, seed=0)

    train_paths_by_generator: dict[str, set[str]] = {}
    for entry in train.entries:
        if entry.generator is not None:
            train_paths_by_generator.setdefault(entry.generator, set()).add(entry.path)
    val_generators = {e.generator for e in val.entries if e.generator is not None}
    assert set(train_paths_by_generator) & val_generators == set()
