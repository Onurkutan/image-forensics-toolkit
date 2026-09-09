"""Tests for imgforensics.data.layouts: Layout adapters and prepare()."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from imgforensics.data.audit import AuditReport
from imgforensics.data.layouts import (
    Layout,
    get_layout,
    load_layouts,
    materialize_parquet,
    prepare,
)
from imgforensics.data.manifest import Manifest
from imgforensics.data.registry import load_registry


def _img(path: Path, size: tuple[int, int] = (16, 12)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(path, format="PNG")


# --- packaged layouts.yaml validation ---------------------------------------


def test_every_layout_has_non_empty_globs() -> None:
    for layout in load_layouts():
        assert layout.real_globs or layout.fake_globs, f"{layout.dataset!r} has no globs at all"


def test_every_layout_dataset_is_in_the_registry_or_is_documented_derived() -> None:
    registry_names = {entry.name for entry in load_registry()}
    for layout in load_layouts():
        assert layout.dataset in registry_names, (
            f"{layout.dataset!r} has no matching registry entry"
        )


def test_get_layout_unknown_dataset_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_layout("does-not-exist")


def test_cocoglide_layout_intentionally_absent() -> None:
    # CocoGlide's acquisition recipe has a verified download URL, but its
    # extracted zip's internal folder layout was never confirmed -- see
    # acquire.yaml's subset_note for CocoGlide. Guessing a layout would be
    # worse than omitting it, so it must not appear here.
    names = {layout.dataset for layout in load_layouts()}
    assert "CocoGlide" not in names


# --- synthetic Synthbuster-style tree (per-generator folders) ---------------


def _make_synthbuster_like_tree(root: Path) -> None:
    for generator in ("dalle3", "sdxl", "firefly"):
        for i in range(2):
            _img(root / generator / f"img{i}.png")


def test_layout_adapter_on_synthbuster_style_tree(tmp_path: Path) -> None:
    _make_synthbuster_like_tree(tmp_path)
    manifest, skipped, report = prepare(
        "Synthbuster", tmp_path, tmp_path / "out" / "manifest.jsonl"
    )

    assert skipped == []
    assert len(manifest.entries) == 6
    assert {entry.label for entry in manifest.entries} == {"fake"}
    generators = {entry.generator for entry in manifest.entries}
    assert generators == {"dalle3", "sdxl", "firefly"}
    assert isinstance(report, AuditReport)
    # Fake-only manifest: audit should flag it as single-label.
    assert not report.ok
    assert any("Only one label" in p for p in report.problems)


# --- synthetic CASIA-style tree (Au/Tp + Gt masks) --------------------------


def _make_casia_like_tree(root: Path) -> None:
    for i in range(3):
        _img(root / "Au" / f"Au_{i}.jpg")
    for i in range(3):
        _img(root / "Tp" / f"Tp_{i}.jpg")
        _img(root / "Gt" / f"Tp_{i}_gt.png")
    # One tampered image with no matching mask, to exercise mask_path=None.
    _img(root / "Tp" / "Tp_orphan.jpg")


def test_layout_adapter_on_casia_style_tree_resolves_masks(tmp_path: Path) -> None:
    _make_casia_like_tree(tmp_path)
    manifest, skipped, report = prepare("CASIA v2.0", tmp_path, tmp_path / "out" / "manifest.jsonl")

    assert skipped == []
    by_path = {entry.path: entry for entry in manifest.entries}
    assert len(manifest.entries) == 7  # 3 Au + 4 Tp; Gt/* excluded from both globs

    assert by_path["Au/Au_0.jpg"].label == "real"
    assert by_path["Au/Au_0.jpg"].mask_path is None

    tampered = by_path["Tp/Tp_0.jpg"]
    assert tampered.label == "fake"
    assert tampered.mask_path == "Gt/Tp_0_gt.png"

    orphan = by_path["Tp/Tp_orphan.jpg"]
    assert orphan.label == "fake"
    assert orphan.mask_path is None

    assert isinstance(report, AuditReport)


def test_prepare_saves_manifest_to_out_path(tmp_path: Path) -> None:
    _make_casia_like_tree(tmp_path)
    out_path = tmp_path / "out" / "manifest.jsonl"
    manifest, _, _ = prepare("CASIA v2.0", tmp_path, out_path)

    assert out_path.exists()
    loaded = Manifest.load(out_path)
    assert len(loaded.entries) == len(manifest.entries)


def test_prepare_uses_registry_license_and_commercial_ok_as_defaults(tmp_path: Path) -> None:
    _make_casia_like_tree(tmp_path)
    manifest, _, _ = prepare("CASIA v2.0", tmp_path, tmp_path / "out" / "manifest.jsonl")

    registry_entry = next(entry for entry in load_registry() if entry.name == "CASIA v2.0")
    assert manifest.meta.license == registry_entry.license
    assert manifest.meta.commercial_ok == registry_entry.commercial_ok


def test_prepare_explicit_license_overrides_registry_default(tmp_path: Path) -> None:
    _make_casia_like_tree(tmp_path)
    manifest, _, _ = prepare(
        "CASIA v2.0",
        tmp_path,
        tmp_path / "out" / "manifest.jsonl",
        license="Custom",
        commercial_ok=True,
    )
    assert manifest.meta.license == "Custom"
    assert manifest.meta.commercial_ok is True


# --- fallback to label_from_parent_folder for an unregistered dataset ------


def test_prepare_falls_back_to_label_from_parent_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "fake").mkdir()
    _img(tmp_path / "real" / "r0.png")
    _img(tmp_path / "fake" / "f0.png")

    manifest, skipped, report = prepare(
        "SomeUnregisteredDataset", tmp_path, tmp_path / "out" / "manifest.jsonl"
    )

    assert skipped == []
    assert len(manifest.entries) == 2
    by_label = {entry.label for entry in manifest.entries}
    assert by_label == {"real", "fake"}
    out = capsys.readouterr().out
    assert "no layout registered" in out
    assert isinstance(report, AuditReport)


# --- COCO layout: all-real, split from top-level folder name --------------


def test_coco_layout_labels_everything_real_and_maps_split(tmp_path: Path) -> None:
    _img(tmp_path / "val2017" / "000001.jpg")
    _img(tmp_path / "val2017" / "000002.jpg")
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "instances_val2017.json").write_text("{}")

    manifest, skipped, _ = prepare("COCO", tmp_path, tmp_path / "out" / "manifest.jsonl")

    assert skipped == []
    assert len(manifest.entries) == 2
    assert all(entry.label == "real" for entry in manifest.entries)
    assert all(entry.split == "val" for entry in manifest.entries)


# --- Layout matching internals: exclude_globs and mask_template edge cases -


def test_layout_exclude_globs_skip_matching_files(tmp_path: Path) -> None:
    layout = Layout(
        dataset="unit-test",
        real_globs=["real/**"],
        fake_globs=["fake/**"],
        exclude_globs=["**/ignore_me/**"],
    )
    root = tmp_path
    _img(root / "real" / "a.png")
    _img(root / "real" / "ignore_me" / "b.png")

    from imgforensics.data.layouts import _callables_from_layout

    label_of, *_ = _callables_from_layout(layout, root)
    assert label_of(root / "real" / "a.png") == "real"
    assert label_of(root / "real" / "ignore_me" / "b.png") is None


def test_layout_mask_template_missing_file_returns_none(tmp_path: Path) -> None:
    layout = Layout(
        dataset="unit-test",
        real_globs=["real/**"],
        fake_globs=["fake/**"],
        mask_template="masks/{stem}.png",
    )
    root = tmp_path
    _img(root / "fake" / "a.png")

    from imgforensics.data.layouts import _callables_from_layout

    *_rest, mask_of = _callables_from_layout(layout, root)
    assert mask_of(root / "fake" / "a.png") is None

    (root / "masks").mkdir()
    _img(root / "masks" / "a.png")
    assert mask_of(root / "fake" / "a.png") == (root / "masks" / "a.png").resolve()


def test_layout_generator_from_regex(tmp_path: Path) -> None:
    layout = Layout(
        dataset="unit-test",
        fake_globs=["**/*.png"],
        generator_from="regex",
        generator_regex=r"^(?P<gen>[^/]+)/",
    )
    root = tmp_path
    _img(root / "sdxl" / "training" / "a.png")

    from imgforensics.data.layouts import _callables_from_layout

    _, generator_of, _, _ = _callables_from_layout(layout, root)
    assert generator_of(root / "sdxl" / "training" / "a.png") == "sdxl"


def test_layout_split_from_regex_with_split_map(tmp_path: Path) -> None:
    layout = Layout(
        dataset="unit-test",
        fake_globs=["**/*.png"],
        split_from="regex",
        split_regex=r"/(?P<split>training|validation|testing)/",
        split_map={"training": "train", "validation": "val", "testing": "test"},
    )
    root = tmp_path
    _img(root / "sdxl" / "training" / "a.png")

    from imgforensics.data.layouts import _callables_from_layout

    _, _, split_of, _ = _callables_from_layout(layout, root)
    assert split_of(root / "sdxl" / "training" / "a.png") == "train"


# --- materialize_parquet: graceful degradation without pyarrow -------------


def test_materialize_parquet_without_pyarrow_prints_instructions_and_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "pyarrow.parquet" or name.startswith("pyarrow"):
            raise ImportError("no pyarrow in this test environment")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    report = materialize_parquet(tmp_path, tmp_path / "out")

    assert report.rows_read == 0
    assert report.written == 0
    out = capsys.readouterr().out
    assert "pyarrow" in out


# --- materialize_parquet: real behavior on a synthetic shard ----------------


def _make_image_bytes(fmt: str, size: tuple[int, int] = (8, 6)) -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", size).save(buffer, format=fmt)
    return buffer.getvalue()


def _write_synthetic_shard(path: Path) -> None:
    """Six rows: 3 real, 3 fake, two generators, one nsfw row, PNG and JPEG bytes."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    png_bytes = _make_image_bytes("PNG")
    jpeg_bytes = _make_image_bytes("JPEG")

    rows = [
        {
            "image_name": "real0",
            "format": "PNG",
            "resolution": "8x6",
            "mode": "RGB",
            "image_data": png_bytes,
            "model_name": None,
            "nsfw_flag": False,
            "prompt": None,
            "real_source": "LAION",
            "subset": "systematic",
            "split": "train",
            "label": 0,
            "architecture": None,
        },
        {
            "image_name": "real1",
            "format": "JPEG",
            "resolution": "8x6",
            "mode": "RGB",
            "image_data": jpeg_bytes,
            "model_name": None,
            "nsfw_flag": False,
            "prompt": None,
            "real_source": "ImageNet",
            "subset": "systematic",
            "split": "val",
            "label": 0,
            "architecture": None,
        },
        {
            "image_name": "real2",
            "format": "PNG",
            "resolution": "8x6",
            "mode": "RGB",
            "image_data": png_bytes,
            "model_name": None,
            "nsfw_flag": False,
            "prompt": None,
            "real_source": None,
            "subset": "systematic",
            "split": "train",
            "label": 0,
            "architecture": None,
        },
        {
            "image_name": "fake0",
            "format": "PNG",
            "resolution": "8x6",
            "mode": "RGB",
            "image_data": png_bytes,
            "model_name": "sdxl",
            "nsfw_flag": False,
            "prompt": "a photo of a cat" * 20,  # exercises the 200-char truncation
            "real_source": None,
            "subset": "manual",
            "split": "train",
            "label": 1,
            "architecture": "diffusion",
        },
        {
            "image_name": "fake1",
            "format": "JPEG",
            "resolution": "8x6",
            "mode": "RGB",
            "image_data": jpeg_bytes,
            "model_name": "dalle3",
            "nsfw_flag": False,
            "prompt": "a dog",
            "real_source": None,
            "subset": "manual",
            "split": "val",
            "label": 1,
            "architecture": "diffusion",
        },
        {
            "image_name": "fake2_nsfw",
            "format": "PNG",
            "resolution": "8x6",
            "mode": "RGB",
            "image_data": png_bytes,
            "model_name": "sdxl",
            "nsfw_flag": True,
            "prompt": "nsfw content",
            "real_source": None,
            "subset": "manual",
            "split": "train",
            "label": 1,
            "architecture": "diffusion",
        },
    ]
    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_materialize_parquet_writes_tree_and_summary(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    src_dir = tmp_path / "src"
    out_dir = tmp_path / "out"
    _write_synthetic_shard(src_dir / "shard-00000.parquet")

    report = materialize_parquet(src_dir, out_dir, progress=False)

    assert report.rows_read == 6
    assert report.written == 5  # 6 rows minus 1 nsfw
    assert report.skipped == {"nsfw": 1}
    assert report.by_label == {"real": 3, "fake": 2}
    assert report.by_generator == {"LAION": 1, "ImageNet": 1, "real": 1, "sdxl": 1, "dalle3": 1}
    assert report.formats == {"PNG": 3, "JPEG": 2}  # the nsfw row's PNG is never counted

    assert (out_dir / "real" / "LAION" / "train" / "real0.png").is_file()
    assert (out_dir / "real" / "ImageNet" / "val" / "real1.jpg").is_file()
    assert (out_dir / "real" / "real" / "train" / "real2.png").is_file()
    assert (out_dir / "fake" / "sdxl" / "train" / "fake0.png").is_file()
    assert (out_dir / "fake" / "dalle3" / "val" / "fake1.jpg").is_file()
    # nsfw row was never written
    assert not (out_dir / "fake" / "sdxl" / "train" / "fake2_nsfw.png").exists()

    summary = json.loads((out_dir / "materialize.json").read_text(encoding="utf-8"))
    assert summary["rows_read"] == 6
    assert summary["written"] == 5

    attribute_lines = (
        (out_dir / "attributes.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(attribute_lines) == 5
    by_path = {json.loads(line)["path"]: json.loads(line) for line in attribute_lines}
    fake0 = by_path["fake/sdxl/train/fake0.png"]
    assert fake0["model_name"] == "sdxl"
    assert fake0["architecture"] == "diffusion"
    assert fake0["subset"] == "manual"
    assert fake0["split"] == "train"
    assert len(fake0["prompt"]) == 200  # truncated from the 320-char repeated prompt


def test_materialize_parquet_is_idempotent(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    src_dir = tmp_path / "src"
    out_dir = tmp_path / "out"
    _write_synthetic_shard(src_dir / "shard-00000.parquet")

    first = materialize_parquet(src_dir, out_dir, progress=False)
    second = materialize_parquet(src_dir, out_dir, progress=False)

    assert first.written == 5
    assert second.written == 0
    assert second.skipped.get("already_exists") == 5
    assert second.skipped.get("nsfw") == 1


def test_materialize_parquet_max_rows_bounds_the_read(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    src_dir = tmp_path / "src"
    out_dir = tmp_path / "out"
    _write_synthetic_shard(src_dir / "shard-00000.parquet")

    report = materialize_parquet(src_dir, out_dir, max_rows=2, progress=False)

    assert report.rows_read == 2


def test_materialize_parquet_rejects_unrecognized_label_value(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    rows = [
        {
            "image_name": "x",
            "format": "PNG",
            "resolution": "8x6",
            "mode": "RGB",
            "image_data": _make_image_bytes("PNG"),
            "model_name": "sdxl",
            "nsfw_flag": False,
            "prompt": None,
            "real_source": None,
            "subset": "manual",
            "split": "train",
            "label": "maybe",
            "architecture": None,
        }
    ]
    pq.write_table(pa.Table.from_pylist(rows), src_dir / "shard.parquet")

    with pytest.raises(ValueError, match="Unrecognized"):
        materialize_parquet(src_dir, tmp_path / "out", progress=False)


def test_prepare_after_materialize_uses_attributes_jsonl_for_generator_and_split(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    src_dir = tmp_path / "src"
    out_dir = tmp_path / "materialized"
    _write_synthetic_shard(src_dir / "shard-00000.parquet")
    materialize_parquet(src_dir, out_dir, progress=False)

    manifest, skipped, report = prepare(
        "Community Forensics", out_dir, tmp_path / "manifest" / "cf.jsonl"
    )

    assert skipped == []
    assert len(manifest.entries) == 5
    by_name = {Path(entry.path).stem: entry for entry in manifest.entries}
    assert by_name["fake0"].generator == "sdxl"
    assert by_name["fake0"].split == "train"
    assert by_name["fake1"].generator == "dalle3"
    assert by_name["fake1"].split == "val"
    assert by_name["real0"].label == "real"
    assert isinstance(report, AuditReport)
