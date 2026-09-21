"""Tests for imgforensics.data.layouts: Layout adapters and prepare()."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from imgforensics.data.audit import AuditReport
from imgforensics.data.layouts import (
    Layout,
    MaskRule,
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


def test_cocoglide_layout_attaches_masks_to_the_manipulated_half(tmp_path: Path) -> None:
    """Only the inpainted images carry a mask; the authentic ones must not.

    CocoGlide names each mask after its *authentic* partner
    (``<coco class>_<image id>_mask.png``) while the manipulated file is named
    ``glide_inpainting_val2017_<image id>_up.png``, so a mask_template pointed
    at the archive's own ``mask/`` folder would attach every mask to the wrong
    half. The packaged layout reads the ``mask_paired/`` folder instead (see
    its ``notes``); this reproduces that shape in miniature and pins the
    result, because getting it backwards would silently corrupt every pixel
    metric rather than fail.
    """
    layout = get_layout("CocoGlide")
    assert layout.mask_template == "mask_paired/{stem}_mask.png"

    _img(tmp_path / "real" / "airplane_139871.png")
    _img(tmp_path / "fake" / "glide_inpainting_val2017_139871_up.png")
    _img(tmp_path / "mask" / "airplane_139871_mask.png")
    _img(tmp_path / "mask_paired" / "glide_inpainting_val2017_139871_up_mask.png")

    manifest, _, _ = prepare("CocoGlide", tmp_path, tmp_path / "out" / "manifest.jsonl")

    by_label = {entry.label: entry for entry in manifest.entries}
    assert set(by_label) == {"real", "fake"}, "both halves must be labeled"
    assert by_label["real"].mask_path is None
    assert by_label["fake"].mask_path == "mask_paired/glide_inpainting_val2017_139871_up_mask.png"
    assert by_label["fake"].generator == "glide"
    assert by_label["real"].generator is None


# --- synthetic Synthbuster-style tree (per-generator folders) ---------------


def _make_synthbuster_like_tree(root: Path) -> None:
    # synthbuster.zip unpacks to one top-level ``synthbuster/`` folder with a
    # folder per generator inside it, and the layout reaches through it.
    for generator in ("dalle3", "sdxl", "firefly"):
        for i in range(2):
            _img(root / "synthbuster" / generator / f"img{i}.png")


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

    _, _, _, mask_of, _ = _callables_from_layout(layout, root)
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

    _, generator_of, _, _, _ = _callables_from_layout(layout, root)
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

    _, _, split_of, _, _ = _callables_from_layout(layout, root)
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


# --- synthetic ITW-SM-style tree (platform in the fake file names) --------------


def _make_itwsm_like_tree(root: Path) -> None:
    for platform in ("facebook", "instagram"):
        for i in range(2):
            _img(root / "0_real" / f"{platform.capitalize()}_real_{i}.jpg")
            _img(root / "1_fake" / f"{platform}_{i}.jpg")


def test_layout_adapter_reads_itwsm_platform_from_fake_file_names(tmp_path: Path) -> None:
    _make_itwsm_like_tree(tmp_path)
    manifest, skipped, _report = prepare("ITW-SM", tmp_path, tmp_path / "out" / "manifest.jsonl")

    assert skipped == []
    assert len(manifest.entries) == 8
    fakes = [entry for entry in manifest.entries if entry.label == "fake"]
    reals = [entry for entry in manifest.entries if entry.label == "real"]
    assert {entry.generator for entry in fakes} == {"facebook", "instagram"}
    # Reals never carry a generator, whatever their file name says.
    assert {entry.generator for entry in reals} == {None}


# --- synthetic TGIF-style tree (mask_rules + extra_regex) ------------------

_TGIF_SPLITS = ("training", "validation", "testing")
# One COCO category name with a space in it, which the layout's regexes must survive.
_TGIF_CATEGORIES = ("airplane", "hair drier")
_TGIF_IDS = ("134886", "135673")
_TGIF_MASK_TYPES = ("bbox", "segm")
_TGIF_VARIATIONS = (0, 1, 2)

# Stand-ins for the real geometries: the full-size original, the 1024-pipeline
# crop, and the 512x512 SD2 crop.
_TGIF_FULL = (64, 48)
_TGIF_1024 = (60, 44)
_TGIF_512 = (32, 32)


def _make_tgif_like_tree(root: Path) -> None:
    """The real TGIF naming in miniature: six folders, three splits, ten masks per id."""
    for split in _TGIF_SPLITS:
        for category in _TGIF_CATEGORIES:
            for image_id in _TGIF_IDS:
                orig = root / "orig" / split / category
                _img(orig / f"{image_id}_orig.png", _TGIF_FULL)
                _img(orig / f"{image_id}_orig_1024.png", _TGIF_1024)
                _img(orig / f"{image_id}_orig_512.png", _TGIF_512)

                sd2_fr = root / "sd2-fr" / split / category
                sdxl_fr = root / "sdxl-fr" / split / category
                sd2_sp = root / "sd2-sp" / split / category
                ps_sp = root / "ps-sp" / split / category

                masks = root / "masks" / split / category
                # Where the two crops sit inside the original; not inpainting masks.
                _img(masks / f"{image_id}_mask_512.png", _TGIF_FULL)
                _img(masks / f"{image_id}_mask_1024.png", _TGIF_FULL)
                for mask_type in _TGIF_MASK_TYPES:
                    _img(masks / f"{image_id}_mask_{mask_type}.png", _TGIF_FULL)
                    _img(masks / f"{image_id}_mask_{mask_type}.png_ps_mask.png", _TGIF_FULL)
                    _img(masks / f"{image_id}_mask_{mask_type}_512.png", _TGIF_512)
                    _img(masks / f"{image_id}_mask_{mask_type}_1024.png", _TGIF_1024)

                    stem = f"{image_id}_mask_{mask_type}.png"
                    for variation in _TGIF_VARIATIONS:
                        _img(sd2_fr / f"{stem}_sd2-512_{variation}.png", _TGIF_512)
                        _img(sdxl_fr / f"{stem}_sdxl-1024_{variation}.png", _TGIF_1024)
                        _img(sd2_sp / f"{stem}_ps_mask.png_sd2_{variation}.png", _TGIF_FULL)
                        _img(ps_sp / f"{stem}_ps_{variation}.png", _TGIF_FULL)


def test_tgif_layout_labels_generators_and_splits(tmp_path: Path) -> None:
    _make_tgif_like_tree(tmp_path)
    manifest, skipped, report = prepare("TGIF", tmp_path, tmp_path / "out" / "manifest.jsonl")

    assert skipped == []
    units = len(_TGIF_SPLITS) * len(_TGIF_CATEGORIES) * len(_TGIF_IDS)
    reals = [entry for entry in manifest.entries if entry.label == "real"]
    fakes = [entry for entry in manifest.entries if entry.label == "fake"]
    # 3 orig variants and 4 x 2 x 3 = 24 fakes per authentic image; masks/ is
    # matched by neither glob and never becomes an entry.
    assert len(reals) == 3 * units
    assert len(fakes) == 24 * units
    assert len(manifest.entries) == 27 * units

    assert {entry.generator for entry in fakes} == {"sd2-fr", "sdxl-fr", "sd2-sp", "ps-sp"}
    assert {entry.generator for entry in reals} == {None}
    counts = manifest.summary()
    assert counts["split"] == {"train": 9 * units, "val": 9 * units, "test": 9 * units}
    assert isinstance(report, AuditReport)


def test_tgif_layout_pairs_each_fake_subset_with_its_own_mask(tmp_path: Path) -> None:
    """Each pipeline's ground truth is the mask at that pipeline's own geometry.

    Getting this wrong is silent: a 512x512 fake scored against the full-size
    mask, or a spliced fake scored against the unbordered mask, still produces
    pixel metrics -- just meaningless ones.
    """
    _make_tgif_like_tree(tmp_path)
    manifest, _, _ = prepare("TGIF", tmp_path, tmp_path / "out" / "manifest.jsonl")
    by_path = {entry.path: entry for entry in manifest.entries}

    prefix = "testing/hair drier"
    assert (
        by_path[f"sd2-fr/{prefix}/134886_mask_bbox.png_sd2-512_0.png"].mask_path
        == f"masks/{prefix}/134886_mask_bbox_512.png"
    )
    assert (
        by_path[f"sdxl-fr/{prefix}/134886_mask_segm.png_sdxl-1024_2.png"].mask_path
        == f"masks/{prefix}/134886_mask_segm_1024.png"
    )
    # Spliced fakes pair with the bordered mask that was fed to the inpainter.
    assert (
        by_path[f"sd2-sp/{prefix}/135673_mask_bbox.png_ps_mask.png_sd2_1.png"].mask_path
        == f"masks/{prefix}/135673_mask_bbox.png_ps_mask.png"
    )
    assert (
        by_path[f"ps-sp/{prefix}/135673_mask_segm.png_ps_1.png"].mask_path
        == f"masks/{prefix}/135673_mask_segm.png_ps_mask.png"
    )

    # Every fake in the tree is paired, and no authentic image ever is.
    assert all(entry.mask_path is not None for entry in manifest.entries if entry.label == "fake")
    assert all(entry.mask_path is None for entry in manifest.entries if entry.label == "real")

    # The paired mask is the one at the image's own geometry.
    for entry in manifest.entries:
        if entry.mask_path is None:
            continue
        with Image.open(tmp_path / entry.mask_path) as mask:
            assert mask.size == (entry.width, entry.height), entry.path


def test_tgif_layout_records_variant_and_mask_type_in_extra(tmp_path: Path) -> None:
    _make_tgif_like_tree(tmp_path)
    manifest, _, _ = prepare("TGIF", tmp_path, tmp_path / "out" / "manifest.jsonl")
    by_path = {entry.path: entry for entry in manifest.entries}

    prefix = "training/airplane"
    assert by_path[f"orig/{prefix}/134886_orig.png"].extra == {"variant": "orig"}
    assert by_path[f"orig/{prefix}/134886_orig_512.png"].extra == {"variant": "orig_512"}
    assert by_path[f"orig/{prefix}/134886_orig_1024.png"].extra == {"variant": "orig_1024"}

    assert by_path[f"sd2-fr/{prefix}/134886_mask_bbox.png_sd2-512_0.png"].extra == {
        "mask_type": "bbox",
        "variation": "0",
    }
    assert by_path[f"ps-sp/{prefix}/135673_mask_segm.png_ps_2.png"].extra == {
        "mask_type": "segm",
        "variation": "2",
    }
    assert by_path[f"sd2-sp/{prefix}/135673_mask_bbox.png_ps_mask.png_sd2_1.png"].extra == {
        "mask_type": "bbox",
        "variation": "1",
    }
    assert by_path[f"sdxl-fr/{prefix}/134886_mask_segm.png_sdxl-1024_2.png"].extra == {
        "mask_type": "segm",
        "variation": "2",
    }
    # Every entry carries one group set or the other, never both, never neither.
    for entry in manifest.entries:
        expected = {"variant"} if entry.label == "real" else {"mask_type", "variation"}
        assert set(entry.extra) == expected, entry.path


def test_tgif_layout_counts_fakes_left_without_a_mask(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unpaired fake is never an error, but it must never be silent either."""
    _make_tgif_like_tree(tmp_path)
    # A mask that did not come out of the archive: the rule matches, the file is gone.
    (tmp_path / "masks/testing/airplane/134886_mask_bbox_512.png").unlink()
    # A file no rule knows how to read: still a fake, just an unpaired one.
    _img(tmp_path / "sd2-fr/testing/airplane/134886_mask_bbox.png_sd2-512_99b.png", _TGIF_512)

    manifest, _, _ = prepare("TGIF", tmp_path, tmp_path / "out" / "manifest.jsonl")
    by_path = {entry.path: entry for entry in manifest.entries}

    missing_file = by_path["sd2-fr/testing/airplane/134886_mask_bbox.png_sd2-512_0.png"]
    assert missing_file.label == "fake"
    assert missing_file.mask_path is None
    unmatched = by_path["sd2-fr/testing/airplane/134886_mask_bbox.png_sd2-512_99b.png"]
    assert unmatched.label == "fake"
    assert unmatched.mask_path is None
    assert unmatched.extra == {}  # extra_regex does not match it either

    out = capsys.readouterr().out
    assert "1 matched no rule" in out
    assert "3 had no mask file on disk" in out  # the deleted mask served 3 variations


def test_tgif_layout_entry_keeps_the_folder_names_the_share_serves() -> None:
    layout = get_layout("TGIF")
    assert layout.real_globs == ["orig/**"]
    assert layout.fake_globs == ["sd2-sp/**", "ps-sp/**", "sd2-fr/**", "sdxl-fr/**"]
    assert len(layout.mask_rules) == 4
    assert layout.mask_template is None
    assert layout.extra_regex is not None


# --- Layout.mask_rules / extra_regex internals -----------------------------


def test_mask_rules_take_precedence_over_mask_template(tmp_path: Path) -> None:
    layout = Layout(
        dataset="unit-test",
        fake_globs=["fake/**"],
        mask_template="masks/{stem}.png",
        mask_rules=[MaskRule(regex=r"^fake/(?P<stem>[^/]+)\.png$", template="gt/{stem}_gt.png")],
    )
    _img(tmp_path / "fake" / "a.png")
    _img(tmp_path / "masks" / "a.png")
    _img(tmp_path / "gt" / "a_gt.png")

    from imgforensics.data.layouts import _callables_from_layout

    _, _, _, mask_of, _ = _callables_from_layout(layout, tmp_path)
    assert mask_of(tmp_path / "fake" / "a.png") == (tmp_path / "gt" / "a_gt.png").resolve()


def test_mask_rules_are_tried_in_order_and_first_match_wins(tmp_path: Path) -> None:
    layout = Layout(
        dataset="unit-test",
        fake_globs=["fake/**"],
        mask_rules=[
            MaskRule(regex=r"^fake/(?P<stem>[^/]+)_sp\.png$", template="gt/{stem}_sp_gt.png"),
            MaskRule(regex=r"^fake/(?P<stem>[^/]+)\.png$", template="gt/{stem}_gt.png"),
        ],
    )

    from imgforensics.data.layouts import _mask_from_rules

    assert _mask_from_rules("fake/a_sp.png", layout) == "gt/a_sp_gt.png"
    assert _mask_from_rules("fake/b.png", layout) == "gt/b_gt.png"
    assert _mask_from_rules("real/b.png", layout) is None


def test_layout_without_mask_rules_still_uses_mask_template(tmp_path: Path) -> None:
    """Every other packaged layout must behave exactly as it did before mask_rules."""
    for packaged in load_layouts():
        if packaged.dataset != "TGIF":
            assert packaged.mask_rules == []
            assert packaged.extra_regex is None

    layout = Layout(dataset="unit-test", fake_globs=["fake/**"], mask_template="masks/{stem}.png")
    _img(tmp_path / "fake" / "a.png")
    _img(tmp_path / "masks" / "a.png")

    from imgforensics.data.layouts import _callables_from_layout

    _, _, _, mask_of, extra_of = _callables_from_layout(layout, tmp_path)
    assert mask_of(tmp_path / "fake" / "a.png") == (tmp_path / "masks" / "a.png").resolve()
    assert extra_of(tmp_path / "fake" / "a.png") == {}


def test_mask_rule_rejects_an_uncompilable_regex() -> None:
    with pytest.raises(ValidationError, match="invalid mask rule regex"):
        MaskRule(regex=r"^fake/(?P<stem>[^/]+\.png$", template="gt/{stem}.png")


def test_mask_rule_rejects_a_template_field_the_regex_has_no_group_for() -> None:
    with pytest.raises(ValidationError, match="no named group"):
        MaskRule(regex=r"^fake/(?P<stem>[^/]+)\.png$", template="gt/{split}/{stem}.png")


def test_layout_rejects_an_uncompilable_extra_regex() -> None:
    with pytest.raises(ValidationError, match="invalid extra_regex"):
        Layout(dataset="unit-test", fake_globs=["fake/**"], extra_regex=r"(?P<oops>")


def test_extra_regex_omits_groups_that_did_not_participate() -> None:
    layout = Layout(
        dataset="unit-test",
        real_globs=["real/**"],
        fake_globs=["fake/**"],
        extra_regex=r"^(?:real/(?P<variant>[a-z]+)|fake/(?P<generator>[a-z]+))_\d+\.png$",
    )

    from imgforensics.data.layouts import _extra_of

    assert _extra_of("real/orig_1.png", layout) == {"variant": "orig"}
    assert _extra_of("fake/sdxl_1.png", layout) == {"generator": "sdxl"}
    assert _extra_of("other/x_1.png", layout) == {}


def test_tgif_layout_reports_full_mask_coverage_on_a_complete_tree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The happy-path form of the coverage line: nothing unpaired, and it still says so."""
    _make_tgif_like_tree(tmp_path)
    manifest, _, _ = prepare("TGIF", tmp_path, tmp_path / "out" / "manifest.jsonl")

    fakes = sum(1 for entry in manifest.entries if entry.label == "fake")
    out = capsys.readouterr().out
    assert (
        f"mask rules: {fakes}/{fakes} fake(s) paired with a mask "
        "(0 matched no rule, 0 had no mask file on disk)"
    ) in out


def test_mask_rule_template_over_a_group_that_did_not_participate_raises() -> None:
    """A template naming a declared-but-optional group fails with the rule and the path.

    Validation cannot catch this one -- the group exists, it simply does not
    take part in every match -- so the failure has to carry enough context to
    find the offending rule instead of surfacing as a bare KeyError.
    """
    layout = Layout(
        dataset="unit-test",
        fake_globs=["fake/**"],
        mask_rules=[
            MaskRule(
                regex=r"^fake/(?:(?P<a>alpha)|beta)_(?P<id>\d+)\.png$",
                template="gt/{a}_{id}.png",
            )
        ],
    )

    from imgforensics.data.layouts import _mask_from_rules

    assert _mask_from_rules("fake/alpha_7.png", layout) == "gt/alpha_7.png"
    with pytest.raises(ValueError, match="could not be filled in"):
        _mask_from_rules("fake/beta_7.png", layout)


def test_mask_rule_template_with_a_positional_field_raises() -> None:
    layout = Layout(
        dataset="unit-test",
        fake_globs=["fake/**"],
        mask_rules=[MaskRule(regex=r"^fake/(?P<id>\d+)\.png$", template="gt/{}.png")],
    )

    from imgforensics.data.layouts import _mask_from_rules

    with pytest.raises(ValueError, match="could not be filled in"):
        _mask_from_rules("fake/7.png", layout)
