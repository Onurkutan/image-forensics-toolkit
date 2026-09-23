"""CLI tests for the `datasets`, `manifest build`, and `audit` commands."""

from __future__ import annotations

from pathlib import Path

from conftest import natural_like_image
from PIL import Image
from typer.testing import CliRunner

from imgforensics.cli import app
from imgforensics.data.manifest import (
    Manifest,
    ManifestEntry,
    ManifestMeta,
    build_manifest,
    label_from_parent_folder,
)

runner = CliRunner()


def test_cli_datasets_list() -> None:
    result = runner.invoke(app, ["datasets", "list"])
    assert result.exit_code == 0
    assert "TGIF" in result.stdout
    assert "Community Forensics" in result.stdout


def test_cli_datasets_show() -> None:
    result = runner.invoke(app, ["datasets", "show", "TGIF"])
    assert result.exit_code == 0
    assert "localization" in result.stdout
    assert "CC BY-SA 4.0" in result.stdout


def test_cli_datasets_show_unknown_name_errors() -> None:
    result = runner.invoke(app, ["datasets", "show", "does-not-exist"])
    assert result.exit_code != 0


def _make_tree(root: Path) -> None:
    (root / "real").mkdir(parents=True)
    (root / "fake").mkdir(parents=True)
    for i in range(4):
        Image.new("RGB", (48, 32)).save(root / "real" / f"r{i}.jpg", format="JPEG", quality=85)
    for i in range(4):
        Image.new("RGB", (48, 32)).save(root / "fake" / f"f{i}.jpg", format="JPEG", quality=85)


def test_cli_manifest_build_and_load(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    out_path = tmp_path / "manifest.jsonl"

    result = runner.invoke(
        app,
        [
            "manifest",
            "build",
            str(tmp_path),
            "--dataset",
            "cli-test",
            "--out",
            str(out_path),
            "--license",
            "MIT",
            "--commercial-ok",
        ],
    )

    assert result.exit_code == 0
    assert out_path.exists()
    manifest = Manifest.load(out_path)
    assert len(manifest.entries) == 8
    assert manifest.meta.dataset == "cli-test"
    assert manifest.meta.commercial_ok is True


def test_cli_manifest_build_no_commercial_ok_flag(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    out_path = tmp_path / "manifest.jsonl"

    result = runner.invoke(
        app,
        ["manifest", "build", str(tmp_path), "--dataset", "cli-test", "--out", str(out_path)],
    )

    assert result.exit_code == 0
    manifest = Manifest.load(out_path)
    assert manifest.meta.commercial_ok is None


def test_cli_audit_passes_on_balanced_manifest(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    manifest_path = tmp_path / "manifest.jsonl"
    runner.invoke(
        app,
        ["manifest", "build", str(tmp_path), "--dataset", "cli-test", "--out", str(manifest_path)],
    )

    result = runner.invoke(app, ["audit", str(manifest_path)])
    assert result.exit_code == 0
    assert "Bias audit report" in result.stdout


def test_cli_audit_strict_exits_nonzero_on_bias(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "fake").mkdir()
    for i in range(10):
        Image.new("RGB", (48, 32)).save(tmp_path / "real" / f"r{i}.jpg", format="JPEG", quality=85)
    for i in range(10):
        Image.new("RGB", (48, 32)).save(tmp_path / "fake" / f"f{i}.png", format="PNG")

    manifest_path = tmp_path / "manifest.jsonl"
    runner.invoke(
        app,
        ["manifest", "build", str(tmp_path), "--dataset", "cli-test", "--out", str(manifest_path)],
    )

    result = runner.invoke(app, ["audit", str(manifest_path), "--strict"])
    assert result.exit_code == 1

    lenient_result = runner.invoke(app, ["audit", str(manifest_path)])
    assert lenient_result.exit_code == 0


def _make_casia_like_tree(root: Path) -> None:
    (root / "Au").mkdir(parents=True, exist_ok=True)
    (root / "Tp").mkdir(parents=True, exist_ok=True)
    (root / "Gt").mkdir(parents=True, exist_ok=True)
    for i in range(3):
        Image.new("RGB", (16, 12)).save(root / "Au" / f"Au_{i}.jpg", format="JPEG")
        Image.new("RGB", (16, 12)).save(root / "Tp" / f"Tp_{i}.jpg", format="JPEG")
        Image.new("L", (16, 12)).save(root / "Gt" / f"Tp_{i}_gt.png", format="PNG")


def test_cli_datasets_prepare_uses_layout_and_writes_manifest(tmp_path: Path) -> None:
    root = tmp_path / "src"
    _make_casia_like_tree(root)
    out_path = tmp_path / "manifest.jsonl"

    result = runner.invoke(
        app, ["datasets", "prepare", "CASIA v2.0", "--src", str(root), "--out", str(out_path)]
    )

    assert result.exit_code == 0
    manifest = Manifest.load(out_path)
    assert len(manifest.entries) == 6
    assert "Bias audit report" in result.stdout


def test_cli_datasets_prepare_unregistered_dataset_falls_back(tmp_path: Path) -> None:
    root = tmp_path / "src"
    (root / "real").mkdir(parents=True)
    (root / "fake").mkdir(parents=True)
    Image.new("RGB", (16, 12)).save(root / "real" / "a.png", format="PNG")
    Image.new("RGB", (16, 12)).save(root / "fake" / "b.png", format="PNG")
    out_path = tmp_path / "manifest.jsonl"

    result = runner.invoke(
        app, ["datasets", "prepare", "NotRegistered", "--src", str(root), "--out", str(out_path)]
    )

    assert result.exit_code == 0
    manifest = Manifest.load(out_path)
    assert len(manifest.entries) == 2


def test_cli_manifest_sample_writes_a_subsample(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    manifest_path = tmp_path / "manifest.jsonl"
    runner.invoke(
        app,
        ["manifest", "build", str(tmp_path), "--dataset", "cli-test", "--out", str(manifest_path)],
    )

    out_path = tmp_path / "sampled.jsonl"
    result = runner.invoke(
        app,
        [
            "manifest",
            "sample",
            str(manifest_path),
            "--n",
            "4",
            "--out",
            str(out_path),
            "--seed",
            "0",
        ],
    )

    assert result.exit_code == 0
    sampled = Manifest.load(out_path)
    assert len(sampled.entries) == 4
    assert "sampled 4 of 8 with seed 0" in sampled.meta.notes


def _filterable_manifest(path: Path) -> None:
    """Save a two-split, two-generator manifest with `extra` and masks, at ``path``."""
    entries: list[ManifestEntry] = []
    counter = 0
    for split in ("train", "val"):
        for index in range(3):
            for variant in ("orig_512", "orig_1024"):
                counter += 1
                entries.append(
                    ManifestEntry(
                        path=f"orig/{split}/{index}_{variant}.png",
                        label="real",
                        source="ds",
                        split=split,  # type: ignore[arg-type]
                        # The two sizes of image 0 are the same file on disk.
                        sha256=f"{0 if index == 0 else counter:064d}",
                        width=512,
                        height=512,
                        format="PNG",
                        extra={"variant": variant},
                    )
                )
        for generator in ("sd2-fr", "ps-sp"):
            for index in range(3):
                counter += 1
                entries.append(
                    ManifestEntry(
                        path=f"{generator}/{split}/{index}.png",
                        label="fake",
                        source="ds",
                        generator=generator,
                        split=split,  # type: ignore[arg-type]
                        mask_path=None if index == 2 else f"masks/{generator}/{index}.png",
                        sha256=f"{counter:064d}",
                        width=512,
                        height=512,
                        format="PNG",
                        extra={"mask_type": "segm" if index else "bbox"},
                    )
                )
    meta = ManifestMeta(dataset="ds", root=str(path.parent), created="2026-01-01")
    Manifest(meta=meta, entries=entries).save(path)


def test_cli_manifest_filter_selects_a_split_label_and_generator(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    _filterable_manifest(manifest_path)
    out_path = tmp_path / "filtered.jsonl"

    result = runner.invoke(
        app,
        [
            "manifest",
            "filter",
            str(manifest_path),
            "--out",
            str(out_path),
            "--split",
            "train",
            "--label",
            "fake",
            "--generator",
            "sd2-fr",
            "--require-mask",
        ],
    )

    assert result.exit_code == 0, result.stdout
    filtered = Manifest.load(out_path)
    assert len(filtered.entries) == 2
    assert {entry.generator for entry in filtered.entries} == {"sd2-fr"}
    assert all(entry.mask_path is not None for entry in filtered.entries)
    assert "split=train" in (filtered.meta.notes or "")
    assert "mask required" in (filtered.meta.notes or "")


def test_cli_manifest_filter_extra_and_dedupe_are_deterministic(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    _filterable_manifest(manifest_path)

    outputs = []
    for run in range(2):
        out_path = tmp_path / f"reals_{run}.jsonl"
        result = runner.invoke(
            app,
            [
                "manifest",
                "filter",
                str(manifest_path),
                "--out",
                str(out_path),
                "--split",
                "train",
                "--label",
                "real",
                "--extra",
                "variant=orig_512",
                "--extra",
                "variant=orig_1024",
                "--dedupe-sha256",
            ],
        )
        assert result.exit_code == 0, result.stdout
        outputs.append(out_path.read_bytes())

    # Six entries, minus the one duplicate pair that dedupe collapses.
    filtered = Manifest.load(tmp_path / "reals_0.jsonl")
    assert len(filtered.entries) == 5
    assert outputs[0] == outputs[1]
    assert "deduped by sha256" in (filtered.meta.notes or "")


def test_cli_manifest_filter_rejects_a_misspelled_split_or_label(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    _filterable_manifest(manifest_path)
    out_path = tmp_path / "out.jsonl"

    for option, value in (("--split", "trian"), ("--label", "reals")):
        result = runner.invoke(
            app,
            ["manifest", "filter", str(manifest_path), "--out", str(out_path), option, value],
        )

        # Without the check this matched nothing, wrote an empty manifest and
        # exited 0 -- a silent failure at the top of a training run.
        assert result.exit_code != 0, f"{option} {value} was accepted"
        assert not out_path.exists()


def test_cli_manifest_filter_rejects_an_extra_without_a_value(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    _filterable_manifest(manifest_path)

    result = runner.invoke(
        app,
        [
            "manifest",
            "filter",
            str(manifest_path),
            "--out",
            str(tmp_path / "out.jsonl"),
            "--extra",
            "variant",
        ],
    )

    assert result.exit_code != 0


def test_cli_manifest_merge_combines_manifests(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _make_tree(root_a)
    _make_tree(root_b)
    manifest_a_path = tmp_path / "a.jsonl"
    manifest_b_path = tmp_path / "b.jsonl"
    runner.invoke(
        app, ["manifest", "build", str(root_a), "--dataset", "a", "--out", str(manifest_a_path)]
    )
    runner.invoke(
        app, ["manifest", "build", str(root_b), "--dataset", "b", "--out", str(manifest_b_path)]
    )

    out_path = tmp_path / "merged.jsonl"
    result = runner.invoke(
        app,
        ["manifest", "merge", str(manifest_a_path), str(manifest_b_path), "--out", str(out_path)],
    )

    assert result.exit_code == 0
    merged = Manifest.load(out_path)
    assert len(merged.entries) == 16
    assert merged.meta.dataset == "a+b"


def test_cli_manifest_crop_center_mode_equalizes_resolution(tmp_path: Path) -> None:
    root = tmp_path / "src"
    (root / "real").mkdir(parents=True)
    (root / "fake").mkdir(parents=True)
    natural_like_image(size=(1024, 1024), seed=1).save(root / "real" / "r0.png", format="PNG")
    natural_like_image(size=(512, 512), seed=2).save(root / "fake" / "f0.png", format="PNG")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest, _ = build_manifest(
        root, dataset="cli-crop", label_of=label_from_parent_folder, progress=False
    )
    manifest.save(manifest_path)

    out_dir = tmp_path / "cropped"
    out_path = tmp_path / "cropped.jsonl"
    result = runner.invoke(
        app,
        [
            "manifest",
            "crop",
            str(manifest_path),
            "--out-dir",
            str(out_dir),
            "--out",
            str(out_path),
            "--size",
            "512",
            "--mode",
            "center",
            "--label",
            "real",
        ],
    )

    assert result.exit_code == 0, result.stdout
    cropped = Manifest.load(out_path)
    assert len(cropped.entries) == 2
    real_entry = next(e for e in cropped.entries if e.label == "real")
    fake_entry = next(e for e in cropped.entries if e.label == "fake")
    assert real_entry.width == 512
    assert real_entry.height == 512
    assert fake_entry.width == 512
    assert fake_entry.height == 512
    # The fake entry, outside --label, is a byte-identical copy.
    assert (out_dir / "fake" / "f0.png").read_bytes() == (root / "fake" / "f0.png").read_bytes()
    assert "Crop report" in result.stdout


def test_cli_manifest_crop_rejects_unknown_mode(tmp_path: Path) -> None:
    root = tmp_path / "src"
    (root / "real").mkdir(parents=True)
    natural_like_image(size=(600, 600), seed=1).save(root / "real" / "a.png", format="PNG")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest, _ = build_manifest(
        root, dataset="cli-crop", label_of=label_from_parent_folder, progress=False
    )
    manifest.save(manifest_path)

    result = runner.invoke(
        app,
        [
            "manifest",
            "crop",
            str(manifest_path),
            "--out-dir",
            str(tmp_path / "cropped"),
            "--out",
            str(tmp_path / "cropped.jsonl"),
            "--size",
            "512",
            "--mode",
            "bogus",
        ],
    )
    assert result.exit_code != 0
