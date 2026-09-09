"""Tests for imgforensics.eval.runner: the benchmark runner and its report tables.

The manifest fixture mirrors what the ``metadata`` signal keys on: reals are
plain camera-less JPEGs (no markers at all -> "uncertain", score 0.5), fakes
are PNGs carrying Automatic1111's ``parameters`` text chunk (-> "fake", score
0.95) -- see ``imgforensics.signals.metadata._ai_markers``. That gap gives
``metadata`` a clean AUC of 1.0 on this fixture without needing any real
generative content.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
from conftest import natural_like_image
from PIL import Image, PngImagePlugin

from imgforensics.data.manifest import Manifest, ManifestEntry, ManifestMeta
from imgforensics.eval.robustness import Perturbation, RobustnessSuite
from imgforensics.eval.runner import BenchmarkConfig, BenchmarkResult, run_benchmark

_IMAGE_SIZE = (96, 96)


def _build_manifest(root: Path, *, with_splits: bool = False) -> Manifest:
    (root / "real").mkdir(parents=True, exist_ok=True)
    (root / "fake").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)

    entries: list[ManifestEntry] = []
    for i in range(6):
        image = natural_like_image(size=_IMAGE_SIZE, seed=10 + i)
        image.save(root / "real" / f"r{i}.jpg", format="JPEG", quality=90)
        entries.append(
            ManifestEntry(
                path=f"real/r{i}.jpg",
                label="real",
                source="synthetic-suite",
                generator=None,
                split=("val" if with_splits and i < 2 else "test"),
                mask_path=None,
                sha256=f"{i:064x}",
                width=_IMAGE_SIZE[0],
                height=_IMAGE_SIZE[1],
                format="JPEG",
                jpeg_quality=90,
            )
        )

    for i in range(6):
        image = natural_like_image(size=_IMAGE_SIZE, seed=20 + i)
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", "Steps: 20, Sampler: Euler a, Stable Diffusion")
        image.save(root / "fake" / f"f{i}.png", format="PNG", pnginfo=info)

        mask_path = None
        if i < 2:
            mask = np.zeros(_IMAGE_SIZE[::-1], dtype=np.uint8)
            mask[16:80, 16:80] = 255  # a 64x64 rectangle
            Image.fromarray(mask, mode="L").save(root / "masks" / f"f{i}_mask.png")
            mask_path = f"masks/f{i}_mask.png"

        entries.append(
            ManifestEntry(
                path=f"fake/f{i}.png",
                label="fake",
                source="synthetic-suite",
                generator="sd-webui",
                split=("val" if with_splits and i < 2 else "test"),
                mask_path=mask_path,
                sha256=f"{100 + i:064x}",
                width=_IMAGE_SIZE[0],
                height=_IMAGE_SIZE[1],
                format="PNG",
                jpeg_quality=None,
            )
        )

    return Manifest(
        meta=ManifestMeta(dataset="runner-test", root=str(root), created=date.today().isoformat()),
        entries=entries,
    )


@pytest.fixture
def manifest_root(tmp_path: Path) -> Path:
    _build_manifest(tmp_path).save(tmp_path / "manifest.jsonl")
    return tmp_path


@pytest.fixture
def manifest(manifest_root: Path) -> Manifest:
    return Manifest.load(manifest_root / "manifest.jsonl")


def _run(manifest: Manifest, **config_kwargs: object) -> BenchmarkResult:
    config = BenchmarkConfig(robustness=None, **config_kwargs)
    return run_benchmark(manifest, config, root=Path(manifest.meta.root), progress=False)


def test_runner_produces_all_tables(manifest: Manifest) -> None:
    result = _run(manifest, detectors=["metadata", "ela"], include_baselines=True)

    assert len(result.records) == 12 * 6  # 12 images * (metadata, ela, 4 baselines)
    assert len(result.pixel_records) == 2  # only the two masked fakes, only ela has a heatmap

    image_rows = result.image_table()
    assert {row.detector for row in image_rows} == {
        "metadata",
        "ela",
        "constant_real",
        "constant_fake",
        "random",
        "signals_mean",
    }

    robustness_rows = result.robustness_table()
    assert robustness_rows  # non-empty

    group_rows = result.per_group_table(key="source")
    assert group_rows  # every entry shares one source, so this is well-defined

    pixel_rows = result.pixel_table()
    assert pixel_rows is not None
    assert {row.detector for row in pixel_rows} == {"ela"}

    timing_rows = result.timing_table()
    assert {row.detector for row in timing_rows} == {row.detector for row in image_rows}

    markdown = result.to_markdown()
    assert "runner-test" in markdown
    assert "Benchmark report" in markdown


def test_constant_baselines_have_auc_one_half(manifest: Manifest) -> None:
    result = _run(manifest, detectors=[], include_baselines=True)

    rows = {row.detector: row for row in result.robustness_table()}
    assert rows["constant_real"].auc == pytest.approx(0.5)
    assert rows["constant_fake"].auc == pytest.approx(0.5)


def test_metadata_has_auc_one_on_clean_level(manifest: Manifest) -> None:
    result = _run(manifest, detectors=["metadata"], include_baselines=False)

    image_rows = {row.detector: row for row in result.image_table()}
    metadata_row = image_rows["metadata"]
    assert metadata_row.fixed is not None
    assert metadata_row.fixed.auc == pytest.approx(1.0)
    # Reals score exactly 0.5 (uncertain, no markers); fakes score 0.95. Under the
    # strict `score > threshold` convention, the fixed threshold of 0.5 does not count
    # an abstaining real as a false "fake" call: accuracy and FPR should be perfect,
    # not the AUC-1.0-but-accuracy-0.5/FPR-1.0 result the old `>=` convention gave.
    assert metadata_row.fixed.accuracy == pytest.approx(1.0)
    assert metadata_row.fixed.fpr == pytest.approx(0.0)


def test_robustness_table_has_one_row_per_level(manifest: Manifest) -> None:
    suite = RobustnessSuite(
        levels=[
            Perturbation(name="clean", kind="clean", params={}),
            Perturbation(name="jpeg_q75", kind="jpeg", params={"quality": 75}),
        ]
    )
    config = BenchmarkConfig(detectors=["metadata"], robustness=suite)
    result = run_benchmark(manifest, config, root=Path(manifest.meta.root), progress=False)

    rows = [row for row in result.robustness_table() if row.detector == "metadata"]
    assert [row.level for row in rows] == ["clean", "jpeg_q75"]


def test_json_round_trip_regenerates_identical_tables(manifest: Manifest, tmp_path: Path) -> None:
    result = _run(manifest, detectors=["metadata", "ela"], include_baselines=True)

    out_path = tmp_path / "results.json"
    result.save_json(out_path)
    reloaded = BenchmarkResult.load_json(out_path)

    assert reloaded.to_markdown() == result.to_markdown()
    assert [row.__dict__ for row in reloaded.robustness_table()] == [
        row.__dict__ for row in result.robustness_table()
    ]


def test_missing_file_is_reported_not_fatal(manifest: Manifest, tmp_path: Path) -> None:
    entries = list(manifest.entries)
    entries.append(
        ManifestEntry(
            path="fake/does_not_exist.png",
            label="fake",
            source="synthetic-suite",
            sha256="f" * 64,
            width=1,
            height=1,
            format="PNG",
        )
    )
    broken_manifest = Manifest(meta=manifest.meta, entries=entries)

    result = run_benchmark(
        broken_manifest,
        BenchmarkConfig(detectors=["metadata"], robustness=None),
        root=Path(manifest.meta.root),
        progress=False,
    )

    assert len(result.missing_files) == 1
    assert "does_not_exist" in result.missing_files[0]
    assert result.entry_count == len(entries)


def test_limit_and_shuffle_are_deterministic(manifest: Manifest) -> None:
    result_a = _run(manifest, detectors=["metadata"], limit=4)
    result_b = _run(manifest, detectors=["metadata"], limit=4)

    paths_a = sorted({record.entry_path for record in result_a.records})
    paths_b = sorted({record.entry_path for record in result_b.records})
    assert paths_a == paths_b
    assert len({record.entry_path for record in result_a.records}) == 4


def test_threshold_caveat_present_without_val_split(manifest: Manifest) -> None:
    result = _run(manifest, detectors=["metadata"])
    assert result.threshold_caveat == "threshold tuned in-sample"


def test_threshold_caveat_absent_with_val_split(tmp_path: Path) -> None:
    manifest_with_splits = _build_manifest(tmp_path, with_splits=True)
    manifest_with_splits.save(tmp_path / "manifest.jsonl")
    loaded = Manifest.load(tmp_path / "manifest.jsonl")

    config = BenchmarkConfig(detectors=["metadata"], robustness=None, threshold_split="val")
    result = run_benchmark(loaded, config, root=tmp_path, progress=False)

    assert result.threshold_caveat is None
