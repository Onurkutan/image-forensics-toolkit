"""Tests for imgforensics.eval.workers: the per-process worker function dispatched
by the benchmark runner's ``ProcessPoolExecutor`` path. Called directly here (no
pool) so these stay fast; the pool itself is exercised in
``tests/test_eval_runner.py``'s ``test_workers_parallel_matches_sequential``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from conftest import natural_like_image
from PIL import Image

from imgforensics.data.manifest import ManifestEntry
from imgforensics.eval import workers as workers_module
from imgforensics.eval.robustness import Perturbation
from imgforensics.eval.workers import WorkerTask, is_worker_eligible, run_worker_task
from imgforensics.signals import SIGNAL_NAMES

_CLEAN = Perturbation(name="clean", kind="clean", params={})


def _entry(**overrides: object) -> ManifestEntry:
    defaults: dict[str, object] = dict(
        path="img.jpg",
        label="real",
        source="test",
        sha256="0" * 64,
        width=64,
        height=64,
        format="JPEG",
        jpeg_quality=90,
    )
    defaults.update(overrides)
    return ManifestEntry(**defaults)  # type: ignore[arg-type]


def _masked_png_entry(mask_path: str) -> ManifestEntry:
    return _entry(
        path="img.png", label="fake", mask_path=mask_path, format="PNG", jpeg_quality=None
    )


def test_is_worker_eligible_covers_signals_and_cheap_baselines() -> None:
    for name in SIGNAL_NAMES:
        assert is_worker_eligible(name), name
    for name in ("constant_real", "constant_fake", "random"):
        assert is_worker_eligible(name), name


def test_is_worker_eligible_excludes_signals_mean_and_unknown_names() -> None:
    for name in ("signals_mean", "dinov2_head", "totally_unknown_detector"):
        assert not is_worker_eligible(name), name


def test_run_worker_task_scores_assigned_detectors(tmp_path: Path) -> None:
    natural_like_image(size=(64, 64), seed=1).save(tmp_path / "img.jpg", format="JPEG", quality=90)
    entry = _entry(path="img.jpg")
    task = WorkerTask(
        root=tmp_path, entry=entry, perturbation=_CLEAN, detector_names=("metadata", "ela")
    )

    result = run_worker_task(task)

    assert {r.detector for r in result.records} == {"metadata", "ela"}
    assert all(r.entry_path == "img.jpg" and r.level == "clean" for r in result.records)
    assert result.missing_files == []


def test_run_worker_task_supports_baseline_names(tmp_path: Path) -> None:
    natural_like_image(size=(32, 32), seed=2).save(tmp_path / "img.jpg", format="JPEG", quality=90)
    entry = _entry(path="img.jpg")
    task = WorkerTask(
        root=tmp_path,
        entry=entry,
        perturbation=_CLEAN,
        detector_names=("constant_real", "constant_fake", "random"),
    )

    result = run_worker_task(task)
    scores = {r.detector: r.score for r in result.records}

    assert scores["constant_real"] == 0.0
    assert scores["constant_fake"] == 1.0
    assert 0.0 <= scores["random"] <= 1.0


def _masked_image(tmp_path: Path, seed: int) -> ManifestEntry:
    """Write a 64x64 PNG plus its mask under ``tmp_path`` and return the entry."""
    natural_like_image(size=(64, 64), seed=seed).save(tmp_path / "img.png", format="PNG")
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:40, 10:40] = 255
    Image.fromarray(mask, mode="L").save(tmp_path / "mask.png")
    return _masked_png_entry("mask.png")


def test_run_worker_task_computes_pixel_metrics_for_masked_clean_entry(tmp_path: Path) -> None:
    entry = _masked_image(tmp_path, seed=3)
    task = WorkerTask(root=tmp_path, entry=entry, perturbation=_CLEAN, detector_names=("ela",))

    result = run_worker_task(task)

    assert len(result.pixel_records) == 1
    assert result.pixel_records[0].detector == "ela"
    assert result.pixel_records[0].entry_path == "img.png"
    assert result.pixel_records[0].level == "clean"


def test_run_worker_task_computes_pixel_metrics_at_a_geometry_preserving_level(
    tmp_path: Path,
) -> None:
    """A JPEG re-encode leaves every pixel where it was, so the stored mask still
    lines up with the heatmap and the record is tagged with that level.
    """
    entry = _masked_image(tmp_path, seed=4)
    jpeg_level = Perturbation(name="jpeg_q75", kind="jpeg", params={"quality": 75})
    task = WorkerTask(root=tmp_path, entry=entry, perturbation=jpeg_level, detector_names=("ela",))

    result = run_worker_task(task)

    assert len(result.pixel_records) == 1
    assert result.pixel_records[0].level == "jpeg_q75"


def test_run_worker_task_skips_pixel_metrics_when_the_level_moves_pixels(tmp_path: Path) -> None:
    entry = _masked_image(tmp_path, seed=7)
    resize_level = Perturbation(name="resize_0.5", kind="resize", params={"scale": 0.5})
    task = WorkerTask(
        root=tmp_path, entry=entry, perturbation=resize_level, detector_names=("ela",)
    )

    result = run_worker_task(task)

    assert len(result.records) == 1  # the score record is unaffected
    assert result.pixel_records == []


def test_run_worker_task_reports_missing_mask_without_dropping_the_score(tmp_path: Path) -> None:
    natural_like_image(size=(64, 64), seed=5).save(tmp_path / "img.png", format="PNG")
    entry = _masked_png_entry("does_not_exist.png")
    task = WorkerTask(root=tmp_path, entry=entry, perturbation=_CLEAN, detector_names=("ela",))

    result = run_worker_task(task)

    assert len(result.records) == 1  # the score record is unaffected
    assert result.pixel_records == []
    assert len(result.missing_files) == 1
    assert "does_not_exist" in result.missing_files[0]


def test_run_worker_task_reports_missing_image(tmp_path: Path) -> None:
    entry = _entry(path="does_not_exist.jpg")
    task = WorkerTask(root=tmp_path, entry=entry, perturbation=_CLEAN, detector_names=("metadata",))

    result = run_worker_task(task)

    assert result.records == []
    assert result.pixel_records == []
    assert len(result.missing_files) == 1
    assert "does_not_exist" in result.missing_files[0]


def test_worker_detector_is_cached_across_tasks(tmp_path: Path) -> None:
    workers_module._DETECTOR_CACHE.clear()
    natural_like_image(size=(32, 32), seed=6).save(tmp_path / "img.jpg", format="JPEG", quality=90)
    entry = _entry(path="img.jpg")
    task = WorkerTask(root=tmp_path, entry=entry, perturbation=_CLEAN, detector_names=("metadata",))

    run_worker_task(task)
    first = workers_module._DETECTOR_CACHE["metadata"]
    run_worker_task(task)
    second = workers_module._DETECTOR_CACHE["metadata"]

    assert first is second
