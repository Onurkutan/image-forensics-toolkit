"""Benchmark runner: evaluate registered detectors over a manifest and produce
Markdown-ready tables (image-level metrics, a robustness table, per-group AUC,
pixel metrics, and timing).

See ``docs/ROADMAP.md``, section 3 ("honest evaluation"): a threshold fixed on
validation, a robustness table, and trivial baselines belong in every
detector comparison. This module wires the manifest loader
(:mod:`imgforensics.data.manifest`), the metric functions
(:mod:`imgforensics.eval.metrics`), and the robustness suite
(:mod:`imgforensics.eval.robustness`) together into one entry point,
:func:`run_benchmark`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel

import imgforensics.signals  # noqa: F401  (side effect: registers every signal detector)
from imgforensics.core import registry
from imgforensics.core.base import BaseDetector
from imgforensics.core.image import ForensicImage
from imgforensics.data.manifest import Manifest, ManifestEntry
from imgforensics.eval.baselines import baseline_detectors
from imgforensics.eval.metrics import (
    ImageMetrics,
    PixelMetrics,
    balanced_accuracy_at_threshold,
    best_threshold,
    roc_auc,
)
from imgforensics.eval.robustness import Perturbation, RobustnessSuite

_SHUFFLE_SEED = 0
_CLEAN_LEVEL_NAME = "clean"
_IN_SAMPLE_CAVEAT = "threshold tuned in-sample"

# ---------------------------------------------------------------------------
# Config and records
# ---------------------------------------------------------------------------


class BenchmarkConfig(BaseModel):
    """What to run: which detectors, whether to include the baselines, and how.

    ``robustness`` is the suite to evaluate every detector under; ``None``
    means "clean level only" (no perturbations). ``limit`` takes the first
    ``N`` manifest entries after a seeded (seed 0) shuffle, so a quick
    ``--limit 20`` run is deterministic and reproducible across invocations.
    ``threshold_split`` chooses how :func:`run_benchmark` tunes each
    detector's operating threshold: ``"val"`` (default) uses the evaluated
    entries' val split when one is present, falling back to in-sample tuning
    otherwise; ``"clean"`` always tunes in-sample, even when a val split
    exists (useful for a quick iteration run where the val split itself is
    also being scored).
    """

    detectors: list[str] = []
    include_baselines: bool = False
    robustness: RobustnessSuite | None = None
    limit: int | None = None
    fixed_threshold: float = 0.5
    threshold_split: Literal["val", "clean"] = "val"


@dataclass
class ScoreRecord:
    """One (image, robustness level, detector) score."""

    entry_path: str
    label: str
    source: str
    generator: str | None
    split: str | None
    level: str
    detector: str
    score: float
    elapsed_ms: float | None


@dataclass
class PixelRecord:
    """One (image, detector) pixel-level metric bundle, computed at the clean level."""

    entry_path: str
    detector: str
    metrics: PixelMetrics


# ---------------------------------------------------------------------------
# Table rows
# ---------------------------------------------------------------------------


@dataclass
class DetectorImageRow:
    """One detector's image-level metrics at both operating thresholds."""

    detector: str
    fixed: ImageMetrics | None
    tuned: ImageMetrics | None
    note: str | None = None


@dataclass
class RobustnessRow:
    """One (detector, level) AUC / balanced-accuracy pair at the fixed threshold."""

    detector: str
    level: str
    auc: float | None
    balanced_accuracy: float | None
    note: str | None = None


@dataclass
class GroupRow:
    """One (detector, group) AUC on the clean level."""

    detector: str
    group: str
    auc: float | None
    note: str | None = None


@dataclass
class PixelRow:
    """One detector's mean pixel metrics, averaged over every image with a mask."""

    detector: str
    n: int
    mean_f1_at_threshold: float
    mean_best_f1: float
    mean_ap: float
    mean_iou: float


@dataclass
class TimingRow:
    """One detector's mean wall-clock time per call."""

    detector: str
    n: int
    mean_elapsed_ms: float | None


def _y_true(records: list[ScoreRecord]) -> np.ndarray:
    return np.array([1 if r.label == "fake" else 0 for r in records], dtype=np.int64)


def _scores(records: list[ScoreRecord]) -> np.ndarray:
    return np.array([r.score for r in records], dtype=np.float64)


def _group_value(record: ScoreRecord, key: str) -> str:
    if key == "generator":
        return record.generator or "none"
    if key == "source":
        return record.source
    raise ValueError(f"Unknown group key {key!r}; expected 'generator' or 'source'")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Every score and pixel record from one :func:`run_benchmark` call, plus the
    small amount of run metadata (manifest name, thresholds, levels) needed to
    render tables. Tables are computed on demand from ``records`` /
    ``pixel_records`` rather than cached, so :meth:`save_json` / :meth:`load_json`
    round-trip without losing anything a table method needs.
    """

    manifest_name: str
    entry_count: int
    levels: list[str]
    fixed_threshold: float
    tuned_thresholds: dict[str, float]
    threshold_caveat: str | None
    missing_files: list[str] = field(default_factory=list)
    records: list[ScoreRecord] = field(default_factory=list)
    pixel_records: list[PixelRecord] = field(default_factory=list)

    # -- tables --------------------------------------------------------

    def image_table(self, level: str = _CLEAN_LEVEL_NAME) -> list[DetectorImageRow]:
        """Per-detector :class:`~imgforensics.eval.metrics.ImageMetrics` at ``level``,
        at both the fixed and tuned thresholds.

        A detector is skipped (``note`` set, ``fixed``/``tuned`` left
        ``None``) when it has no records at ``level``, or when only one
        label is present there (AUC is undefined).
        """
        rows: list[DetectorImageRow] = []
        for detector in sorted({r.detector for r in self.records}):
            level_records = [r for r in self.records if r.detector == detector and r.level == level]
            if not level_records:
                rows.append(
                    DetectorImageRow(detector, None, None, note=f"no records at level {level!r}")
                )
                continue
            y_true = _y_true(level_records)
            if len(np.unique(y_true)) < 2:
                rows.append(
                    DetectorImageRow(
                        detector,
                        None,
                        None,
                        note="only one label present at this level; AUC undefined",
                    )
                )
                continue
            scores = _scores(level_records)
            fixed = ImageMetrics.compute(y_true, scores, threshold=self.fixed_threshold)
            tuned_threshold = self.tuned_thresholds.get(detector, self.fixed_threshold)
            tuned = ImageMetrics.compute(y_true, scores, threshold=tuned_threshold)
            rows.append(DetectorImageRow(detector, fixed, tuned))
        return rows

    def robustness_table(self) -> list[RobustnessRow]:
        """AUC and balanced accuracy at the fixed threshold, per detector per level."""
        rows: list[RobustnessRow] = []
        detectors = sorted({r.detector for r in self.records})
        for detector in detectors:
            for level in self.levels:
                level_records = [
                    r for r in self.records if r.detector == detector and r.level == level
                ]
                if not level_records:
                    rows.append(RobustnessRow(detector, level, None, None, note="no records"))
                    continue
                y_true = _y_true(level_records)
                if len(np.unique(y_true)) < 2:
                    rows.append(
                        RobustnessRow(
                            detector,
                            level,
                            None,
                            None,
                            note="only one label present; AUC undefined",
                        )
                    )
                    continue
                scores = _scores(level_records)
                auc = roc_auc(y_true, scores)
                balanced_accuracy = balanced_accuracy_at_threshold(
                    y_true, scores, self.fixed_threshold
                )
                rows.append(RobustnessRow(detector, level, auc, balanced_accuracy))
        return rows

    def per_group_table(self, key: str = "generator") -> list[GroupRow]:
        """AUC per group (``"generator"`` or ``"source"``) on the clean level."""
        rows: list[GroupRow] = []
        clean_records = [r for r in self.records if r.level == _CLEAN_LEVEL_NAME]
        for detector in sorted({r.detector for r in clean_records}):
            detector_records = [r for r in clean_records if r.detector == detector]
            groups = sorted({_group_value(r, key) for r in detector_records})
            for group in groups:
                group_records = [r for r in detector_records if _group_value(r, key) == group]
                y_true = _y_true(group_records)
                if len(np.unique(y_true)) < 2:
                    rows.append(
                        GroupRow(
                            detector, group, None, note="only one label present; AUC undefined"
                        )
                    )
                    continue
                auc = roc_auc(y_true, _scores(group_records))
                rows.append(GroupRow(detector, group, auc))
        return rows

    def pixel_table(self) -> list[PixelRow] | None:
        """Mean F1@0.5, best-F1, AP, IoU per detector, or ``None`` with no pixel records."""
        if not self.pixel_records:
            return None
        rows: list[PixelRow] = []
        for detector in sorted({r.detector for r in self.pixel_records}):
            detector_records = [r for r in self.pixel_records if r.detector == detector]
            rows.append(
                PixelRow(
                    detector=detector,
                    n=len(detector_records),
                    mean_f1_at_threshold=float(
                        np.mean([r.metrics.f1_at_threshold for r in detector_records])
                    ),
                    mean_best_f1=float(np.mean([r.metrics.best_f1 for r in detector_records])),
                    mean_ap=float(np.mean([r.metrics.ap for r in detector_records])),
                    mean_iou=float(np.mean([r.metrics.iou for r in detector_records])),
                )
            )
        return rows

    def timing_table(self) -> list[TimingRow]:
        """Mean elapsed milliseconds per detector, over every scored record."""
        rows: list[TimingRow] = []
        for detector in sorted({r.detector for r in self.records}):
            timed = [
                r.elapsed_ms
                for r in self.records
                if r.detector == detector and r.elapsed_ms is not None
            ]
            rows.append(
                TimingRow(
                    detector=detector,
                    n=len(timed),
                    mean_elapsed_ms=float(np.mean(timed)) if timed else None,
                )
            )
        return rows

    # -- report ----------------------------------------------------------

    def to_markdown(self) -> str:
        """Render every table as one Markdown document."""
        lines = [f"# Benchmark report: {self.manifest_name}", ""]
        lines.append(f"- Entries evaluated: {self.entry_count}")
        lines.append(f"- Robustness levels: {', '.join(self.levels)}")
        lines.append(f"- Fixed threshold: {self.fixed_threshold:g}")
        if self.threshold_caveat:
            lines.append(f"- Threshold caveat: {self.threshold_caveat}")
        if self.missing_files:
            lines.append(f"- Missing files: {len(self.missing_files)}")
        lines.append("")

        lines.append(f"## Image metrics ({_CLEAN_LEVEL_NAME})")
        lines.append("")
        lines.append(
            "| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr "
            "| ece | brier |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for image_row in self.image_table():
            if image_row.note or image_row.fixed is None or image_row.tuned is None:
                note = image_row.note or "skipped"
                lines.append(f"| {image_row.detector} | - | {note} | | | | | | | |")
                continue
            for label, metrics in (("fixed", image_row.fixed), ("tuned", image_row.tuned)):
                lines.append(
                    f"| {image_row.detector} | {label} | {metrics.auc:.3f} | {metrics.ap:.3f} | "
                    f"{metrics.accuracy:.3f} | {metrics.balanced_accuracy:.3f} | "
                    f"{metrics.fpr:.3f} | {metrics.tpr:.3f} | {metrics.ece:.3f} | "
                    f"{metrics.brier:.3f} |"
                )
        lines.append("")

        lines.append("## Robustness (AUC / balanced accuracy at the fixed threshold)")
        lines.append("")
        lines.append("| detector | level | auc | balanced_accuracy |")
        lines.append("|---|---|---|---|")
        for robustness_row in self.robustness_table():
            if robustness_row.auc is None or robustness_row.balanced_accuracy is None:
                note = robustness_row.note or "skipped"
                lines.append(f"| {robustness_row.detector} | {robustness_row.level} | - | {note} |")
            else:
                lines.append(
                    f"| {robustness_row.detector} | {robustness_row.level} | "
                    f"{robustness_row.auc:.3f} | {robustness_row.balanced_accuracy:.3f} |"
                )
        lines.append("")

        for key in ("generator", "source"):
            group_rows = self.per_group_table(key=key)
            if not group_rows:
                continue
            lines.append(f"## Per-{key} AUC ({_CLEAN_LEVEL_NAME})")
            lines.append("")
            lines.append(f"| detector | {key} | auc |")
            lines.append("|---|---|---|")
            for group_row in group_rows:
                value = (
                    f"{group_row.auc:.3f}" if group_row.auc is not None else (group_row.note or "-")
                )
                lines.append(f"| {group_row.detector} | {group_row.group} | {value} |")
            lines.append("")

        pixel_rows = self.pixel_table()
        if pixel_rows:
            lines.append("## Pixel metrics")
            lines.append("")
            lines.append("| detector | n | mean f1@0.5 | mean best-f1 | mean ap | mean iou |")
            lines.append("|---|---|---|---|---|---|")
            for prow in pixel_rows:
                lines.append(
                    f"| {prow.detector} | {prow.n} | {prow.mean_f1_at_threshold:.3f} | "
                    f"{prow.mean_best_f1:.3f} | {prow.mean_ap:.3f} | {prow.mean_iou:.3f} |"
                )
            lines.append("")

        lines.append("## Timing")
        lines.append("")
        lines.append("| detector | n | mean elapsed_ms |")
        lines.append("|---|---|---|")
        for trow in self.timing_table():
            ms = f"{trow.mean_elapsed_ms:.2f}" if trow.mean_elapsed_ms is not None else "-"
            lines.append(f"| {trow.detector} | {trow.n} | {ms} |")
        lines.append("")

        return "\n".join(lines)

    # -- persistence -------------------------------------------------------

    def save_json(self, path: str | Path) -> None:
        """Write every record (plus the run metadata tables need) as JSON."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "manifest_name": self.manifest_name,
            "entry_count": self.entry_count,
            "levels": self.levels,
            "fixed_threshold": self.fixed_threshold,
            "tuned_thresholds": self.tuned_thresholds,
            "threshold_caveat": self.threshold_caveat,
            "missing_files": self.missing_files,
            "records": [asdict(r) for r in self.records],
            "pixel_records": [
                {"entry_path": r.entry_path, "detector": r.detector, "metrics": asdict(r.metrics)}
                for r in self.pixel_records
            ],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> BenchmarkResult:
        """Load a result previously written by :meth:`save_json`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        records = [ScoreRecord(**r) for r in data["records"]]
        pixel_records = [
            PixelRecord(
                entry_path=r["entry_path"],
                detector=r["detector"],
                metrics=PixelMetrics(**r["metrics"]),
            )
            for r in data["pixel_records"]
        ]
        return cls(
            manifest_name=data["manifest_name"],
            entry_count=data["entry_count"],
            levels=data["levels"],
            fixed_threshold=data["fixed_threshold"],
            tuned_thresholds=data["tuned_thresholds"],
            threshold_caveat=data["threshold_caveat"],
            missing_files=data["missing_files"],
            records=records,
            pixel_records=pixel_records,
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _build_detectors(config: BenchmarkConfig) -> list[BaseDetector]:
    instances: list[BaseDetector] = []
    for name in config.detectors:
        detector_cls = registry.get(name)
        instance = detector_cls()
        instance.load()
        instances.append(instance)
    if config.include_baselines:
        for instance in baseline_detectors():
            instance.load()
            instances.append(instance)
    if not instances:
        raise ValueError(
            "no detectors selected: pass config.detectors and/or config.include_baselines=True"
        )
    return instances


def _shuffled_entries(entries: list[ManifestEntry], limit: int | None) -> list[ManifestEntry]:
    order = np.random.default_rng(_SHUFFLE_SEED).permutation(len(entries))
    shuffled = [entries[i] for i in order]
    return shuffled if limit is None else shuffled[:limit]


def _load_mask(mask_path: Path, heatmap_shape: tuple[int, int]) -> np.ndarray:
    with Image.open(mask_path) as mask_img:
        mask = np.asarray(mask_img.convert("L"), dtype=np.uint8) > 127
    if mask.shape != heatmap_shape:
        resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (heatmap_shape[1], heatmap_shape[0]), Image.Resampling.NEAREST
        )
        mask = np.asarray(resized) > 127
    return mask


def _tune_threshold(
    records: list[ScoreRecord], detector: str, level: str, split: str | None, fixed_threshold: float
) -> float:
    pool = [
        r
        for r in records
        if r.detector == detector and r.level == level and (split is None or r.split == split)
    ]
    if not pool:
        return fixed_threshold
    y_true = _y_true(pool)
    if len(np.unique(y_true)) < 2:
        return fixed_threshold
    threshold, _ = best_threshold(y_true, _scores(pool))
    return threshold


def run_benchmark(
    manifest: Manifest,
    config: BenchmarkConfig,
    *,
    root: Path | None = None,
    progress: bool = True,
) -> BenchmarkResult:
    """Run every configured detector over ``manifest``, at every robustness level.

    Images are resolved against ``root`` (defaulting to
    ``manifest.meta.root``) and loaded with :meth:`ForensicImage.from_path`.
    A missing image or mask file is recorded in the result's
    ``missing_files`` and skipped -- it does not abort the run. Pixel metrics
    are computed only at the clean level (the only level where a heatmap and
    the manifest's stored mask share the same geometry without needing to
    warp the mask through the perturbation).

    See :class:`BenchmarkConfig` for ``limit``/threshold-tuning behaviour.
    """
    resolved_root = Path(root) if root is not None else Path(manifest.meta.root)
    detectors = _build_detectors(config)
    suite = config.robustness or RobustnessSuite(
        levels=[Perturbation(name=_CLEAN_LEVEL_NAME, kind="clean", params={})]
    )
    level_names = [level.name for level in suite.levels]

    entries = _shuffled_entries(manifest.entries, config.limit)

    records: list[ScoreRecord] = []
    pixel_records: list[PixelRecord] = []
    missing_files: list[str] = []

    total = len(entries)
    for index, entry in enumerate(entries, start=1):
        if progress and (index % 20 == 0 or index == total):
            print(f"[benchmark] {index}/{total} entries")

        try:
            forensic_image = ForensicImage.from_path(resolved_root / entry.path)
        except OSError as exc:
            missing_files.append(f"{entry.path}: {exc}")
            continue

        for perturbation in suite.levels:
            perturbed = suite.apply(perturbation, forensic_image)
            for instance in detectors:
                result = instance.run(perturbed)
                records.append(
                    ScoreRecord(
                        entry_path=entry.path,
                        label=entry.label,
                        source=entry.source,
                        generator=entry.generator,
                        split=entry.split,
                        level=perturbation.name,
                        detector=instance.name,
                        score=result.score,
                        elapsed_ms=result.elapsed_ms,
                    )
                )
                if (
                    perturbation.kind == "clean"
                    and entry.mask_path is not None
                    and result.heatmap is not None
                ):
                    mask_path = resolved_root / entry.mask_path
                    try:
                        mask = _load_mask(mask_path, result.heatmap.shape)
                    except OSError as exc:
                        missing_files.append(f"{entry.mask_path}: {exc}")
                        continue
                    metrics = PixelMetrics.compute(mask, result.heatmap)
                    pixel_records.append(
                        PixelRecord(entry_path=entry.path, detector=instance.name, metrics=metrics)
                    )

    has_val_split = config.threshold_split == "val" and any(
        r.level == _CLEAN_LEVEL_NAME and r.split == "val" for r in records
    )
    tuned_thresholds = {
        detector.name: _tune_threshold(
            records,
            detector.name,
            _CLEAN_LEVEL_NAME,
            "val" if has_val_split else None,
            config.fixed_threshold,
        )
        for detector in detectors
    }
    threshold_caveat = None if has_val_split else _IN_SAMPLE_CAVEAT

    return BenchmarkResult(
        manifest_name=manifest.meta.dataset,
        entry_count=total,
        levels=level_names,
        fixed_threshold=config.fixed_threshold,
        tuned_thresholds=tuned_thresholds,
        threshold_caveat=threshold_caveat,
        missing_files=missing_files,
        records=records,
        pixel_records=pixel_records,
    )
