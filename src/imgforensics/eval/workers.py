"""Process-pool worker for the classical *signal* detectors in the benchmark runner.

:func:`imgforensics.eval.runner.run_benchmark` can evaluate the classical
signal detectors (:data:`imgforensics.signals.SIGNAL_NAMES`, plus the
``constant_real``/``constant_fake``/``random`` baselines) in a
``concurrent.futures.ProcessPoolExecutor`` so they no longer serialize with a
slower, GPU-based detector running in the same process. Every other
detector -- a learned detector such as ``dinov2_head``, or ``signals_mean``,
which needs to run every signal itself -- stays in the main process; see
:func:`is_worker_eligible`.

Windows only supports the ``spawn`` multiprocessing start method, which
re-imports this module in each worker process and calls the target function
by qualified name -- no closures or lambdas can cross that boundary, so
:func:`run_worker_task` and every helper it calls are plain, module-level
functions. :data:`_DETECTOR_CACHE` builds each assigned detector once per
worker process (not once per task), keyed by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from imgforensics.core import registry
from imgforensics.core.base import BaseDetector
from imgforensics.core.image import ForensicImage
from imgforensics.data.manifest import ManifestEntry
from imgforensics.eval.baselines import ConstantDetector, RandomDetector
from imgforensics.eval.metrics import PixelMetrics
from imgforensics.eval.records import PixelRecord, ScoreRecord, _load_mask
from imgforensics.eval.robustness import Perturbation, RobustnessSuite
from imgforensics.signals import SIGNAL_NAMES  # side effect: registers every signal detector

#: Baseline detectors cheap and torch-free enough to run in a worker process.
#: ``signals_mean`` is deliberately excluded: it must iterate every signal
#: itself in one place, so it stays in the main process alongside any learned
#: detector (see the module docstring).
_WORKER_BASELINE_NAMES: frozenset[str] = frozenset({"constant_real", "constant_fake", "random"})

_EMPTY_SUITE = RobustnessSuite()

#: Built lazily, once per worker process, keyed by detector name.
_DETECTOR_CACHE: dict[str, BaseDetector] = {}


def is_worker_eligible(name: str) -> bool:
    """Whether detector ``name`` may run in a worker process.

    True for every classical signal (:data:`imgforensics.signals.SIGNAL_NAMES`)
    and the three cheap baselines; false for everything else, which includes
    ``signals_mean`` and any learned detector (``dinov2_head`` and any
    detector registered only for a test), so those always stay in the main
    process exactly as before this feature existed.
    """
    return name in SIGNAL_NAMES or name in _WORKER_BASELINE_NAMES


def _build_worker_detector(name: str) -> BaseDetector:
    if name == "constant_real":
        return ConstantDetector(0.0)
    if name == "constant_fake":
        return ConstantDetector(1.0)
    if name == "random":
        return RandomDetector(seed=0)
    return registry.get(name)()


def _get_worker_detector(name: str) -> BaseDetector:
    """Return this process's cached instance of detector ``name``, building it on first use."""
    instance = _DETECTOR_CACHE.get(name)
    if instance is None:
        instance = _build_worker_detector(name)
        instance.load()
        _DETECTOR_CACHE[name] = instance
    return instance


@dataclass
class WorkerTask:
    """One (entry, robustness level) work unit dispatched to a worker process."""

    root: Path
    entry: ManifestEntry
    perturbation: Perturbation
    detector_names: tuple[str, ...]


@dataclass
class WorkerResult:
    """What :func:`run_worker_task` returns for one :class:`WorkerTask`."""

    records: list[ScoreRecord] = field(default_factory=list)
    pixel_records: list[PixelRecord] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)


def run_worker_task(task: WorkerTask) -> WorkerResult:
    """Load ``task.entry``'s image, apply ``task.perturbation``, and score it.

    Runs every detector in ``task.detector_names`` (built and cached per
    worker process by :func:`_get_worker_detector`) against the perturbed
    image and returns plain, picklable :class:`~imgforensics.eval.records.ScoreRecord`
    values. Heatmaps are dropped except at the clean level for an entry that
    carries a ``mask_path``, in which case
    :class:`~imgforensics.eval.metrics.PixelMetrics` are computed here (in the
    worker) and returned as :class:`~imgforensics.eval.records.PixelRecord` values.

    A missing image is reported the same way the sequential runner reports
    one -- an entry appended to ``missing_files``, no records -- though in
    practice the main process already filters those out before dispatching
    any task, since one missing image would otherwise fail once per
    robustness level instead of once per entry.
    """
    entry = task.entry
    perturbation = task.perturbation

    try:
        forensic_image = ForensicImage.from_path(task.root / entry.path)
    except OSError as exc:
        return WorkerResult(missing_files=[f"{entry.path}: {exc}"])

    perturbed = _EMPTY_SUITE.apply(perturbation, forensic_image)

    records: list[ScoreRecord] = []
    pixel_records: list[PixelRecord] = []
    missing_files: list[str] = []

    for name in task.detector_names:
        instance = _get_worker_detector(name)
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
        heatmap = result.heatmap
        if perturbation.kind == "clean" and entry.mask_path is not None and heatmap is not None:
            mask_path = task.root / entry.mask_path
            try:
                mask = _load_mask(mask_path, heatmap.shape)
            except OSError as exc:
                missing_files.append(f"{entry.mask_path}: {exc}")
                continue
            metrics = PixelMetrics.compute(mask, heatmap)
            pixel_records.append(
                PixelRecord(entry_path=entry.path, detector=instance.name, metrics=metrics)
            )

    return WorkerResult(records=records, pixel_records=pixel_records, missing_files=missing_files)
