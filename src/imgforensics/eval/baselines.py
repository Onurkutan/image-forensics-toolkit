"""Trivial baseline detectors for the benchmark runner.

``docs/ROADMAP.md``, section 3 ("honest evaluation"): every benchmark table
should include trivial baselines, so a learned detector's numbers can be read
against a floor rather than in isolation. These classes are deliberately
*not* registered with :mod:`imgforensics.core.registry` (no ``@register``
decorator) -- ``imgforensics analyze`` iterates the registry, and a user
asking to "analyze this image" should never see "constant_fake: 1.00" in the
results. They are only reachable by name through
:func:`baseline_detectors` / :mod:`imgforensics.eval.runner`.
"""

from __future__ import annotations

import hashlib

import numpy as np

import imgforensics.signals  # noqa: F401  (side effect: registers every signal detector)
from imgforensics.core import registry
from imgforensics.core.base import BaseDetector
from imgforensics.core.image import ForensicImage
from imgforensics.core.types import DetectionResult, label_from_score
from imgforensics.utils.image_io import to_numpy


class ConstantDetector(BaseDetector):
    """Ignores the image and always returns the same score.

    ``ConstantDetector(0.0)`` is named ``"constant_real"``, ``ConstantDetector(1.0)``
    is named ``"constant_fake"`` -- any other score falls back to a
    ``"constant_<score>"`` name. An AUC of 0.5 against either one on a
    balanced evaluation set is the expected floor.
    """

    def __init__(self, score: float) -> None:
        self.score = score
        if score == 0.0:
            self.name = "constant_real"
        elif score == 1.0:
            self.name = "constant_fake"
        else:
            self.name = f"constant_{score:g}"

    def predict(self, image: ForensicImage) -> DetectionResult:
        return DetectionResult(
            detector=self.name,
            score=self.score,
            label=label_from_score(self.score),
            details={"note": "baseline: ignores the image, always returns a constant score"},
        )


class RandomDetector(BaseDetector):
    """Returns a score drawn uniformly from [0, 1], seeded per image.

    The draw is seeded from ``sha256(image bytes + str(seed))`` so repeated
    runs over the same image and ``seed`` are reproducible, while different
    images get independent draws.
    """

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def predict(self, image: ForensicImage) -> DetectionResult:
        basis = image.raw if image.raw is not None else to_numpy(image.rgb).tobytes()
        digest = hashlib.sha256(basis + str(self.seed).encode("utf-8")).digest()
        per_image_seed = int.from_bytes(digest[:8], byteorder="big")
        score = float(np.random.default_rng(per_image_seed).uniform(0.0, 1.0))
        return DetectionResult(
            detector=self.name,
            score=score,
            label=label_from_score(score),
            details={"seed": self.seed},
        )


class SignalsMeanDetector(BaseDetector):
    """Runs every registered signal detector and returns the mean of their scores.

    ``details["per_signal"]`` carries the individual score each signal
    produced, so a mean-score outlier can be traced back to its source.
    Uses :meth:`BaseDetector.predict` (not :meth:`~BaseDetector.run`) on each
    signal, so this detector's own ``elapsed_ms`` (filled in by ``.run()`` at
    the call site) reflects the whole ensemble.
    """

    name = "signals_mean"

    def predict(self, image: ForensicImage) -> DetectionResult:
        per_signal: dict[str, float] = {}
        for signal_name in registry.available():
            detector_cls = registry.get(signal_name)
            instance = detector_cls()
            instance.load()
            per_signal[signal_name] = instance.predict(image).score

        mean_score = float(np.mean(list(per_signal.values()))) if per_signal else 0.5
        return DetectionResult(
            detector=self.name,
            score=mean_score,
            label=label_from_score(mean_score),
            details={"per_signal": per_signal},
        )


def baseline_detectors() -> list[BaseDetector]:
    """Every baseline: two constants, the random baseline, and the signals-mean ensemble."""
    return [
        ConstantDetector(0.0),
        ConstantDetector(1.0),
        RandomDetector(seed=0),
        SignalsMeanDetector(),
    ]
