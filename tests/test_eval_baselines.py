"""Tests for imgforensics.eval.baselines: trivial detectors kept out of the global registry."""

from __future__ import annotations

from conftest import natural_like_image

from imgforensics.core import registry
from imgforensics.core.image import ForensicImage
from imgforensics.eval.baselines import (
    ConstantDetector,
    RandomDetector,
    SignalsMeanDetector,
    baseline_detectors,
)
from imgforensics.signals import SIGNAL_NAMES


def test_baselines_are_not_in_the_global_registry() -> None:
    names = set(registry.available())
    assert "constant_real" not in names
    assert "constant_fake" not in names
    assert "random" not in names
    assert "signals_mean" not in names


def test_constant_detector_names_and_scores() -> None:
    real = ConstantDetector(0.0)
    fake = ConstantDetector(1.0)
    assert real.name == "constant_real"
    assert fake.name == "constant_fake"

    image = ForensicImage.from_pil(natural_like_image(size=(48, 48), seed=1))
    assert real.run(image).score == 0.0
    assert fake.run(image).score == 1.0


def test_random_detector_is_seeded_per_image() -> None:
    detector = RandomDetector(seed=0)
    image_a = ForensicImage.from_pil(natural_like_image(size=(48, 48), seed=1))
    image_b = ForensicImage.from_pil(natural_like_image(size=(48, 48), seed=2))

    score_a1 = detector.predict(image_a).score
    score_a2 = detector.predict(image_a).score
    score_b = detector.predict(image_b).score

    assert 0.0 <= score_a1 <= 1.0
    assert score_a1 == score_a2  # deterministic given the same image and seed
    assert score_a1 != score_b


def test_signals_mean_detector_averages_every_registered_signal() -> None:
    detector = SignalsMeanDetector()
    image = ForensicImage.from_pil(natural_like_image(size=(48, 48), seed=1))

    result = detector.predict(image)

    per_signal = result.details["per_signal"]
    # Every classical signal, and nothing else: the registry may also hold the
    # learned detector when the optional ml extra is installed, and a baseline
    # that absorbed it would no longer be the floor it is meant to be.
    assert set(per_signal) == set(SIGNAL_NAMES)
    assert set(SIGNAL_NAMES) <= set(registry.available())
    assert result.score == sum(per_signal.values()) / len(per_signal)


def test_baseline_detectors_returns_all_four() -> None:
    detectors = baseline_detectors()
    names = {detector.name for detector in detectors}
    assert names == {"constant_real", "constant_fake", "random", "signals_mean"}
