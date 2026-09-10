"""Shared pytest fixtures/helpers for signal tests.

``natural_like_image`` is the one thing every Phase 1c signal test needs: a
synthetic image with actual texture (smooth low-frequency structure, pixel
noise, and a few hard-edged shapes) so block matching and JPEG re-encoding
behave the way they would on a real photo. A flat or pure-noise image would
either match every block trivially (flat) or none at all (pure noise), and
JPEG quantization needs some low-frequency content to have anything to
quantize.

``synthetic_fusion_records`` is the fusion tests' equivalent: a set of
:class:`~imgforensics.eval.records.ScoreRecord` for a fixed number of
synthetic images and named detectors with a controllable score/label
relationship, so a stacking fuser can be fitted and its learned weights
checked without running any real detector or ML model.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image, ImageDraw

from imgforensics.eval.records import ScoreRecord


def synthetic_fusion_records(
    detector_scores: dict[str, Callable[[float, np.random.Generator], float]] | None = None,
    *,
    n_per_class: int = 200,
    level: str = "clean",
    source: str = "synthetic",
    seed: int = 0,
) -> list[ScoreRecord]:
    """Build ``2 * n_per_class`` images' worth of :class:`ScoreRecord` for named detectors.

    Args:
        detector_scores: ``{detector name: fn(true_label_as_float, rng) -> score}``.
            Defaults to three detectors that exercise the three cases a
            stacking fuser has to handle: ``"informative"`` (correlates with
            the true label), ``"inverted"`` (anti-correlates), and
            ``"abstaining"`` (always 0.5, carries no information).
        n_per_class: Number of real images and of fake images (so
            ``2 * n_per_class`` images total).
        level: Robustness level recorded on every row.
        source: Source name recorded on every row.
        seed: Seed for the score noise.
    """
    if detector_scores is None:
        detector_scores = {
            "informative": lambda y, rng: float(
                np.clip(y * 0.9 + 0.05 + rng.normal(0, 0.05), 1e-3, 1 - 1e-3)
            ),
            "inverted": lambda y, rng: float(
                np.clip((1 - y) * 0.9 + 0.05 + rng.normal(0, 0.05), 1e-3, 1 - 1e-3)
            ),
            "abstaining": lambda _y, _rng: 0.5,
        }

    rng = np.random.default_rng(seed)
    records: list[ScoreRecord] = []
    labels = ["real"] * n_per_class + ["fake"] * n_per_class
    for index, label in enumerate(labels):
        entry_path = f"synthetic/{label}/{index:04d}.png"
        y = 1.0 if label == "fake" else 0.0
        for detector, score_fn in detector_scores.items():
            records.append(
                ScoreRecord(
                    entry_path=entry_path,
                    label=label,
                    source=source,
                    generator=None,
                    split=None,
                    level=level,
                    detector=detector,
                    score=score_fn(y, rng),
                    elapsed_ms=None,
                )
            )
    return records


def natural_like_image(size: tuple[int, int] = (512, 512), seed: int = 0) -> Image.Image:
    """Build a deterministic, textured RGB test image.

    Composed of a handful of low-frequency sinusoids per channel (different
    frequencies/phases so channels are not perfectly correlated), additive
    Gaussian pixel noise, and a few solid-colour rectangles/circles scattered
    on top (so there are hard edges as well as smooth gradients -- closer to
    a real photo than either a flat image or pure noise).
    """
    width, height = size
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)

    channels = []
    for _ in range(3):
        channel = np.zeros((height, width), dtype=np.float32)
        for _ in range(4):
            freq_x = rng.uniform(0.5, 4.0)
            freq_y = rng.uniform(0.5, 4.0)
            phase = rng.uniform(0, 2 * np.pi)
            amplitude = rng.uniform(15, 40)
            channel += amplitude * np.sin(
                2 * np.pi * freq_x * x / width + 2 * np.pi * freq_y * y / height + phase
            )
        channels.append(channel + 128.0)
    array = np.stack(channels, axis=2)
    array += rng.normal(0, 8, array.shape).astype(np.float32)
    array = np.clip(array, 0, 255).astype(np.uint8)

    image = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(image)
    for _ in range(6):
        color = tuple(int(v) for v in rng.integers(0, 256, size=3))
        box_w = int(rng.uniform(20, max(21, min(width, height) // 4)))
        box_h = int(rng.uniform(20, max(21, min(width, height) // 4)))
        x0 = int(rng.uniform(0, max(1, width - box_w)))
        y0 = int(rng.uniform(0, max(1, height - box_h)))
        if rng.uniform() < 0.5:
            draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=color)
        else:
            draw.ellipse([x0, y0, x0 + box_w, y0 + box_h], fill=color)

    return image
