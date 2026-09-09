"""The learned whole-image detector: a trained head over a frozen backbone.

Registered as ``dinov2_head``, so it runs inside ``imgforensics analyze`` and
the benchmark runner next to the classical signals. The registry name is
fixed even though the backbone is not: which frozen space the head reads is
recorded in the checkpoint and read back from it, so retraining the same
detector on CLIP features does not change the name users type.

**Where the checkpoint comes from**, in order: the ``checkpoint_dir`` passed
to the constructor, else the ``IMGFORENSICS_HEAD_DIR`` environment variable,
else ``weights/dinov2_head/`` (gitignored -- weights are never committed; see
``docs/ROADMAP.md``, section 7).

**With no checkpoint, the detector abstains** rather than failing: it returns
0.5 / ``"uncertain"`` with a ``details["reason"]`` saying where it looked and
how to train one. A missing optional model must not break ``analyze`` for the
signals that *are* available -- the same reasoning as the ``c2pa`` signal
without its optional library.

This module deliberately imports no ``torch`` at module scope, even though it
only ever runs with the ``ml`` extra installed:
:mod:`imgforensics.detectors` imports it for its registration side effect, and
paying a multi-second torch import on every CLI invocation -- including ones
that never touch this detector -- would be a poor trade. The heavy imports sit
inside :meth:`LearnedDetector.load`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from imgforensics.core.base import BaseDetector
from imgforensics.core.image import ForensicImage
from imgforensics.core.registry import register
from imgforensics.core.types import DetectionResult, label_from_score
from imgforensics.detectors.crops import crop_boxes

if TYPE_CHECKING:  # pragma: no cover - import-time typing only, never at runtime
    from imgforensics.detectors.features import FeatureExtractor
    from imgforensics.detectors.head import CheckpointMeta, MultiLayerHead

#: Environment variable naming the directory a trained head is loaded from.
HEAD_DIR_ENV = "IMGFORENSICS_HEAD_DIR"

#: Fallback checkpoint directory, relative to the working directory.
DEFAULT_HEAD_DIR = Path("weights/dinov2_head")

_TRAIN_HINT = "imgforensics train head --config configs/head_dinov2.yaml"
_ABSTAIN_SCORE = 0.5


def resolve_checkpoint_dir(checkpoint_dir: str | Path | None = None) -> Path:
    """The directory a head would be loaded from: argument, then env var, then default."""
    if checkpoint_dir is not None:
        return Path(checkpoint_dir)
    from_env = os.environ.get(HEAD_DIR_ENV)
    if from_env:
        return Path(from_env)
    return DEFAULT_HEAD_DIR


@register("dinov2_head")
class LearnedDetector(BaseDetector):
    """Scores an image with a trained :class:`~imgforensics.detectors.head.MultiLayerHead`.

    One prediction is: crop the image at native resolution per the
    checkpoint's crop policy, run the frozen backbone once over those crops,
    push each crop's multi-layer features through the head, calibrate the
    logits, and average the crop probabilities into the image score. The
    per-crop probabilities are also painted back onto a heatmap of the
    image's own shape, so a large image says *where* it looks generated and
    not only *how much*.

    The backbone is loaded once per instance and reused across
    :meth:`predict` calls (it is the expensive part), which is why the
    benchmark runner -- one instance, many images -- pays for it once.
    """

    name = "dinov2_head"

    def __init__(self, checkpoint_dir: str | Path | None = None) -> None:
        """Point the detector at a checkpoint directory (nothing is read yet).

        Args:
            checkpoint_dir: Directory holding ``head.json`` and
                ``head.safetensors``. ``None`` falls back to
                :data:`HEAD_DIR_ENV` and then :data:`DEFAULT_HEAD_DIR` -- and
                the fallback is resolved at :meth:`load` time, so setting the
                environment variable after constructing the detector still
                works.
        """
        self._configured_dir = checkpoint_dir
        self.checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir)
        self.device = "cpu"
        self.meta: CheckpointMeta | None = None
        self._head: MultiLayerHead | None = None
        self._extractor: FeatureExtractor | None = None
        self._attempted = False
        self._reason: str | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether a trained head was found and loaded."""
        return self._head is not None and self.meta is not None

    def load(self, device: str = "auto") -> None:
        """Read the checkpoint, if there is one, and build the head on ``device``.

        Never raises for a missing checkpoint -- the detector simply stays
        unloaded and :meth:`predict` abstains with a reason. ``device``
        defaults to ``"auto"`` rather than the base class's ``"cpu"`` because
        callers that run this detector at all (the CLI, the benchmark runner)
        call ``load()`` with no argument, and a frozen ViT belongs on the GPU
        when there is one.
        """
        self._attempted = True
        self.checkpoint_dir = resolve_checkpoint_dir(self._configured_dir)
        metadata_path = self.checkpoint_dir / "head.json"
        weights_path = self.checkpoint_dir / "head.safetensors"

        if not (metadata_path.is_file() and weights_path.is_file()):
            self._reason = f"no trained head found at {self.checkpoint_dir}; run: {_TRAIN_HINT}"
            return

        import torch
        from safetensors.torch import load_file

        from imgforensics.detectors.backbones import resolve_device
        from imgforensics.detectors.head import CheckpointMeta, MultiLayerHead

        meta = CheckpointMeta.model_validate_json(metadata_path.read_text(encoding="utf-8"))
        self.device = resolve_device(device)
        head = MultiLayerHead(meta.head)
        head.load_state_dict(load_file(str(weights_path)))
        head.eval()
        head.requires_grad_(False)

        self.meta = meta
        self._head = head.to(torch.device(self.device))
        self._extractor = None
        self._reason = None

    def _extractor_for(self, meta: CheckpointMeta) -> FeatureExtractor:
        """The lazily built feature extractor, shared across :meth:`predict` calls."""
        if self._extractor is None:
            from imgforensics.detectors.features import FeatureExtractor

            self._extractor = FeatureExtractor(
                backbone=meta.backbone, device=self.device, crop_policy=meta.crop
            )
        return self._extractor

    def _abstain(self) -> DetectionResult:
        """The result returned when no trained head is available."""
        reason = (
            self._reason or f"no trained head found at {self.checkpoint_dir}; run: {_TRAIN_HINT}"
        )
        return DetectionResult(
            detector=self.name,
            score=_ABSTAIN_SCORE,
            label="uncertain",
            details={"reason": reason},
        )

    def _heatmap(
        self, image: ForensicImage, meta: CheckpointMeta, per_crop: np.ndarray
    ) -> np.ndarray | None:
        """Paint each crop's probability over the region it was cut from.

        Pixels no crop covered stay 0, and pixels several crops covered (only
        possible in ``random`` mode, whose crops may overlap) get the mean of
        those crops' probabilities. A crop that fell entirely in the
        reflection padding of an undersized image covers nothing and is
        skipped, so a tiny image can legitimately produce an all-zero map.
        """
        material = image.raw if image.raw is not None else np.asarray(image.rgb).tobytes()
        boxes = crop_boxes(image.rgb, meta.crop, seed_material=material)
        if len(boxes) != len(per_crop):  # pragma: no cover - the two share one crop policy
            return None

        totals = np.zeros((image.height, image.width), dtype=np.float64)
        counts = np.zeros_like(totals)
        for box, probability in zip(boxes, per_crop, strict=True):
            if box.is_empty:
                continue
            rows = slice(box.top, box.top + box.height)
            columns = slice(box.left, box.left + box.width)
            totals[rows, columns] += float(probability)
            counts[rows, columns] += 1.0

        covered = counts > 0
        heatmap = np.zeros_like(totals)
        heatmap[covered] = totals[covered] / counts[covered]
        return np.clip(heatmap, 0.0, 1.0).astype(np.float32)

    def predict(self, image: ForensicImage) -> DetectionResult:
        """Score one image, or abstain when no trained head is available."""
        if not self._attempted:
            self.load()
        head, meta = self._head, self.meta
        if head is None or meta is None:
            return self._abstain()

        import torch

        from imgforensics.detectors.head import aggregate_crops

        features = self._extractor_for(meta).features_for_image(image)
        with torch.no_grad():
            logits = (
                head(torch.as_tensor(features, device=torch.device(self.device)))
                .cpu()
                .numpy()
                .astype(np.float64)
            )

        calibration = meta.calibration
        # One quantity, read twice: the crop probabilities reported in the
        # details are the sigmoid of exactly the logits the score aggregates.
        calibrated_logits = (logits + calibration.bias) / calibration.temperature
        per_crop = calibration.apply(logits)
        score = aggregate_crops(calibrated_logits)

        details: dict[str, Any] = {
            "backbone": meta.backbone,
            "checkpoint": f"{self.checkpoint_dir.name} ({meta.created})",
            "n_crops": int(len(per_crop)),
            "per_crop": [round(float(value), 4) for value in per_crop],
            "temperature": round(calibration.temperature, 4),
            "bias": round(calibration.bias, 4),
        }
        return DetectionResult(
            detector=self.name,
            score=score,
            label=label_from_score(score),
            heatmap=self._heatmap(image, meta, per_crop),
            details=details,
        )
