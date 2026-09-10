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

**Two maps, two passes.** Alongside the crop-probability heatmap, the detector
produces a Grad-CAM attribution map
(:mod:`imgforensics.detectors.attribution`) saying where *inside* those crops
the head looked; ``IMGFORENSICS_HEAD_ATTRIBUTION`` turns it off. It is a
second forward pass rather than a gradient-carrying version of the first one
on purpose: the scoring path -- and the cached-feature path that shares its
code with the training sweep -- has to keep producing exactly the numbers that
were benchmarked, and folding ``enable_grad`` into it would change what the
backbone runs under for every caller, including the ones that never ask for an
explanation. The score therefore still comes from the untouched no-grad path,
and a failure in the attribution pass is caught and reported in the details
rather than allowed to fail the prediction.

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
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from imgforensics.core.base import BaseDetector
from imgforensics.core.image import ForensicImage
from imgforensics.core.parameters import ParameterSpec
from imgforensics.core.registry import register
from imgforensics.core.types import DetectionResult, ToolKind, label_from_score
from imgforensics.detectors.crops import CropBox, crop_boxes

if TYPE_CHECKING:  # pragma: no cover - import-time typing only, never at runtime
    from imgforensics.detectors.features import FeatureExtractor
    from imgforensics.detectors.head import CheckpointMeta, MultiLayerHead

#: Environment variable naming the directory a trained head is loaded from.
HEAD_DIR_ENV = "IMGFORENSICS_HEAD_DIR"

#: Environment variable switching the Grad-CAM attribution map off.
HEAD_ATTRIBUTION_ENV = "IMGFORENSICS_HEAD_ATTRIBUTION"

#: Fallback checkpoint directory, relative to the working directory.
DEFAULT_HEAD_DIR = Path("weights/dinov2_head")

_TRAIN_HINT = "imgforensics train head --config configs/head_dinov2.yaml"
_ABSTAIN_SCORE = 0.5

#: Values of :data:`HEAD_ATTRIBUTION_ENV` that switch attribution off.
_ATTRIBUTION_OFF = frozenset({"0", "false", "no", "off"})

#: Longest attribution failure message kept in the details.
_ERROR_LIMIT = 200


def resolve_checkpoint_dir(checkpoint_dir: str | Path | None = None) -> Path:
    """The directory a head would be loaded from: argument, then env var, then default."""
    if checkpoint_dir is not None:
        return Path(checkpoint_dir)
    from_env = os.environ.get(HEAD_DIR_ENV)
    if from_env:
        return Path(from_env)
    return DEFAULT_HEAD_DIR


def resolve_attribution(attribution: bool | None = None) -> bool:
    """Whether to compute the attribution map: argument, then env var, then on.

    ``IMGFORENSICS_HEAD_ATTRIBUTION`` is read case-insensitively and only
    ``0``, ``false``, ``no`` and ``off`` switch the map off; unset, ``1``,
    ``true``, ``yes``, ``on`` -- and anything unrecognised -- leave it on. The
    default is on because the explanation is the point of the report and the
    extra cost is one backward pass over a handful of 224 px crops, not
    another model.
    """
    if attribution is not None:
        return bool(attribution)
    from_env = os.environ.get(HEAD_ATTRIBUTION_ENV)
    if from_env is None:
        return True
    return from_env.strip().lower() not in _ATTRIBUTION_OFF


@register("dinov2_head")
class LearnedDetector(BaseDetector):
    """Scores an image with a trained :class:`~imgforensics.detectors.head.MultiLayerHead`.

    One prediction is: crop the image at native resolution per the
    checkpoint's crop policy, run the frozen backbone once over those crops,
    push each crop's multi-layer features through the head, calibrate the
    logits, and average the crop probabilities into the image score. The
    per-crop probabilities are also painted back onto a heatmap of the
    image's own shape, so a large image says *where* it looks generated and
    not only *how much*, and a second pass paints a Grad-CAM attribution map
    saying where inside those crops the head looked (see the module
    docstring).

    The backbone is loaded once per instance and reused across
    :meth:`predict` calls (it is the expensive part), which is why the
    benchmark runner -- one instance, many images -- pays for it once.
    """

    name = "dinov2_head"
    kind: ClassVar[ToolKind] = "detector"

    @classmethod
    def parameters(cls) -> list[ParameterSpec]:
        """Whether to paint the Grad-CAM attribution map alongside the heatmap.

        The declared default is what the constructor would pick on its own --
        :data:`HEAD_ATTRIBUTION_ENV` when it is set, on otherwise -- so a
        deployment that switches attribution off through the environment sees
        it offered as off rather than being turned back on by a caller who
        accepted "the default". ``checkpoint_dir`` is not offered: which
        checkpoint is installed is a deployment setting, not a reading of the
        image.
        """
        return [
            ParameterSpec(
                name="attribution",
                kind="bool",
                default=resolve_attribution(),
                description=(
                    "Also compute the Grad-CAM map showing where inside the scored crops "
                    "the head looked. Costs one extra forward and backward pass over those "
                    "crops; the score, label and heatmap are the same either way."
                ),
            )
        ]

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        attribution: bool | None = None,
    ) -> None:
        """Point the detector at a checkpoint directory (nothing is read yet).

        Args:
            checkpoint_dir: Directory holding ``head.json`` and
                ``head.safetensors``. ``None`` falls back to
                :data:`HEAD_DIR_ENV` and then :data:`DEFAULT_HEAD_DIR` -- and
                the fallback is resolved at :meth:`load` time, so setting the
                environment variable after constructing the detector still
                works.
            attribution: Whether to compute the Grad-CAM attribution map.
                ``None`` (the default) reads :data:`HEAD_ATTRIBUTION_ENV` at
                :meth:`load` time, on the same "set it after constructing"
                reasoning as ``checkpoint_dir``.
        """
        self._configured_dir = checkpoint_dir
        self._configured_attribution = attribution
        self.checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir)
        self.attribution = resolve_attribution(attribution)
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
        self.attribution = resolve_attribution(self._configured_attribution)
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

    def _boxes_for(self, image: ForensicImage, meta: CheckpointMeta) -> list[CropBox]:
        """Where this image's crops were cut from, seeded the way the crops were.

        Shared by the heatmap and the attribution map so the two can never
        paint their per-crop values over different rectangles.
        """
        material = image.raw if image.raw is not None else np.asarray(image.rgb).tobytes()
        return crop_boxes(image.rgb, meta.crop, seed_material=material)

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
        boxes = self._boxes_for(image, meta)
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

    def _attribution(
        self, image: ForensicImage, meta: CheckpointMeta, head: MultiLayerHead
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """The Grad-CAM map for this image, plus the ``details`` entry describing it.

        Everything here is best-effort: a backbone without the block list
        Grad-CAM hooks, a crop grid that is not square, or anything else the
        gradient pass trips over yields ``(None, {"error": ...})`` and leaves
        the score, the label and the heatmap exactly as they were.
        """
        try:
            import torch

            from imgforensics.detectors.attribution import (
                grad_cam,
                stitch_attribution,
                target_blocks,
            )

            extractor = self._extractor_for(meta)
            batch = torch.from_numpy(extractor.batch_for_image(image))
            per_crop_maps = grad_cam(extractor.model, extractor.spec, head, batch, meta.calibration)
            stitched = stitch_attribution(
                (image.height, image.width),
                self._boxes_for(image, meta),
                per_crop_maps,
                meta.crop.size,
            )
            weights = [float(value) for value in head.layer_weights().detach().cpu()]
            blocks = target_blocks(
                extractor.spec.layers, len(getattr(extractor.model, "blocks", ())), weights
            )
            details = {
                "method": "grad-cam",
                "target_blocks": sorted(blocks),
                "grid": int(per_crop_maps.shape[-1]),
            }
            return stitched, details
        except Exception as exc:
            # Broad on purpose: an explanation is never worth a failed run.
            message = f"{type(exc).__name__}: {exc}"
            return None, {"error": message[:_ERROR_LIMIT]}

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

        attribution: np.ndarray | None = None
        if self.attribution:
            attribution, details["attribution"] = self._attribution(image, meta, head)

        return DetectionResult(
            detector=self.name,
            score=score,
            label=label_from_score(score),
            heatmap=self._heatmap(image, meta, per_crop),
            attribution=attribution,
            details=details,
        )
