"""The small trainable head that sits on top of cached frozen-backbone features.

The backbone is frozen and its features are cached
(:mod:`imgforensics.detectors.features`), so this is the only part of the
learned detector that trains -- roughly a million parameters over a fixed
feature space, minutes on a 6 GB GPU rather than days.

Two design points come straight from the survey
(``docs/research/01_ai_generated_image_detection.md``, section 7a):

- **Read several depths, not just the last block.** The discriminative signal
  is spread across the transformer's depth, so the head sees the CLS token of
  each selected block plus the pooled output, projects each one separately
  (its own LayerNorm and Linear, because different depths have different
  scales and different meanings), and combines them with *learned* importance
  weights instead of a fixed choice of layer. :meth:`MultiLayerHead.layer_weights`
  exposes those weights, which is the cheapest interpretability this
  architecture offers: it says which depths the trained head actually uses.
- **Score crops, then aggregate.** An image is cropped, never resized, so a
  prediction exists per crop and has to be reduced to one image score.
  :func:`aggregate_crops` does that reduction over *probabilities*, not
  logits -- averaging logits would let one extreme crop dominate an image
  that is otherwise unremarkable.

:class:`CheckpointMeta` is the schema of the ``head.json`` that ships beside
every trained ``head.safetensors``. It lives here rather than in
:mod:`imgforensics.detectors.train` because both the writer (training) and
the reader (inference) need it, and a checkpoint that cannot say which
backbone, crop policy and training data it came from is not reusable.

``torch`` is imported at module scope here (a :class:`torch.nn.Module`
subclass cannot be declared without it), so this module needs the optional
``ml`` extra. :mod:`imgforensics.detectors.learned`, which has to stay
importable without it, imports this module inside the methods that need it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch import nn

from imgforensics.detectors.crops import CropPolicy

#: How a per-crop prediction is reduced to one image-level probability.
CropAggregation = Literal["mean_prob", "max_prob"]


class HeadOptions(BaseModel):
    """The head's hyperparameters, without the shape the features dictate.

    Split out from :class:`HeadConfig` so a training config can carry the
    choices a user makes while ``n_layers`` and ``dim`` are read off the
    cached features (see :meth:`with_shape`) -- there is no way to write them
    down wrongly.

    Attributes:
        proj_dim: Width each layer's features are projected to before the
            layers are combined.
        hidden_dim: Width of the classifier MLP's hidden layer.
        dropout: Dropout probability inside the MLP.
        layer_weighting: ``"softmax"`` learns one importance weight per layer
            (a softmax over free logits, so the weights stay positive and sum
            to 1); ``"mean"`` fixes them to a uniform average, which is the
            ablation that says whether the learned weighting is earning its
            keep.
    """

    proj_dim: int = Field(default=256, gt=0)
    hidden_dim: int = Field(default=256, gt=0)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    layer_weighting: Literal["softmax", "mean"] = "softmax"

    def with_shape(self, n_layers: int, dim: int) -> HeadConfig:
        """This set of options plus the feature shape, as a full :class:`HeadConfig`."""
        return HeadConfig(n_layers=n_layers, dim=dim, **self.model_dump())


class HeadConfig(HeadOptions):
    """A fully specified head: :class:`HeadOptions` plus the feature shape it reads.

    Attributes:
        n_layers: Number of feature rows per crop -- the selected transformer
            blocks plus the pooled output (``len(spec.layers) + 1``).
        dim: Backbone feature width.
    """

    n_layers: int = Field(gt=0)
    dim: int = Field(gt=0)


class MultiLayerHead(nn.Module):
    """Per-layer projection, learned layer weighting, then a two-layer MLP.

    Input is ``(B, n_layers, dim)`` float32 -- one row per selected backbone
    depth, as :func:`imgforensics.detectors.backbones.extract` produces it --
    and the output is ``(B,)`` raw logits (positive means "generated"). The
    logits are deliberately uncalibrated; :mod:`imgforensics.detectors.train`
    fits a temperature and bias on the validation split afterwards.

    Size, for the default DINOv2 ViT-B/14 shape (5 rows of 768) with
    ``proj_dim = hidden_dim = 256``: about 1.06 M parameters, of which the
    five 768->256 projections are 984 K. Everything else -- the norms, the
    layer logits, the whole MLP -- is under 74 K.
    """

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.config = config
        self.norms = nn.ModuleList([nn.LayerNorm(config.dim) for _ in range(config.n_layers)])
        self.projections = nn.ModuleList(
            [nn.Linear(config.dim, config.proj_dim) for _ in range(config.n_layers)]
        )
        # Free logits, softmaxed in layer_weights(). Initialised to zeros, so
        # training starts from the uniform average that "mean" fixes forever.
        self.layer_logits = nn.Parameter(torch.zeros(config.n_layers))
        self.mlp = nn.Sequential(
            nn.Linear(config.proj_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def layer_weights(self) -> torch.Tensor:
        """The importance weight of each backbone layer: ``(n_layers,)``, summing to 1.

        A softmax over :attr:`layer_logits` under ``"softmax"`` weighting, and
        a constant ``1 / n_layers`` under ``"mean"``. Reported in the training
        checkpoint so a run says which depths its head leaned on.
        """
        if self.config.layer_weighting == "mean":
            return torch.full_like(self.layer_logits, 1.0 / self.config.n_layers)
        return torch.softmax(self.layer_logits, dim=0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Logits for a ``(B, n_layers, dim)`` batch of crop features: ``(B,)``.

        Raises:
            ValueError: ``features`` is not 3-D, or its layer/feature axes do
                not match the configured shape.
        """
        expected = (self.config.n_layers, self.config.dim)
        if features.ndim != 3 or tuple(features.shape[1:]) != expected:
            raise ValueError(
                f"expected features of shape (B, {expected[0]}, {expected[1]}), "
                f"got {tuple(features.shape)}"
            )

        projected = torch.stack(
            [
                projection(norm(features[:, index]))
                for index, (norm, projection) in enumerate(
                    zip(self.norms, self.projections, strict=True)
                )
            ],
            dim=1,
        )  # (B, n_layers, proj_dim)
        pooled = (projected * self.layer_weights().view(1, -1, 1)).sum(dim=1)
        return self.mlp(pooled).squeeze(-1)


def aggregate_crops(
    logits_per_crop: Sequence[float] | np.ndarray | torch.Tensor,
    mode: CropAggregation = "mean_prob",
) -> float:
    """Reduce one image's per-crop logits to a single probability in ``[0, 1]``.

    ``"mean_prob"`` (the default) averages the crops' probabilities: every
    crop of an image carries the image's label during training, so the mean
    is the estimator that matches how the head was fitted, and it is stable
    when a crop lands on empty sky. ``"max_prob"`` takes the most suspicious
    crop instead -- more sensitive to a locally generated region, and
    correspondingly more false-positive-prone on large images.

    Raises:
        ValueError: ``logits_per_crop`` is empty, or ``mode`` is unknown.
    """
    logits = np.asarray(
        logits_per_crop.detach().cpu().numpy()
        if isinstance(logits_per_crop, torch.Tensor)
        else logits_per_crop,
        dtype=np.float64,
    ).reshape(-1)
    if logits.size == 0:
        raise ValueError("aggregate_crops() requires at least one crop logit")

    probabilities = 1.0 / (1.0 + np.exp(-logits))
    if mode == "mean_prob":
        return float(probabilities.mean())
    if mode == "max_prob":
        return float(probabilities.max())
    raise ValueError(f"Unknown aggregation mode {mode!r}. Use 'mean_prob' or 'max_prob'.")


class Calibration(BaseModel):
    """The post-hoc logit transform fitted on the validation split.

    Temperature scaling with a bias term: ``sigmoid((logit + bias) /
    temperature)``. The temperature fixes over- or under-confidence (a value
    above 1 softens the probabilities), the bias moves the operating point,
    which matters when the training mix is not the deployment mix. The
    identity (``temperature = 1``, ``bias = 0``) is a valid, fitted-nothing
    calibration and is what an uncalibratable run records.
    """

    temperature: float = Field(default=1.0, gt=0.0)
    bias: float = 0.0

    def apply(self, logits: Sequence[float] | np.ndarray) -> np.ndarray:
        """Calibrated probabilities for raw ``logits``, as a float64 array."""
        array = np.asarray(logits, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-(array + self.bias) / self.temperature))


class ManifestRef(BaseModel):
    """One manifest a head was trained or validated on, recorded in its checkpoint.

    The ``sha256`` is of the manifest *file*, so a checkpoint can be matched
    against the exact entry list it saw -- a manifest that has since been
    resampled will not match. ``license`` and ``commercial_ok`` are copied
    from the manifest's metadata and travel with the weights: a head trained
    on research-only data is itself research-only, and the checkpoint is the
    only place that fact can survive the move from data to model.
    """

    role: str
    path: str
    dataset: str
    entries: int
    sha256: str
    license: str | None = None
    commercial_ok: bool | None = None


class ValMetrics(BaseModel):
    """Validation-split metrics of the selected epoch, at image level.

    ``balanced_accuracy`` is at the fixed 0.5 operating point and
    ``balanced_accuracy_tuned`` at ``threshold``, which was tuned on this same
    split -- an in-sample number, and an optimistic one; the honest figure for
    a report is the AUC and the 0.5-threshold accuracy.
    """

    auc: float
    balanced_accuracy: float
    balanced_accuracy_tuned: float
    threshold: float
    ece_before: float
    ece_after: float


class CheckpointMeta(BaseModel):
    """``head.json``: everything needed to reuse a trained head, and to audit it.

    A checkpoint is self-describing on purpose. The weights alone are
    meaningless without the backbone and crop policy that produced their
    input, so those are recorded rather than assumed, and
    :class:`imgforensics.detectors.learned.LearnedDetector` rebuilds its
    whole inference pipeline from this file. The training-data provenance
    (``manifests``, ``licenses``, ``commercial_ok``) is recorded for the
    licensing policy in ``docs/ROADMAP.md``, section 7.
    """

    format_version: int = 1
    package_version: str
    created: str
    backbone: str
    layers: list[int] = Field(default_factory=list)
    crop: CropPolicy
    augment_config: str | None = None
    augment_hash: str
    views: int
    head: HeadConfig
    calibration: Calibration
    best_epoch: int
    epochs_run: int
    val: ValMetrics
    manifests: list[ManifestRef] = Field(default_factory=list)
    commercial_ok: bool | None = None
    licenses: list[str] = Field(default_factory=list)
