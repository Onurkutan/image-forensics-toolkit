"""Training loop for the multi-layer head, and the self-describing checkpoint it writes.

The expensive half of this pipeline already ran: the frozen backbone turned
every image into a small cached feature array
(:mod:`imgforensics.detectors.features`). What is left is a logistic-regression
-shaped problem over a fixed feature space, so :func:`train_head` keeps the
whole training set in memory, runs full-precision AdamW over it, and finishes
in minutes on the 6 GB GPU in ``docs/ROADMAP.md``, section 6.

Three choices are worth stating outright:

- **A crop is a sample; an image is the unit that is scored.** Each crop
  carries its image's label during training (there is no crop-level ground
  truth), but every validation number is computed *after* aggregating an
  image's crops, because that is what the deployed detector reports. The
  ``image_index`` array is what connects the two.
- **Model selection on AUC, calibration afterwards.** The best epoch is the
  one with the best image-level validation AUC -- a ranking metric, unaffected
  by how mis-scaled the logits are -- and only then is a temperature and bias
  fitted, on the same split, to make the probabilities mean something. Fitting
  calibration cannot change which epoch was chosen, and the reported ECE
  before/after says how much it was needed.
- **The validation split is only ever seen un-augmented.** Augmented views
  exist for training images alone; validation reads view 0, so an epoch's
  numbers are comparable across runs with different augmentation policies.

The checkpoint (:class:`~imgforensics.detectors.head.CheckpointMeta`) records
the backbone, crop policy, augmentation, head shape, calibration, validation
metrics and the training manifests -- including their licenses and the
``commercial_ok`` flag, ANDed across them, so a head trained on research-only
data says so.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from pydantic import BaseModel, Field
from torch import nn

from imgforensics import __version__
from imgforensics.data.manifest import Manifest
from imgforensics.detectors.backbones import get_backbone, resolve_device
from imgforensics.detectors.crops import CropPolicy
from imgforensics.detectors.features import NO_AUGMENT_HASH, FeatureCache, FeatureExtractor
from imgforensics.detectors.head import (
    Calibration,
    CheckpointMeta,
    HeadOptions,
    ManifestRef,
    MultiLayerHead,
    ValMetrics,
)
from imgforensics.eval.metrics import (
    balanced_accuracy_at_threshold,
    best_threshold,
    expected_calibration_error,
    roc_auc,
)
from imgforensics.eval.preprocess import AugmentationConfig

#: File names inside a checkpoint directory.
WEIGHTS_FILENAME = "head.safetensors"
METADATA_FILENAME = "head.json"
LOG_FILENAME = "training_log.jsonl"

_FIXED_THRESHOLD = 0.5
_EVAL_CHUNK = 8192
_CALIBRATION_MAX_ITER = 100
_CALIBRATION_LR = 0.1
_TEMPERATURE_BOUNDS = (0.05, 20.0)
_EPSILON = 1e-7


class TrainConfig(BaseModel):
    """Everything one ``imgforensics train head`` run needs, loadable from YAML.

    Attributes:
        train_manifest: Manifest of the images the head is fitted on.
        val_manifest: Manifest used for per-epoch model selection, the
            calibration fit and the reported metrics. Keep it disjoint from
            ``train_manifest``; nothing here can check that for you.
        cache_dir: Feature cache to read from and write to.
        backbone: Frozen backbone name; must match the cached features.
        crop: Crop policy; also part of the feature cache key.
        augment: Path to an :class:`~imgforensics.eval.preprocess.AugmentationConfig`
            YAML, applied to training views above 0. ``None`` disables
            augmentation entirely.
        views: Views per training image (1 = the un-augmented image only).
        head: Head hyperparameters; ``n_layers`` and ``dim`` are read off the
            cached features rather than configured.
        epochs: Maximum epochs; early stopping may end the run sooner.
        batch_size: Crops per optimizer step.
        lr: AdamW peak learning rate, annealed to zero on a cosine schedule.
        weight_decay: AdamW weight decay.
        label_smoothing: Pulls the binary targets toward 0.5 by this amount.
            Worth a small value here because crop labels are inherited from
            the image and some crops genuinely carry no evidence.
        balance_classes: Sample crops inversely to their class frequency, so
            an unbalanced manifest does not bias the head toward the majority
            label.
        early_stopping_patience: Stop after this many epochs without a new
            best validation AUC.
        seed: Seeds torch, the sampler and the parameter initialisation.
        out_dir: Directory the checkpoint is written to.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, or an explicit device.
        max_train_images: Optional cap on training manifest entries (the
            first N), for a quick smoke run.
        max_val_images: The same cap for the validation manifest.
    """

    train_manifest: Path
    val_manifest: Path
    cache_dir: Path = Path("data/features")
    backbone: str = "dinov2_vitb14"
    crop: CropPolicy = Field(default_factory=CropPolicy)
    augment: str | None = None
    views: int = Field(default=1, ge=1)
    head: HeadOptions = Field(default_factory=HeadOptions)
    epochs: int = Field(default=20, ge=1)
    batch_size: int = Field(default=256, ge=1)
    lr: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-2, ge=0.0)
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)
    balance_classes: bool = True
    early_stopping_patience: int = Field(default=5, ge=1)
    seed: int = 0
    out_dir: Path = Path("weights/dinov2_head")
    device: str = "auto"
    max_train_images: int | None = None
    max_val_images: int | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """Load a config from a YAML mapping of this model's fields."""
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)

    def augmentation(self) -> AugmentationConfig | None:
        """The loaded augmentation config, or ``None`` when ``augment`` is unset."""
        if self.augment is None:
            return None
        return AugmentationConfig.from_yaml(self.augment)


class EpochRecord(BaseModel):
    """One line of ``training_log.jsonl``: what an epoch cost and what it bought."""

    epoch: int
    train_loss: float
    val_auc: float
    val_balanced_accuracy: float
    lr: float
    seconds: float
    best: bool


class TrainReport(BaseModel):
    """What :func:`train_head` produced: the checkpoint, where it landed, and the log."""

    out_dir: Path
    weights_path: Path
    metadata_path: Path
    log_path: Path
    meta: CheckpointMeta
    epochs: list[EpochRecord] = Field(default_factory=list)
    train_crops: int
    train_images: int
    val_crops: int
    val_images: int
    layer_weights: list[float] = Field(default_factory=list)
    head_parameters: int = 0
    elapsed_s: float = 0.0


@dataclass
class _Samples:
    """Cached features flattened to crop-level samples, plus the image they came from.

    ``features`` is ``(n_crops, n_layers, dim)``, ``labels`` is the image
    label repeated per crop, ``image_index`` maps each crop to a row of
    ``image_labels``. Keeping both levels means a training step can run over
    crops while every reported metric runs over images.
    """

    features: np.ndarray
    labels: np.ndarray
    image_index: np.ndarray
    image_labels: np.ndarray

    @property
    def n_crops(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_images(self) -> int:
        return int(self.image_labels.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_layers, dim)`` of one crop's features."""
        return int(self.features.shape[1]), int(self.features.shape[2])


def _sha256_file(path: Path) -> str:
    """The sha256 of a file's bytes, hex-encoded."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_ref(role: str, path: Path, manifest: Manifest, entries: int) -> ManifestRef:
    """Describe one manifest for the checkpoint's provenance record."""
    return ManifestRef(
        role=role,
        path=path.as_posix(),
        dataset=manifest.meta.dataset,
        entries=entries,
        sha256=_sha256_file(path),
        license=manifest.meta.license,
        commercial_ok=manifest.meta.commercial_ok,
    )


def _combine_commercial_ok(values: Sequence[bool | None]) -> bool | None:
    """AND over the manifests' ``commercial_ok``, with unknown dominating unknown-ness.

    A single ``False`` makes the result ``False`` -- one research-only source
    is enough to make the trained head research-only. Otherwise an unverified
    source (``None``) keeps the answer unknown rather than optimistic; only
    an all-``True`` set yields ``True``.
    """
    if any(value is False for value in values):
        return False
    if any(value is None for value in values):
        return None
    return True


def _collect(
    manifest: Manifest,
    manifest_path: Path,
    cache: FeatureCache,
    extractor: FeatureExtractor,
    limit: int | None,
    progress: bool,
) -> _Samples:
    """Load (computing anything missing) every view's features for one manifest.

    The extractor is handed the cache, so this both *ensures* the features
    exist -- a user can run ``train`` without a separate ``features extract``
    -- and reads them back. Results arrive in input order, exactly
    ``extractor.views`` per path, which is what lets each one be attributed to
    its image without matching on paths.
    """
    entries = manifest.entries[:limit] if limit is not None else manifest.entries
    if not entries:
        raise ValueError(f"{manifest_path} has no entries to train or validate on")

    root = Path(manifest.meta.root)
    paths = [root / entry.path for entry in entries]
    image_labels = np.asarray([1.0 if entry.label == "fake" else 0.0 for entry in entries])

    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for position, (_, _, features) in enumerate(
        extractor.features_for_paths(paths, cache=cache, progress=progress)
    ):
        image = position // extractor.views
        blocks.append(np.asarray(features, dtype=np.float32))
        n_crops = features.shape[0]
        labels.append(np.full(n_crops, image_labels[image], dtype=np.float32))
        indices.append(np.full(n_crops, image, dtype=np.int64))

    return _Samples(
        features=np.concatenate(blocks, axis=0),
        labels=np.concatenate(labels, axis=0),
        image_index=np.concatenate(indices, axis=0),
        image_labels=image_labels,
    )


def _crop_logits(model: MultiLayerHead, features: torch.Tensor) -> torch.Tensor:
    """Every crop's logit, in chunks, with the model in eval mode and no grad."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], _EVAL_CHUNK):
            chunk = features[start : start + _EVAL_CHUNK].to(device, non_blocking=True)
            outputs.append(model(chunk))
    model.train(was_training)
    return torch.cat(outputs) if outputs else torch.empty(0, device=device)


def _image_probs(crop_probs: np.ndarray, image_index: np.ndarray, n_images: int) -> np.ndarray:
    """Mean crop probability per image -- the numpy twin of ``aggregate_crops("mean_prob")``."""
    totals = np.zeros(n_images, dtype=np.float64)
    counts = np.zeros(n_images, dtype=np.float64)
    np.add.at(totals, image_index, crop_probs)
    np.add.at(counts, image_index, 1.0)
    return totals / np.maximum(counts, 1.0)


def _epoch_indices(labels: torch.Tensor, balance: bool, generator: torch.Generator) -> torch.Tensor:
    """One epoch's sample order: a permutation, or a class-balanced draw with replacement."""
    n = labels.shape[0]
    if not balance:
        return torch.randperm(n, generator=generator)

    positives = float(labels.sum().item())
    negatives = float(n - positives)
    if positives == 0.0 or negatives == 0.0:
        return torch.randperm(n, generator=generator)

    weights = torch.where(labels > 0.5, 1.0 / positives, 1.0 / negatives).double()
    return torch.multinomial(weights, n, replacement=True, generator=generator)


def _image_ece(
    calibration: Calibration,
    logits: np.ndarray,
    image_index: np.ndarray,
    image_labels: np.ndarray,
) -> float:
    """Expected calibration error of the aggregated image probabilities under ``calibration``."""
    probabilities = _image_probs(calibration.apply(logits), image_index, len(image_labels))
    return expected_calibration_error(image_labels, probabilities)


def _fit_calibration(
    logits: np.ndarray, image_index: np.ndarray, image_labels: np.ndarray
) -> Calibration:
    """Fit ``temperature``/``bias`` by minimizing the *image-level* NLL with LBFGS.

    The objective is the negative log-likelihood of the aggregated image
    probability, ``mean_crops sigmoid((logit + bias) / T)``, not of the
    individual crops: that is the quantity the detector reports and the one
    the ECE is measured on, so it is the one worth calibrating. Optimizing
    ``log T`` keeps the temperature positive without a constraint.

    The fitted temperature is then clamped into :data:`_TEMPERATURE_BOUNDS`
    -- on a small or perfectly separable validation split the likelihood is
    maximized by ``T -> 0``, which would ship a step function -- and the
    result is *accepted only if it does not increase the expected calibration
    error*. The likelihood is what can be optimized smoothly, but the ECE is
    what the checkpoint reports and what a calibrated probability is for, and
    the two do come apart on a degenerate split. Calibration is a free win or
    it is not applied at all: a run that cannot improve on the raw logits
    records ``temperature = 1, bias = 0`` and an unchanged ECE, which is the
    honest outcome rather than a hidden one.

    Also returns the identity when the validation split has only one class
    (nothing to fit) or the optimizer produced non-finite values.
    """
    identity = Calibration()
    if len(np.unique(image_labels)) < 2:
        return identity

    device = torch.device("cpu")
    crop_logits = torch.as_tensor(logits, dtype=torch.float64, device=device)
    index = torch.as_tensor(image_index, dtype=torch.int64, device=device)
    targets = torch.as_tensor(image_labels, dtype=torch.float64, device=device)
    counts = torch.zeros_like(targets).index_add_(0, index, torch.ones_like(crop_logits))
    counts = counts.clamp(min=1.0)

    log_temperature = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature, bias], lr=_CALIBRATION_LR, max_iter=_CALIBRATION_MAX_ITER
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        probs = torch.sigmoid((crop_logits + bias) / torch.exp(log_temperature))
        image_probs = torch.zeros_like(targets).index_add_(0, index, probs) / counts
        image_probs = image_probs.clamp(_EPSILON, 1.0 - _EPSILON)
        loss = -(
            targets * torch.log(image_probs) + (1.0 - targets) * torch.log(1.0 - image_probs)
        ).mean()
        loss.backward()
        return loss

    optimizer.step(closure)  # type: ignore[arg-type]

    temperature = float(torch.exp(log_temperature).item())
    shift = float(bias.item())
    if not (np.isfinite(temperature) and np.isfinite(shift)):
        return identity

    low, high = _TEMPERATURE_BOUNDS
    candidate = Calibration(temperature=min(max(temperature, low), high), bias=shift)
    if _image_ece(candidate, logits, image_index, image_labels) > _image_ece(
        identity, logits, image_index, image_labels
    ):
        return identity
    return candidate


def _write_log(path: Path, records: Sequence[EpochRecord]) -> None:
    """Write one JSON object per epoch, in order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def _save_weights(model: MultiLayerHead, path: Path) -> None:
    """Write the head's tensors to ``path`` in the safetensors format."""
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()
    }
    save_file(tensors, str(path))


def train_head(config: TrainConfig, progress: bool = True) -> TrainReport:
    """Train, calibrate and save a :class:`~imgforensics.detectors.head.MultiLayerHead`.

    Extracts (or reads from cache) the features of both manifests, trains for
    up to ``config.epochs`` epochs keeping the best image-level validation
    AUC, fits a temperature and bias on the validation split, and writes
    ``head.safetensors``, ``head.json`` and ``training_log.jsonl`` into
    ``config.out_dir``.

    Raises:
        ValueError: either manifest is empty, or the training split contains
            only one class.
    """
    started = time.perf_counter()
    spec = get_backbone(config.backbone)
    device = torch.device(resolve_device(config.device))
    augmentation = config.augmentation()
    cache = FeatureCache(config.cache_dir)

    train_manifest = Manifest.load(config.train_manifest)
    val_manifest = Manifest.load(config.val_manifest)

    train_samples = _collect(
        train_manifest,
        Path(config.train_manifest),
        cache,
        FeatureExtractor(
            backbone=config.backbone,
            device=config.device,
            crop_policy=config.crop,
            augment=augmentation,
            views=config.views,
        ),
        config.max_train_images,
        progress,
    )
    # Validation always reads the un-augmented view 0: an epoch's numbers must
    # not depend on which augmentation policy the training views used.
    val_samples = _collect(
        val_manifest,
        Path(config.val_manifest),
        cache,
        FeatureExtractor(
            backbone=config.backbone,
            device=config.device,
            crop_policy=config.crop,
            views=1,
        ),
        config.max_val_images,
        progress,
    )

    if len(np.unique(train_samples.labels)) < 2:
        raise ValueError(
            f"{config.train_manifest} has only one class; a binary head cannot be trained on it"
        )

    n_layers, dim = train_samples.shape
    head_config = config.head.with_shape(n_layers=n_layers, dim=dim)

    torch.manual_seed(config.seed)
    model = MultiLayerHead(head_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(config.seed)

    # Features stay in host memory and only the current batch moves to the
    # device: a large multi-view cache (tens of thousands of images) would not
    # fit next to the backbone on a 6 GB GPU if it were resident all at once.
    train_features = torch.as_tensor(train_samples.features)
    train_labels = torch.as_tensor(train_samples.labels)
    smoothed = train_labels * (1.0 - config.label_smoothing) + 0.5 * config.label_smoothing
    val_features = torch.as_tensor(val_samples.features)

    best_auc = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = copy.deepcopy(model.state_dict())
    since_best = 0
    records: list[EpochRecord] = []

    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = _epoch_indices(train_labels, config.balance_classes, generator)

        total_loss = 0.0
        steps = 0
        for start in range(0, order.shape[0], config.batch_size):
            batch = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            inputs = train_features[batch].to(device, non_blocking=True)
            targets = smoothed[batch].to(device, non_blocking=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1

        learning_rate = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        crop_probs = torch.sigmoid(_crop_logits(model, val_features)).cpu().numpy()
        image_probs = _image_probs(crop_probs, val_samples.image_index, val_samples.n_images)
        auc = roc_auc(val_samples.image_labels, image_probs)
        accuracy = balanced_accuracy_at_threshold(
            val_samples.image_labels, image_probs, _FIXED_THRESHOLD
        )

        improved = auc > best_auc
        if improved:
            best_auc, best_epoch, since_best = auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since_best += 1

        records.append(
            EpochRecord(
                epoch=epoch,
                train_loss=total_loss / max(steps, 1),
                val_auc=auc,
                val_balanced_accuracy=accuracy,
                lr=learning_rate,
                seconds=time.perf_counter() - epoch_started,
                best=improved,
            )
        )
        if progress:
            print(
                f"[train] epoch {epoch}/{config.epochs} "
                f"loss={records[-1].train_loss:.4f} val_auc={auc:.4f} "
                f"val_bacc={accuracy:.4f}{' *' if improved else ''}"
            )
        if since_best >= config.early_stopping_patience:
            break

    model.load_state_dict(best_state)
    val_logits = _crop_logits(model, val_features).cpu().numpy().astype(np.float64)

    uncalibrated = _image_probs(
        1.0 / (1.0 + np.exp(-val_logits)), val_samples.image_index, val_samples.n_images
    )
    calibration = _fit_calibration(val_logits, val_samples.image_index, val_samples.image_labels)
    calibrated = _image_probs(
        calibration.apply(val_logits), val_samples.image_index, val_samples.n_images
    )

    tuned_threshold, tuned_accuracy = best_threshold(val_samples.image_labels, calibrated)
    metrics = ValMetrics(
        auc=roc_auc(val_samples.image_labels, calibrated),
        balanced_accuracy=balanced_accuracy_at_threshold(
            val_samples.image_labels, calibrated, _FIXED_THRESHOLD
        ),
        balanced_accuracy_tuned=tuned_accuracy,
        threshold=tuned_threshold,
        ece_before=expected_calibration_error(val_samples.image_labels, uncalibrated),
        ece_after=expected_calibration_error(val_samples.image_labels, calibrated),
    )

    manifests = [
        _manifest_ref("train", Path(config.train_manifest), train_manifest, train_samples.n_images),
        _manifest_ref("val", Path(config.val_manifest), val_manifest, val_samples.n_images),
    ]
    meta = CheckpointMeta(
        package_version=__version__,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        backbone=config.backbone,
        layers=list(spec.layers),
        crop=config.crop,
        augment_config=config.augment,
        augment_hash=NO_AUGMENT_HASH if augmentation is None else augmentation.fingerprint(),
        views=config.views,
        head=head_config,
        calibration=calibration,
        best_epoch=best_epoch,
        epochs_run=len(records),
        val=metrics,
        manifests=manifests,
        commercial_ok=_combine_commercial_ok([ref.commercial_ok for ref in manifests]),
        licenses=sorted({ref.license for ref in manifests if ref.license}),
    )

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / WEIGHTS_FILENAME
    metadata_path = out_dir / METADATA_FILENAME
    log_path = out_dir / LOG_FILENAME
    _save_weights(model, weights_path)
    metadata_path.write_text(
        json.dumps(meta.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_log(log_path, records)

    return TrainReport(
        out_dir=out_dir,
        weights_path=weights_path,
        metadata_path=metadata_path,
        log_path=log_path,
        meta=meta,
        epochs=records,
        train_crops=train_samples.n_crops,
        train_images=train_samples.n_images,
        val_crops=val_samples.n_crops,
        val_images=val_samples.n_images,
        layer_weights=[float(value) for value in model.layer_weights().detach().cpu()],
        head_parameters=sum(parameter.numel() for parameter in model.parameters()),
        elapsed_s=time.perf_counter() - started,
    )
