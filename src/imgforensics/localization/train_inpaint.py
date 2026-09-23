"""Training loop for the ``dino_inpaint`` patch head, and the checkpoint it writes.

Unlike :mod:`imgforensics.detectors.train`, this one cannot precompute its
inputs: the head is fitted on *crops*, the crops are placed randomly every
epoch, and half of a fake's crops are steered onto its mask, so the frozen
backbone has to run inside the training loop rather than once into a cache.
That is the dominant cost (about 35 crops/s on this project's 6 GB card), and
everything below is arranged around it.

Four choices worth stating outright:

- **Only the fully regenerated fakes.** TGIF's spliced subsets (``sd2-sp``,
  ``ps-sp``) are what the Phase 4a localizers already handle; training on them
  would let this model learn the composite seam instead of the diffusion
  texture it exists to find. The training set is ``sd2-fr`` and ``sdxl-fr``
  against authentic images, in a fixed 0.35/0.35/0.30 mix, so the two
  generators and the negative class each get a predictable share of every
  epoch regardless of how many files each has on disk.
- **Half of a fake's crops are guided onto its mask.** A uniformly placed
  448 px crop of an ``sdxl-fr`` image hits the inpainted region only 64% of
  the time and, when it does, covers 8% of it; a run trained on that spends
  most of its gradient on all-negative crops. Guiding half of them onto a
  uniformly drawn mask pixel -- placed uniformly among the crops that *cover*
  it, never centered on it -- raises that to 99.6% without teaching the head
  where in a crop to look (:func:`~imgforensics.localization.dino_inpaint.sample_crop_box`).
- **Both classes are augmented identically.** The reals are pristine PNGs and
  so are the fakes, so augmentation is not about realism here; it is about
  making sure the head cannot separate the two classes by anything the
  augmentation touches. ``configs/augment_inpaint.yaml`` keeps the
  re-encodings and the noise, and drops the blur, the rescale and the cutout,
  which would destroy the high-frequency evidence the model reads.
- **Validation runs the inference path, not the training one.** A fixed,
  seeded 500-entry subset of the val split is scored exactly as
  ``imgforensics analyze`` would score it -- native resolution, tiled,
  overlaps averaged -- and the metrics are computed on the 14x-downsampled
  grid, so an epoch's number is the number the deployed localizer would
  produce, not an optimistic one measured on the crops it trained on.

Model selection is on the mean best-F1 over the validation fakes; the reals'
mean area above 0.5 is logged next to it, because a model that paints every
image is easy to build and its fake-only metrics look fine.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from PIL import Image
from pydantic import BaseModel, Field
from torch import nn
from torch.utils.data import DataLoader, Dataset

from imgforensics import __version__
from imgforensics.data.manifest import Manifest, ManifestEntry
from imgforensics.detectors.backbones import resolve_device
from imgforensics.detectors.train import _combine_commercial_ok, _sha256_file
from imgforensics.eval.metrics import PixelMetrics
from imgforensics.eval.preprocess import AugmentationConfig, _pad_to_at_least, augment
from imgforensics.localization._inpaint_model import (
    HEAD_PREFIX,
    LoRALinear,
    PatchHead,
    attach_lora,
    checkpoint_tensors,
    infer_heatmap,
    load_patch_backbone,
    patch_features,
    set_lora_training,
)
from imgforensics.localization.dino_inpaint import (
    LOG_FILENAME,
    MASK_THRESHOLD,
    METADATA_FILENAME,
    WEIGHTS_FILENAME,
    InpaintCheckpointMeta,
    InpaintConfig,
    InpaintValMetrics,
    LoraConfig,
    ManifestRef,
    TrainEcho,
    patch_targets,
    sample_crop_box,
)

_DICE_EPSILON = 1.0


class InpaintTrainConfig(BaseModel):
    """Everything one ``imgforensics train localizer`` run needs, loadable from YAML.

    Attributes:
        train_fake_manifest: Fakes the head is fitted on; every entry must
            carry a mask, and its ``generator`` must be a key of ``sampling``.
        train_real_manifest: Authentic images, the negative class. Their
            target is all-zero, so they need no mask.
        val_manifest: The fixed subset used for model selection and for the
            metrics in the checkpoint. Keep it disjoint from both training
            manifests; nothing here can check that for you.
        model: Backbone, blocks, crop size, stride and head shape -- the half
            of the configuration inference needs too, recorded in the
            checkpoint so it cannot drift.
        patch_target: ``"soft"`` regresses each patch's mask *area*,
            ``"hard"`` thresholds it at 0.5 first (the declared ablation).
        sampling: Share of each epoch's crops drawn from each class:
            ``"real"`` plus one key per fake generator. Normalized if the
            values do not sum to 1.
        positive_crop_fraction: Probability that a fake's crop is steered onto
            a mask pixel rather than placed uniformly.
        augment: Path to an
            :class:`~imgforensics.eval.preprocess.AugmentationConfig` YAML
            applied to every crop of both classes; ``None`` disables it.
        flip_probability: Probability of a horizontal flip of crop and target.
        pos_weight: Weight on the positive class inside the BCE term, for a
            target whose mean is around 0.1.
        dice_weight: Weight on the soft-Dice term added to the BCE.
        epochs: Maximum epochs; early stopping may end the run sooner.
        crops_per_epoch: Crops drawn per epoch (an epoch is a budget here,
            not a pass over a finite sample set).
        batch_size: Crops per optimizer step.
        lr: AdamW peak learning rate for the head, annealed on a cosine
            schedule.
        lora_lr: The same for the LoRA factors, used only when ``lora`` is
            set; the adapters want a smaller step than a head starting from
            scratch.
        weight_decay: AdamW weight decay.
        lora: Stage 2's adapters, or ``None`` for a frozen backbone.
        warm_start: A stage-1 checkpoint directory whose head weights this run
            starts from.
        grad_checkpointing: Recompute block activations in the backward pass.
            Only useful above batch size 4 with ``lora`` set; it costs about a
            third of the throughput.
        early_stopping_patience: Stop after this many epochs without a new
            best validation best-F1.
        num_workers: Dataloader worker processes decoding and augmenting
            crops while the GPU runs the backbone.
        seed: Seeds torch, the crop sampler and the head's initialisation.
        out_dir: Directory the checkpoint is written to.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, or an explicit device.
        val_limit: Optional cap on validation entries (the first N), for a
            quick smoke run.
    """

    train_fake_manifest: Path
    train_real_manifest: Path
    val_manifest: Path
    model: InpaintConfig = Field(default_factory=InpaintConfig)
    patch_target: Literal["soft", "hard"] = "soft"
    sampling: dict[str, float] = Field(
        default_factory=lambda: {"real": 0.30, "sd2-fr": 0.35, "sdxl-fr": 0.35}
    )
    positive_crop_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    augment: str | None = "configs/augment_inpaint.yaml"
    flip_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    pos_weight: float = Field(default=3.0, gt=0.0)
    dice_weight: float = Field(default=0.5, ge=0.0)
    epochs: int = Field(default=20, ge=1)
    crops_per_epoch: int = Field(default=20000, ge=1)
    batch_size: int = Field(default=8, ge=1)
    lr: float = Field(default=3e-4, gt=0.0)
    lora_lr: float = Field(default=1e-4, gt=0.0)
    weight_decay: float = Field(default=1e-2, ge=0.0)
    lora: LoraConfig | None = None
    warm_start: Path | None = None
    grad_checkpointing: bool = False
    early_stopping_patience: int = Field(default=4, ge=1)
    num_workers: int = Field(default=0, ge=0)
    seed: int = 0
    out_dir: Path = Path("weights/dino_inpaint")
    device: str = "auto"
    val_limit: int | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> InpaintTrainConfig:
        """Load a config from a YAML mapping of this model's fields."""
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)

    def augmentation(self) -> AugmentationConfig | None:
        """The loaded augmentation config, or ``None`` when ``augment`` is unset."""
        if self.augment is None:
            return None
        return AugmentationConfig.from_yaml(self.augment)

    def groups(self) -> list[str]:
        """The sampled class names, in a fixed order (``"real"`` first)."""
        return ["real", *sorted(name for name in self.sampling if name != "real")]

    def probabilities(self) -> list[float]:
        """:meth:`groups`' sampling shares, normalized to sum to 1."""
        weights = np.asarray([self.sampling[name] for name in self.groups()], dtype=np.float64)
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("sampling fractions must not all be zero")
        return [float(value) for value in weights / total]


@dataclass(frozen=True)
class CropSample:
    """One training image: where its pixels are, where its mask is, and its class.

    Plain strings and ints rather than a :class:`~imgforensics.data.manifest.ManifestEntry`
    so a list of these pickles cheaply into a dataloader worker on Windows,
    where every worker is a fresh process.
    """

    path: str
    mask_path: str | None
    group: int


class CropDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Random 448 px crops of the training images, with their patch targets.

    ``len()`` is the epoch's crop budget, not the number of images: an index
    is a *draw*, which picks a class by the configured shares, an image within
    it uniformly, and a crop position inside that image. The draw is seeded
    from ``(seed, epoch, index)``, so an epoch is reproducible, two epochs
    differ, and a worker process needs no state of its own.

    Module-level, and holding nothing but plain data, so it survives the
    pickling a Windows dataloader worker requires.
    """

    def __init__(
        self,
        samples: Sequence[CropSample],
        *,
        roots: Sequence[str],
        probabilities: Sequence[float],
        crop_size: int,
        patch: int,
        patch_target: Literal["soft", "hard"],
        positive_crop_fraction: float,
        flip_probability: float,
        augmentation: AugmentationConfig | None,
        mean: Sequence[float],
        std: Sequence[float],
        crops_per_epoch: int,
        seed: int,
    ) -> None:
        self.by_group: list[list[CropSample]] = []
        for group in range(len(roots)):
            self.by_group.append([sample for sample in samples if sample.group == group])
        self.roots = [str(root) for root in roots]
        self.probabilities = [float(value) for value in probabilities]
        self.crop_size = crop_size
        self.patch = patch
        self.patch_target = patch_target
        self.positive_crop_fraction = positive_crop_fraction
        self.flip_probability = flip_probability
        self.augmentation = augmentation
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)
        self.crops_per_epoch = crops_per_epoch
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Re-seed the draws so the next epoch sees different crops."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self.crops_per_epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng([self.seed, self.epoch, index])
        group = int(rng.choice(len(self.by_group), p=self.probabilities))
        pool = self.by_group[group]
        sample = pool[int(rng.integers(0, len(pool)))]

        pixels, mask = _read_sample(Path(self.roots[group]), sample)
        crop, target_mask = _cut_crop(
            pixels,
            mask,
            self.crop_size,
            rng,
            positive_crop_fraction=self.positive_crop_fraction,
        )
        if rng.random() < self.flip_probability:
            crop = crop[:, ::-1]
            target_mask = target_mask[:, ::-1]
        if self.augmentation is not None:
            crop = np.asarray(
                augment(Image.fromarray(crop, mode="RGB"), self.augmentation, rng), dtype=np.uint8
            )

        target = patch_targets(target_mask, self.patch, self.patch_target)
        normalized = (
            np.asarray(crop, dtype=np.float32) / 255.0 - np.asarray(self.mean, dtype=np.float32)
        ) / np.asarray(self.std, dtype=np.float32)
        return (
            torch.from_numpy(normalized.transpose(2, 0, 1).copy()),
            torch.from_numpy(target[None, ...].copy()),
        )


def _read_sample(root: Path, sample: CropSample) -> tuple[np.ndarray, np.ndarray]:
    """One training image as ``(HxWx3 uint8 pixels, HxW bool mask)``.

    A sample with no mask -- every authentic image -- gets an all-zero one of
    the right shape, which is its target: nothing in it was inpainted.
    """
    from imgforensics.utils.image_io import load_image

    pixels = np.asarray(load_image(root / sample.path).convert("RGB"), dtype=np.uint8)
    if sample.mask_path is None:
        return pixels, np.zeros(pixels.shape[:2], dtype=bool)

    with Image.open(root / sample.mask_path) as handle:
        mask = np.asarray(handle.convert("L"), dtype=np.uint8) > 127
    if mask.shape != pixels.shape[:2]:
        resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (pixels.shape[1], pixels.shape[0]), Image.Resampling.NEAREST
        )
        mask = np.asarray(resized) > 127
    return pixels, mask


def _cut_crop(
    pixels: np.ndarray,
    mask: np.ndarray,
    size: int,
    rng: np.random.Generator,
    *,
    positive_crop_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One ``size`` x ``size`` crop of an image and the matching piece of its mask.

    With probability ``positive_crop_fraction`` -- and only when the mask has
    anything in it -- the crop is placed to cover a uniformly drawn mask
    pixel. An image smaller than ``size`` is reflection-padded to fit (no TGIF
    image is, so this is a guard), its mask with zeros rather than a
    reflection, which would invent inpainted area that is not there.
    """
    height, width = pixels.shape[:2]
    contains: tuple[int, int] | None = None
    if positive_crop_fraction > 0.0 and rng.random() < positive_crop_fraction:
        positions = np.flatnonzero(mask)
        if positions.size:
            chosen = int(positions[int(rng.integers(0, positions.size))])
            contains = (chosen // width, chosen % width)

    top, left = sample_crop_box(height, width, size, rng, contains=contains)
    crop = pixels[top : top + size, left : left + size]
    crop_mask = mask[top : top + size, left : left + size]
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = _pad_to_at_least(crop, size)[:size, :size]
        padded_mask = np.zeros((size, size), dtype=bool)
        padded_mask[: crop_mask.shape[0], : crop_mask.shape[1]] = crop_mask
        crop_mask = padded_mask
    return np.ascontiguousarray(crop), np.ascontiguousarray(crop_mask)


def inpaint_loss(
    logits: torch.Tensor, targets: torch.Tensor, pos_weight: float, dice_weight: float
) -> torch.Tensor:
    """Weighted BCE on the patch grid plus a soft-Dice term.

    The two answer different failure modes. BCE with a ``pos_weight`` above 1
    counteracts a target whose mean is about 0.1 -- without it the cheapest
    solution is to predict zero everywhere. Soft Dice is computed per crop
    over the whole grid, so it scores the *shape* of the prediction rather
    than its per-patch correctness, and a crop that finds the region but
    smears it is penalised where BCE alone would be nearly satisfied.
    """
    weight = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)
    if dice_weight <= 0.0:
        return loss

    probabilities = torch.sigmoid(logits).flatten(1)
    flat_targets = targets.flatten(1)
    intersection = (probabilities * flat_targets).sum(dim=1)
    union = probabilities.sum(dim=1) + flat_targets.sum(dim=1)
    dice = 1.0 - (2.0 * intersection + _DICE_EPSILON) / (union + _DICE_EPSILON)
    return loss + dice_weight * dice.mean()


class InpaintEpochRecord(BaseModel):
    """One line of ``training_log.jsonl``: what an epoch cost and what it bought."""

    epoch: int
    train_loss: float
    val_best_f1: float
    val_ap: float
    val_f1_at_threshold: float
    val_real_area: float
    lr: float
    seconds: float
    crops_per_second: float
    amp_scale: float
    """The gradient scaler's factor at the end of the epoch (see :func:`train_inpaint`).

    It is ``1.0`` on the CPU, where the scaler is disabled and float16
    autocast never runs. On CUDA it starts at 65536 and the scaler halves it
    every time a step overflows, so a value that keeps falling epoch after
    epoch -- or one that has collapsed to single digits -- is the visible
    symptom of a run whose gradients do not fit in float16. Without it, that
    failure looks exactly like a model that is not learning.
    """

    best: bool


class InpaintTrainReport(BaseModel):
    """What :func:`train_inpaint` produced: the checkpoint, where it landed, and the log."""

    out_dir: Path
    weights_path: Path
    metadata_path: Path
    log_path: Path
    meta: InpaintCheckpointMeta
    epochs: list[InpaintEpochRecord] = Field(default_factory=list)
    train_images: dict[str, int] = Field(default_factory=dict)
    crops_per_epoch: int = 0
    val_images: int = 0
    head_parameters: int = 0
    lora_parameters: int = 0
    crops_per_second: float = 0.0
    elapsed_s: float = 0.0


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


def _samples_for(
    manifest: Manifest, groups: Sequence[str], *, real_group: int
) -> tuple[list[CropSample], dict[str, int]]:
    """Turn a manifest's entries into :class:`CropSample`, dropping what cannot be used.

    A fake whose ``generator`` is not one of the sampled groups, and a fake
    with no mask, are skipped: the first has no share of the epoch to be drawn
    into and the second has no target.
    """
    index_of = {name: index for index, name in enumerate(groups)}
    samples: list[CropSample] = []
    counts: dict[str, int] = {}
    for entry in manifest.entries:
        group = real_group if entry.label == "real" else index_of.get(entry.generator or "", -1)
        if group < 0:
            continue
        if entry.label == "fake" and entry.mask_path is None:
            continue
        samples.append(CropSample(entry.path, entry.mask_path, group))
        counts[groups[group]] = counts.get(groups[group], 0) + 1
    return samples, counts


def _validation_metrics(
    backbone: nn.Module,
    head: PatchHead,
    entries: Sequence[ManifestEntry],
    root: Path,
    config: InpaintConfig,
    *,
    device: str,
    mean: Sequence[float],
    std: Sequence[float],
) -> InpaintValMetrics:
    """Score the validation subset through the inference path.

    Every entry is run exactly as ``analyze`` would run it, and both the
    heatmap and the mask are reduced to the patch grid before the metrics see
    them: comparing them at full resolution would report the bilinear
    upsampling's smoothness rather than the model's decisions, and would cost
    196 times the arithmetic to do it.

    Which half an entry belongs to is read from its ``label``, not from
    whether it has a mask. A fake with no mask is a manifest that was built
    wrongly, and treating it as authentic would quietly move it into the
    false-positive average and flatter the run; it is skipped and counted
    instead, and the count is printed once at the end.
    """
    from imgforensics.utils.image_io import load_image

    bundles: list[PixelMetrics] = []
    real_areas: list[float] = []
    unusable = 0
    for entry in entries:
        if entry.label == "fake" and entry.mask_path is None:
            unusable += 1
            continue
        pixels = np.asarray(load_image(root / entry.path).convert("RGB"), dtype=np.uint8)
        heatmap, _ = infer_heatmap(
            backbone, head, pixels, config, device=device, mean=mean, std=std
        )
        small = patch_targets(heatmap, config.patch, "soft")
        if entry.label == "real":
            real_areas.append(float((small > MASK_THRESHOLD).mean()))
            continue
        assert entry.mask_path is not None  # guarded above
        with Image.open(root / entry.mask_path) as handle:
            mask = np.asarray(handle.convert("L"), dtype=np.uint8) > 127
        if mask.shape != heatmap.shape:
            resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (heatmap.shape[1], heatmap.shape[0]), Image.Resampling.NEAREST
            )
            mask = np.asarray(resized) > 127
        bundles.append(PixelMetrics.compute(patch_targets(mask, config.patch, "hard") > 0.5, small))

    if unusable:
        print(f"[train] {unusable} validation fake(s) have no mask and were skipped")

    def _mean(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return InpaintValMetrics(
        best_f1=_mean([bundle.best_f1 for bundle in bundles]),
        ap=_mean([bundle.ap for bundle in bundles]),
        f1_at_threshold=_mean([bundle.f1_at_threshold for bundle in bundles]),
        iou=_mean([bundle.iou for bundle in bundles]),
        real_area_above_threshold=_mean(real_areas),
        fake_entries=len(bundles),
        real_entries=len(real_areas),
    )


def _write_log(path: Path, records: Sequence[InpaintEpochRecord]) -> None:
    """Write one JSON object per epoch, in order, replacing whatever was there.

    Called after *every* epoch rather than once at the end: a run that takes
    hours is otherwise invisible until it finishes, and a crash takes the
    whole history with it. Rewriting the file whole (instead of appending)
    keeps it consistent with the in-memory list even if an earlier write was
    interrupted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def _write_meta(path: Path, meta: InpaintCheckpointMeta) -> None:
    """Write ``inpaint.json`` -- sorted keys, so two runs diff cleanly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(meta.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_meta(
    config: InpaintTrainConfig,
    *,
    groups: Sequence[str],
    probabilities: Sequence[float],
    augmentation: AugmentationConfig | None,
    lora: LoraConfig | None,
    manifests: Sequence[ManifestRef],
    best_epoch: int,
    epochs_run: int,
    val: InpaintValMetrics,
) -> InpaintCheckpointMeta:
    """Describe the checkpoint as it stands after ``epochs_run`` epochs.

    Built once per improving epoch and once more at the end, so a run that
    dies halfway leaves an ``inpaint.json`` describing the best epoch it
    actually reached rather than no file at all. ``created`` is therefore the
    time this file was written, not the time the run started.
    """
    return InpaintCheckpointMeta(
        package_version=__version__,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        backbone=config.model.backbone,
        layers=list(config.model.layers),
        crop_size=config.model.crop_size,
        stride=config.model.stride,
        patch=config.model.patch,
        head=config.model.head_config(),
        lora=lora,
        train=TrainEcho(
            sampling=dict(zip(groups, probabilities, strict=True)),
            positive_crop_fraction=config.positive_crop_fraction,
            patch_target=config.patch_target,
            pos_weight=config.pos_weight,
            dice_weight=config.dice_weight,
            flip_probability=config.flip_probability,
            augment_config=config.augment,
            augment_hash="none" if augmentation is None else augmentation.fingerprint(),
            crops_per_epoch=config.crops_per_epoch,
            batch_size=config.batch_size,
            lr=config.lr,
            lora_lr=None if config.lora is None else config.lora_lr,
            seed=config.seed,
        ),
        best_epoch=best_epoch,
        epochs_run=epochs_run,
        val=val,
        manifests=list(manifests),
        commercial_ok=_combine_commercial_ok([ref.commercial_ok for ref in manifests]),
        licenses=sorted({ref.license for ref in manifests if ref.license}),
    )


def _warm_start(head: PatchHead, checkpoint_dir: Path) -> None:
    """Load a previous run's head weights into ``head`` (stage 2's starting point)."""
    from safetensors.torch import load_file

    tensors = load_file(str(checkpoint_dir / WEIGHTS_FILENAME))
    head.load_state_dict(
        {
            name[len(HEAD_PREFIX) :]: value
            for name, value in tensors.items()
            if name.startswith(HEAD_PREFIX)
        },
        strict=True,
    )


def train_inpaint(config: InpaintTrainConfig, progress: bool = True) -> InpaintTrainReport:
    """Train a :class:`~imgforensics.localization._inpaint_model.PatchHead` and save it.

    Draws ``crops_per_epoch`` crops per epoch for up to ``config.epochs``
    epochs, keeping the weights of the best mean validation best-F1, and
    writes ``inpaint.safetensors``, ``inpaint.json`` and
    ``training_log.jsonl`` into ``config.out_dir``.

    Raises:
        ValueError: a manifest is empty, or a sampled class ended up with no
            usable image (which would make its share of every epoch a draw
            from nothing).
    """
    started = time.perf_counter()
    groups = config.groups()
    probabilities = config.probabilities()
    device = resolve_device(config.device)
    augmentation = config.augmentation()

    fake_manifest = Manifest.load(config.train_fake_manifest)
    real_manifest = Manifest.load(config.train_real_manifest)
    val_manifest = Manifest.load(config.val_manifest)

    fake_samples, fake_counts = _samples_for(fake_manifest, groups, real_group=-1)
    real_samples, real_counts = _samples_for(real_manifest, groups, real_group=0)
    counts = {**real_counts, **fake_counts}
    missing = [name for name in groups if counts.get(name, 0) == 0]
    if missing:
        raise ValueError(
            f"no usable images for sampled class(es) {', '.join(missing)}; "
            "check the manifests' generators and masks"
        )

    val_entries = list(val_manifest.entries)
    if config.val_limit is not None:
        val_entries = val_entries[: config.val_limit]
    if not val_entries:
        raise ValueError(f"{config.val_manifest} has no entries to validate on")

    torch.manual_seed(config.seed)
    backbone, device, mean, std = load_patch_backbone(config.model, device)
    head = PatchHead(config.model.head_config()).to(torch.device(device))
    if config.warm_start is not None:
        _warm_start(head, Path(config.warm_start))

    lora_meta: LoraConfig | None = None
    parameter_groups: list[dict[str, Any]] = [{"params": list(head.parameters()), "lr": config.lr}]
    if config.lora is not None:
        modules = attach_lora(backbone, config.lora)
        backbone.to(torch.device(device))
        lora_meta = config.lora.model_copy(update={"modules": modules})
        lora_parameters = [
            parameter for parameter in backbone.parameters() if parameter.requires_grad
        ]
        parameter_groups.append({"params": lora_parameters, "lr": config.lora_lr})
        if config.grad_checkpointing:
            net: Any = backbone
            net.set_grad_checkpointing(True)
    else:
        # Stage 1 trains the head alone. load_backbone already clears every
        # requires_grad, but stating the invariant here keeps it local to the
        # loop that depends on it -- and keeps the report's lora_parameters
        # count from picking up a backbone that arrived unfrozen.
        backbone.requires_grad_(False)

    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    # The forward pass runs in float16 on CUDA, and a float16 gradient that
    # underflows becomes a zero without any error. The scaler multiplies the
    # loss by a large factor before the backward pass, unscales the gradients
    # inside optimizer.step(), and halves the factor whenever a step overflows
    # -- so both the head and, in stage 2, the LoRA factors are stepped through
    # it, since they share one optimizer. Disabled on the CPU, where autocast
    # is off and the whole loop is float32 anyway.
    torch_device = torch.device(device)
    scaler = torch.amp.GradScaler("cuda", enabled=torch_device.type == "cuda")

    dataset = CropDataset(
        [*real_samples, *fake_samples],
        roots=[real_manifest.meta.root, *([fake_manifest.meta.root] * (len(groups) - 1))],
        probabilities=probabilities,
        crop_size=config.model.crop_size,
        patch=config.model.patch,
        patch_target=config.patch_target,
        positive_crop_fraction=config.positive_crop_fraction,
        flip_probability=config.flip_probability,
        augmentation=augmentation,
        mean=mean,
        std=std,
        crops_per_epoch=config.crops_per_epoch,
        seed=config.seed,
    )

    # Built before the loop so an improving epoch can write a complete
    # checkpoint the moment it happens, not only when the run finishes.
    manifests = [
        _manifest_ref(
            "train_fake", Path(config.train_fake_manifest), fake_manifest, len(fake_samples)
        ),
        _manifest_ref(
            "train_real", Path(config.train_real_manifest), real_manifest, len(real_samples)
        ),
        _manifest_ref("val", Path(config.val_manifest), val_manifest, len(val_entries)),
    ]
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / WEIGHTS_FILENAME
    metadata_path = out_dir / METADATA_FILENAME
    log_path = out_dir / LOG_FILENAME

    best_score = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = copy.deepcopy(head.state_dict())
    best_lora: dict[str, torch.Tensor] = _lora_state(backbone)
    best_metrics: InpaintValMetrics | None = None
    since_best = 0
    records: list[InpaintEpochRecord] = []

    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        dataset.set_epoch(epoch)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
            persistent_workers=False,
        )

        head.train()
        backbone.eval()  # frozen, and its norm statistics must stay frozen too
        # ...except the adapters, which are the only part of the backbone that
        # trains: eval() would switch their dropout off and silently make the
        # rate the checkpoint records a fiction.
        set_lora_training(backbone, True)
        total_loss = 0.0
        steps = 0
        for crops, targets in loader:
            crops = crops.to(torch_device, non_blocking=True)
            targets = targets.to(torch_device, non_blocking=True)
            with torch.autocast(
                device_type=torch_device.type,
                dtype=torch.float16,
                enabled=torch_device.type == "cuda",
            ):
                if config.lora is None:
                    with torch.no_grad():
                        features = patch_features(backbone, crops, config.model.layers)
                    logits = head(features.float())
                else:
                    logits = head(patch_features(backbone, crops, config.model.layers).float())
            loss = inpaint_loss(logits.float(), targets, config.pos_weight, config.dice_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.item())
            steps += 1

        learning_rate = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        train_seconds = time.perf_counter() - epoch_started

        metrics = _validation_metrics(
            backbone,
            head,
            val_entries,
            Path(val_manifest.meta.root),
            config.model,
            device=device,
            mean=mean,
            std=std,
        )
        improved = metrics.best_f1 > best_score
        if improved:
            best_score, best_epoch, since_best = metrics.best_f1, epoch, 0
            best_state = copy.deepcopy(head.state_dict())
            best_lora = _lora_state(backbone)
            best_metrics = metrics
        else:
            since_best += 1

        records.append(
            InpaintEpochRecord(
                epoch=epoch,
                train_loss=total_loss / max(steps, 1),
                val_best_f1=metrics.best_f1,
                val_ap=metrics.ap,
                val_f1_at_threshold=metrics.f1_at_threshold,
                val_real_area=metrics.real_area_above_threshold,
                lr=learning_rate,
                seconds=time.perf_counter() - epoch_started,
                crops_per_second=config.crops_per_epoch / max(train_seconds, 1e-9),
                amp_scale=float(scaler.get_scale()),
                best=improved,
            )
        )
        record = records[-1]
        _write_log(log_path, records)
        if improved:
            # The live model *is* the best one right now, so the checkpoint on
            # disk is always the best epoch so far -- a run killed at hour
            # three leaves something loadable behind instead of nothing.
            _save_tensors(checkpoint_tensors(head, backbone if lora_meta else None), weights_path)
            _write_meta(
                metadata_path,
                _build_meta(
                    config,
                    groups=groups,
                    probabilities=probabilities,
                    augmentation=augmentation,
                    lora=lora_meta,
                    manifests=manifests,
                    best_epoch=best_epoch,
                    epochs_run=len(records),
                    val=metrics,
                ),
            )
        if progress:
            # One line per epoch, flushed: a redirected stdout is block
            # buffered, so an unflushed print leaves a multi-hour run looking
            # hung in the log file it is being watched through.
            print(
                f"[train] epoch {epoch}/{config.epochs} "
                f"loss={record.train_loss:.4f} best_f1={metrics.best_f1:.4f} "
                f"ap={metrics.ap:.4f} f1@0.5={metrics.f1_at_threshold:.4f} "
                f"real_area={metrics.real_area_above_threshold:.4f} "
                f"{record.crops_per_second:.1f} crops/s {record.seconds:.1f}s "
                f"amp_scale={record.amp_scale:g}"
                f"{' *' if improved else ''}",
                flush=True,
            )
        if since_best >= config.early_stopping_patience:
            break

    head.load_state_dict(best_state)
    _restore_lora(backbone, best_lora)
    assert best_metrics is not None  # at least one epoch always runs

    # The best epoch already wrote these; this rewrites the same tensors from
    # the restored best state and refreshes the epoch count, so the file is
    # identical in substance whether or not the run reached its last epoch.
    meta = _build_meta(
        config,
        groups=groups,
        probabilities=probabilities,
        augmentation=augmentation,
        lora=lora_meta,
        manifests=manifests,
        best_epoch=best_epoch,
        epochs_run=len(records),
        val=best_metrics,
    )
    _save_tensors(checkpoint_tensors(head, backbone if lora_meta else None), weights_path)
    _write_meta(metadata_path, meta)
    _write_log(log_path, records)

    return InpaintTrainReport(
        out_dir=out_dir,
        weights_path=weights_path,
        metadata_path=metadata_path,
        log_path=log_path,
        meta=meta,
        epochs=records,
        train_images=counts,
        crops_per_epoch=config.crops_per_epoch,
        val_images=len(val_entries),
        head_parameters=sum(parameter.numel() for parameter in head.parameters()),
        lora_parameters=sum(
            parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad
        ),
        crops_per_second=float(np.mean([record.crops_per_second for record in records])),
        elapsed_s=time.perf_counter() - started,
    )


def _lora_state(backbone: nn.Module) -> dict[str, torch.Tensor]:
    """A deep copy of every attached adapter's factors, for the best-epoch snapshot."""
    state: dict[str, torch.Tensor] = {}
    for name, module in backbone.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_a"] = module.lora_a.detach().clone()
            state[f"{name}.lora_b"] = module.lora_b.detach().clone()
    return state


def _restore_lora(backbone: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Put a snapshot from :func:`_lora_state` back into the attached adapters."""
    if not state:
        return
    with torch.no_grad():
        for name, module in backbone.named_modules():
            if isinstance(module, LoRALinear):
                module.lora_a.copy_(state[f"{name}.lora_a"])
                module.lora_b.copy_(state[f"{name}.lora_b"])


def _save_tensors(tensors: dict[str, Any], path: Path) -> None:
    """Write a checkpoint's tensors to ``path`` in the safetensors format.

    :func:`imgforensics.detectors.train._save_weights` does the same thing for
    a whole module; this one takes the dict
    :func:`~imgforensics.localization._inpaint_model.checkpoint_tensors` built,
    because what belongs in this file is the head *plus* the backbone's LoRA
    factors and nothing else -- neither module's ``state_dict`` is that set.
    """
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
