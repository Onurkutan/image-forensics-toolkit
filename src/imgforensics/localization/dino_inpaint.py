"""A trained patch head over a frozen DINOv2, as the ``dino_inpaint`` localizer.

Phase 4b of ``docs/ROADMAP.md``. The two pretrained localizers of Phase 4a
answer the splicing question well and the *inpainting* question badly: on
TGIF's fully regenerated subsets CAT-Net's best-F1 falls to 0.33 and IML-ViT's
to 0.21 (``docs/benchmarks/09_tgif_localizers_summary.md``), a few points above
a predict-everything baseline. Both were trained on spliced composites, where
the evidence is a seam between two capture pipelines; a diffusion model that
regenerates a whole region leaves no seam, only a patch of the image whose
texture statistics came from a decoder rather than from a sensor. That is a
different question, and it needs a model trained on it.

The architecture is the cheapest thing that can answer it, and it is the same
recipe :mod:`imgforensics.detectors.head` already uses at image level, moved
down to the patch grid:

- **A frozen DINOv2 ViT-B/14 at 448 px**, read at blocks 5, 8 and 11 through
  timm's ``forward_intermediates(..., output_fmt="NCHW")``, which hands back
  the patch tokens as a ``(B, 768, 32, 32)`` feature map per block rather than
  a CLS vector. Three depths concatenated is 2304 channels per patch. Nothing
  in the backbone trains, so one 6 GB card runs it at about 35 crops/s under
  float16 autocast.
- **A fully convolutional head** (:class:`~imgforensics.localization._inpaint_model.PatchHead`,
  about 1.19 M parameters): a LayerNorm per source block, a 1x1 projection to
  256, one 3x3 convolution so a patch can see its neighbours, and a 1x1
  classifier. The loss is computed on the 32x32 patch grid against the mask's
  own per-patch area, and the logits are upsampled 14x only at inference --
  training on the upsampled map would spend every gradient on the 196 pixels
  a patch cannot distinguish between.

**The image is never resized.** Inference tiles it at :data:`CROP_SIZE` with
stride :data:`TILE_STRIDE` and averages the overlaps, exactly as
:mod:`imgforensics.localization.catnet` does, because the patch grid a ViT
sees is defined in pixels: a resized image changes what one patch covers and
therefore what the head was trained to recognise. An image smaller than a tile
on some axis is reflection-padded up to the next whole patch rather than
stretched (a 256x256 CocoGlide image becomes one 266 px tile), and the padding
is cropped off the finished heatmap.

**Image-level score.** The same rule as every other localizer here: the mean
of the top :data:`~imgforensics.localization._scoring.TOP_FRACTION` of the
heatmap (:func:`~imgforensics.localization._scoring.top_fraction_score`), so
the four of them stay comparable on one benchmark table.

**Checkpoints are trained, not downloaded** -- see
:mod:`imgforensics.localization.train_inpaint` and ``imgforensics train
localizer``. With none installed the detector abstains (0.5, ``"uncertain"``,
a ``details["reason"]`` naming the directory and the training command) rather
than failing a run, and the checkpoint records which manifests it was fitted
on, under which licenses: a model trained on TGIF is research-only, and the
``commercial_ok`` flag is how that survives the move from data to weights.

Like every other module in this package, this one imports no ``torch`` at
module scope: it is imported for its registration side effect by
:mod:`imgforensics.localization`, and the heavy imports sit inside
:meth:`DinoInpaintLocalizer.load`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

from imgforensics.core.base import BaseDetector
from imgforensics.core.image import ForensicImage
from imgforensics.core.registry import register
from imgforensics.core.types import DetectionResult, ToolKind, label_from_score
from imgforensics.localization._scoring import TOP_FRACTION, top_fraction_score

if TYPE_CHECKING:  # pragma: no cover - import-time typing only, never at runtime
    from imgforensics.localization._inpaint_model import PatchHead

#: Edge length of the square the backbone is fed, and of an inference tile.
#: 448 px is 32x32 patches of 14, the largest grid that leaves room for
#: gradients through the attention projections on a 6 GB card.
CROP_SIZE = 448

#: Step between tile origins, 24 patches: a 112 px overlap, averaged.
TILE_STRIDE = 336

#: The backbone's patch size. Every grid dimension, every padding target and
#: every upsampling factor in this module is this number.
PATCH = 14

#: Training crops snap their top-left corner down to a multiple of this,
#: mirroring :data:`imgforensics.detectors.crops._WINDOW_ALIGN`'s reasoning:
#: a crop that starts mid-block re-phases the JPEG grid its own augmentation
#: then re-encodes on.
CROP_ALIGN = 8

#: Environment variable naming the directory a trained localizer loads from.
INPAINT_DIR_ENV = "IMGFORENSICS_INPAINT_DIR"

#: Fallback checkpoint directory, relative to the working directory.
DEFAULT_INPAINT_DIR = Path("weights/dino_inpaint")

#: File names inside a checkpoint directory.
WEIGHTS_FILENAME = "inpaint.safetensors"
METADATA_FILENAME = "inpaint.json"
LOG_FILENAME = "training_log.jsonl"

#: Probability above which a patch counts as inpainted, matching this
#: project's pixel metrics (:func:`imgforensics.eval.metrics.pixel_f1`).
MASK_THRESHOLD = 0.5

_ABSTAIN_SCORE = 0.5
_TRAIN_HINT = (
    "imgforensics train localizer --config configs/experiments/4b_dino_inpaint_stage1.yaml"
)
_SHA_PREFIX_LENGTH = 12


class PatchHeadConfig(BaseModel):
    """Shape and width of the convolutional head over the patch grid.

    Attributes:
        dim: Backbone feature width per block (768 for a ViT-B).
        n_layers: How many blocks' feature maps are concatenated; the head's
            input therefore has ``n_layers * dim`` channels.
        proj_dim: Width the concatenated features are projected to.
        context_kernel: Edge length of the one convolution that lets a patch
            see its neighbours. ``1`` disables it -- the declared ablation,
            which says whether the head is reading local context or only the
            patch it is classifying.
    """

    dim: int = Field(default=768, gt=0)
    n_layers: int = Field(default=3, gt=0)
    proj_dim: int = Field(default=256, gt=0)
    context_kernel: int = Field(default=3, ge=1)


class LoraConfig(BaseModel):
    """Low-rank adapters on the backbone's attention projections (stage 2).

    Stage 1 trains the head alone over a fully frozen backbone. If that lands
    in the partial band, stage 2 unfreezes the backbone the cheapest way there
    is: a rank-``rank`` update on every block's ``attn.qkv``, which is 294,912
    trainable parameters against the backbone's 86 M and adds nothing at all
    to the saved file beyond its own A/B factors.

    ``modules`` is filled in by
    :func:`~imgforensics.localization._inpaint_model.attach_lora` and recorded
    in the checkpoint, so a file says exactly which projections it patches.
    """

    rank: int = Field(default=8, gt=0)
    alpha: float = Field(default=16.0, gt=0.0)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    modules: list[str] = Field(default_factory=list)


class TrainEcho(BaseModel):
    """The training choices a checkpoint has to remember to be reproducible.

    Not the whole config -- manifest paths and the output directory are
    recorded elsewhere -- but everything that decides what the head actually
    saw: how crops were drawn across the three classes, how often a crop was
    steered onto the mask, whether the target was the patch's mask *area* or a
    thresholded version of it, and how the two loss terms were weighted.
    """

    sampling: dict[str, float] = Field(default_factory=dict)
    positive_crop_fraction: float = 0.5
    patch_target: Literal["soft", "hard"] = "soft"
    pos_weight: float = 3.0
    dice_weight: float = 0.5
    flip_probability: float = 0.5
    augment_config: str | None = None
    augment_hash: str
    crops_per_epoch: int
    batch_size: int
    lr: float
    lora_lr: float | None = None
    seed: int = 0


class ManifestRef(BaseModel):
    """One manifest a localizer was trained or validated on, for its checkpoint.

    Field for field :class:`imgforensics.detectors.head.ManifestRef`, and read
    the same way: the ``sha256`` is of the manifest *file*, so a checkpoint
    can be matched against the exact entry list it saw, and ``license`` /
    ``commercial_ok`` travel with the weights because a model trained on
    research-only data is itself research-only. It is redeclared here rather
    than imported because ``detectors/head.py`` imports ``torch`` at module
    scope -- it declares an ``nn.Module`` -- and this module must stay
    torch-free, being imported on every CLI invocation.
    """

    role: str
    path: str
    dataset: str
    entries: int
    sha256: str
    license: str | None = None
    commercial_ok: bool | None = None


class InpaintValMetrics(BaseModel):
    """Validation metrics of the selected epoch, on the 14x-downsampled grid.

    Every number but ``real_area_above_threshold`` is a mean over the
    *fully regenerated fakes* of the validation subset, computed by
    :meth:`imgforensics.eval.metrics.PixelMetrics.compute` on one (mask,
    heatmap) pair per image. ``best_f1`` is the model-selection metric: a
    threshold-free summary, so the epoch chosen does not depend on how
    mis-scaled that epoch's logits happen to be.

    ``real_area_above_threshold`` is the mean fraction of an *authentic*
    image's patches the model puts above 0.5 -- the false-positive number the
    fake-only metrics cannot show, and the one that decides whether the
    localizer is usable on a real photograph.
    """

    best_f1: float
    ap: float
    f1_at_threshold: float
    iou: float
    threshold: float = MASK_THRESHOLD
    real_area_above_threshold: float = 0.0
    fake_entries: int = 0
    real_entries: int = 0


class InpaintConfig(BaseModel):
    """What the backbone and head are, and how the image is cut up for them.

    Shared by training and inference so the two cannot drift: a head fitted on
    448 px crops of blocks 5/8/11 is only meaningful when it is fed the same
    thing, which is why every one of these fields is written into the
    checkpoint and read back out of it rather than defaulted at load time.
    """

    backbone: str = "dinov2_vitb14"
    layers: list[int] = Field(default_factory=lambda: [5, 8, 11])
    crop_size: int = Field(default=CROP_SIZE, gt=0)
    stride: int = Field(default=TILE_STRIDE, gt=0)
    patch: int = Field(default=PATCH, gt=0)
    head: PatchHeadConfig = Field(default_factory=PatchHeadConfig)

    def head_config(self) -> PatchHeadConfig:
        """The head's config with ``n_layers`` forced to match :attr:`layers`.

        The number of source blocks is not a free choice -- it is however many
        the backbone is read at -- so it is derived here instead of being
        configured twice and left to disagree.
        """
        return self.head.model_copy(update={"n_layers": len(self.layers)})

    @property
    def grid(self) -> int:
        """Patches per side of one crop: ``crop_size / patch``."""
        return self.crop_size // self.patch


class InpaintCheckpointMeta(BaseModel):
    """``inpaint.json``: everything needed to reuse a trained localizer, and to audit it.

    Modelled on :class:`imgforensics.detectors.head.CheckpointMeta` and
    self-describing for the same reason: the tensors alone are meaningless
    without the backbone, the blocks they read and the resolution they were
    fitted at, so :class:`DinoInpaintLocalizer` rebuilds its whole inference
    pipeline from this file instead of assuming any of it.
    """

    format_version: int = 1
    package_version: str
    created: str
    backbone: str
    layers: list[int] = Field(default_factory=list)
    crop_size: int = CROP_SIZE
    stride: int = TILE_STRIDE
    patch: int = PATCH
    head: PatchHeadConfig
    lora: LoraConfig | None = None
    train: TrainEcho
    best_epoch: int
    epochs_run: int
    val: InpaintValMetrics
    manifests: list[ManifestRef] = Field(default_factory=list)
    commercial_ok: bool | None = None
    licenses: list[str] = Field(default_factory=list)

    def config(self) -> InpaintConfig:
        """The inference-time half of this checkpoint, as an :class:`InpaintConfig`."""
        return InpaintConfig(
            backbone=self.backbone,
            layers=list(self.layers),
            crop_size=self.crop_size,
            stride=self.stride,
            patch=self.patch,
            head=self.head,
        )


def tile_origins(extent: int, tile: int, stride: int = TILE_STRIDE) -> list[int]:
    """Tile start offsets covering ``extent`` pixels with ``tile``-wide windows.

    :func:`imgforensics.localization.catnet._tile_origins`, re-derived for a
    patch-14 model: one origin at 0 when the extent fits in a single tile,
    otherwise origins every ``stride`` pixels with the last pulled back to
    ``extent - tile`` so every tile is full-size and only the final overlap is
    wider than the rest.

    :data:`TILE_STRIDE` is 24 patches, so every origin but the pulled-back
    last one is a multiple of :data:`PATCH`; the last one is too whenever
    ``extent`` is, which :func:`pad_for_tiles` guarantees for any image small
    enough to need padding at all. A larger image whose side is not a multiple
    of 14 gets one tile whose patch grid is offset against the others', which
    costs nothing: each tile is evaluated on its own grid and its output is
    upsampled back to pixels before the overlaps are averaged.
    """
    if extent <= tile:
        return [0]
    origins = list(range(0, extent - tile, stride))
    origins.append(extent - tile)
    return origins


def pad_for_tiles(
    pixels: np.ndarray, tile: int = CROP_SIZE, patch: int = PATCH
) -> tuple[np.ndarray, tuple[int, int]]:
    """Reflection-pad ``pixels`` until every axis can hold a whole patch grid.

    Each axis is treated on its own: an axis at or above ``tile`` is left
    exactly as it is, and a shorter one grows to the next multiple of
    ``patch`` and no further -- a 256x256 image becomes 266x266, one tile,
    rather than being stretched to 448, and a 256x384 image becomes 266x392
    rather than a 392x392 square. The padding is invented pixels, so the model
    should see as few of them as possible, and the amount differs per axis.

    The border is ``cv2.BORDER_REFLECT_101``, the mode
    :func:`imgforensics.eval.preprocess._pad_to_at_least` uses; that helper
    itself is not reused here because it takes one size for *both* axes, which
    is exactly the over-padding this function has to avoid.

    Returns:
        The padded array and the ``(top, left)`` offset of the original image
        within it, which is what the finished heatmap is cropped back to.
    """
    height, width = pixels.shape[:2]
    pad_h = max(0, -(-height // patch) * patch - height) if height < tile else 0
    pad_w = max(0, -(-width // patch) * patch - width) if width < tile else 0
    if pad_h == 0 and pad_w == 0:
        return pixels, (0, 0)

    top, left = pad_h // 2, pad_w // 2
    padded = cv2.copyMakeBorder(
        pixels, top, pad_h - top, left, pad_w - left, cv2.BORDER_REFLECT_101
    )
    return padded, (top, left)


def stitch_tiles(
    tiles: Sequence[tuple[int, int, np.ndarray]],
    shape: tuple[int, int],
    *,
    crop: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Average overlapping tile maps into one map of ``shape``, then crop it.

    ``tiles`` is a sequence of ``(top, left, map)``; every pixel covered by
    more than one tile gets the mean of their values, which is
    :mod:`imgforensics.localization.catnet`'s rule and the only one that does
    not put a visible seam at every stride boundary. Accumulation is in
    float64 and the result is clipped into ``[0, 1]`` and returned as float32.

    Args:
        tiles: The tile maps and where they sit in the padded image.
        shape: ``(height, width)`` of the padded image the tiles came from.
        crop: ``(top, left, height, width)`` of the original image within that
            padded one (see :func:`pad_for_tiles`). ``None`` keeps everything.

    Raises:
        ValueError: ``tiles`` is empty, or leaves part of ``shape`` uncovered.
    """
    if not tiles:
        raise ValueError("stitch_tiles() requires at least one tile")

    height, width = shape
    totals = np.zeros((height, width), dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.float64)
    for top, left, values in tiles:
        bottom, right = top + values.shape[0], left + values.shape[1]
        totals[top:bottom, left:right] += values
        counts[top:bottom, left:right] += 1.0

    if not counts.all():
        raise ValueError(
            f"stitch_tiles() left {int((counts == 0).sum())} of {counts.size} pixels uncovered"
        )

    stitched = np.clip(totals / counts, 0.0, 1.0)
    if crop is not None:
        top, left, crop_height, crop_width = crop
        stitched = stitched[top : top + crop_height, left : left + crop_width]
    return stitched.astype(np.float32)


def patch_targets(
    mask: np.ndarray, patch: int = PATCH, mode: Literal["soft", "hard"] = "soft"
) -> np.ndarray:
    """Reduce a pixel map to one value per ``patch`` x ``patch`` block.

    The training target: a patch is not inpainted or authentic, it is *some
    fraction* inpainted, and that fraction is what the head is asked to
    predict (``"soft"``). ``"hard"`` thresholds it at 0.5 afterwards -- the
    declared ablation, which throws away the boundary patches' gradient in
    exchange for a target that matches the metric.

    The same reduction takes a heatmap and a ground-truth mask down to the
    grid the validation metrics are computed on, which is why this takes any
    float array and not only a binary mask. Rows and columns past the last
    whole patch are dropped rather than partially averaged, so a 683 px side
    contributes 48 patches and forgets 11 pixels.

    Returns:
        A float32 array of shape ``(height // patch, width // patch)``.
    """
    values = np.asarray(mask, dtype=np.float32)
    height = values.shape[0] // patch * patch
    width = values.shape[1] // patch * patch
    blocks = values[:height, :width].reshape(height // patch, patch, width // patch, patch)
    averaged = blocks.mean(axis=(1, 3))
    if mode == "hard":
        return (averaged > 0.5).astype(np.float32)
    return averaged


def sample_crop_box(
    height: int,
    width: int,
    size: int,
    rng: np.random.Generator,
    *,
    contains: tuple[int, int] | None = None,
    align: int = CROP_ALIGN,
) -> tuple[int, int]:
    """Top-left corner of one ``size`` x ``size`` training crop.

    Without ``contains`` the corner is drawn uniformly over the aligned
    positions that fit inside the image. With ``contains = (y, x)`` it is
    drawn uniformly over the aligned positions whose crop *covers* that pixel
    -- every one of them, not the one that centers it. That distinction is the
    whole point of the guided draw: centering every mask pixel would teach the
    head that inpainting lives in the middle of a crop, which is exactly the
    shortcut a 32x32 patch grid is able to learn.

    Corners are multiples of ``align`` for the same reason
    :data:`imgforensics.detectors.crops._WINDOW_ALIGN` exists -- the crop's
    own JPEG augmentation re-encodes on an 8x8 grid, and a crop that starts
    mid-block puts that grid in a different phase for every sample. In the one
    case where no aligned corner covers the requested pixel (it is within
    ``align`` pixels of the far edge *and* the image's own ``extent - size``
    is not aligned) the largest covering corner is used unaligned, because
    covering the pixel is the guarantee a caller asked for and the alignment
    is an optimization.

    An axis shorter than ``size`` yields 0 on that axis; the caller is
    expected to pad the resulting crop (no TGIF image is smaller than 512 px,
    so this is a guard rather than a path).

    Returns:
        A ``(top, left)`` pair.
    """
    return (
        _sample_offset(height, size, rng, None if contains is None else contains[0], align),
        _sample_offset(width, size, rng, None if contains is None else contains[1], align),
    )


def _sample_offset(
    extent: int, size: int, rng: np.random.Generator, cover: int | None, align: int
) -> int:
    """One axis of :func:`sample_crop_box`."""
    if extent <= size:
        return 0
    low, high = 0, extent - size
    if cover is not None:
        low = max(low, cover - size + 1)
        high = min(high, cover)
    first = -(-low // align) * align
    if first > high:  # no aligned corner covers the pixel; see sample_crop_box
        return high
    choices = (high - first) // align + 1
    return first + align * int(rng.integers(0, choices))


def resolve_checkpoint_dir(checkpoint_dir: str | Path | None = None) -> Path:
    """The directory a localizer would load from: argument, then env var, then default."""
    if checkpoint_dir is not None:
        return Path(checkpoint_dir)
    from_env = os.environ.get(INPAINT_DIR_ENV)
    if from_env:
        return Path(from_env)
    return DEFAULT_INPAINT_DIR


@register("dino_inpaint")
class DinoInpaintLocalizer(BaseDetector):
    """Localizes AI-inpainted regions with a trained patch head over frozen DINOv2.

    One prediction is: pad the image up to whole patches if it is smaller than
    a tile, cut it into :data:`CROP_SIZE` tiles at :data:`TILE_STRIDE`, run the
    frozen backbone and the head over each, upsample every tile's patch
    probabilities back to pixels, average the overlaps and crop off the
    padding. The heatmap *is* the output; the image-level score is the mean of
    its top 1% (see the module docstring).

    Model and head are loaded once per instance and reused across
    :meth:`predict` calls, so the benchmark runner -- one instance, many
    images -- pays the backbone load once.
    """

    name = "dino_inpaint"
    kind: ClassVar[ToolKind] = "localizer"

    def __init__(self, checkpoint_dir: str | Path | None = None, device: str = "auto") -> None:
        """Point the localizer at a checkpoint directory (nothing is read yet).

        Args:
            checkpoint_dir: Directory holding ``inpaint.json`` and
                ``inpaint.safetensors``. ``None`` falls back to
                :data:`INPAINT_DIR_ENV` and then :data:`DEFAULT_INPAINT_DIR`
                -- resolved at :meth:`load` time, so setting the environment
                variable after constructing the detector still works.
            device: ``"auto"`` (CUDA when visible), ``"cpu"``, or an explicit
                device string.
        """
        self._configured_dir = checkpoint_dir
        self._configured_device = device
        self.checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir)
        self.device = "cpu"
        self.meta: InpaintCheckpointMeta | None = None
        self._backbone: Any = None
        self._head: PatchHead | None = None
        self._normalization: tuple[tuple[float, ...], tuple[float, ...]] | None = None
        self._digest: str | None = None
        self._attempted = False
        self._reason: str | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether a trained checkpoint was found and loaded."""
        return self._head is not None and self.meta is not None

    def load(self, device: str = "auto") -> None:
        """Read the checkpoint, if there is one, and build the model on ``device``.

        Never raises for a missing checkpoint -- the localizer stays unloaded
        and :meth:`predict` abstains with a reason. ``device`` defaults to
        ``"auto"`` rather than the base class's ``"cpu"``: callers that run
        this localizer at all call ``load()`` with no argument, and a frozen
        ViT-B belongs on the GPU when there is one.
        """
        self._attempted = True
        self.checkpoint_dir = resolve_checkpoint_dir(self._configured_dir)
        weights_path = self.checkpoint_dir / WEIGHTS_FILENAME
        metadata_path = self.checkpoint_dir / METADATA_FILENAME

        missing = [path for path in (weights_path, metadata_path) if not path.is_file()]
        if missing:
            self._reason = (
                f"no trained inpainting localizer in {self.checkpoint_dir} "
                f"(missing {', '.join(path.name for path in missing)}); train one with: "
                f"{_TRAIN_HINT}"
            )
            return

        from imgforensics.data.acquire import _sha256_of_file
        from imgforensics.localization._inpaint_model import load_checkpoint

        requested = device if device != "auto" else self._configured_device
        meta = InpaintCheckpointMeta.model_validate_json(metadata_path.read_text(encoding="utf-8"))
        backbone, head, resolved_device, normalization = load_checkpoint(
            meta, weights_path, requested
        )

        self.meta = meta
        self._backbone = backbone
        self._head = head
        self.device = resolved_device
        self._normalization = normalization
        self._digest = _sha256_of_file(weights_path)
        self._reason = None

    def _abstain(self) -> DetectionResult:
        """The result returned when no checkpoint is available."""
        reason = self._reason or (
            f"no trained inpainting localizer in {self.checkpoint_dir}; train one with: "
            f"{_TRAIN_HINT}"
        )
        return DetectionResult(
            detector=self.name,
            score=_ABSTAIN_SCORE,
            label="uncertain",
            details={"reason": reason},
        )

    def predict(self, image: ForensicImage) -> DetectionResult:
        """Localize inpainted regions, or abstain when no checkpoint is installed."""
        if not self._attempted:
            self.load()
        if self._head is None or self.meta is None or self._normalization is None:
            return self._abstain()

        from imgforensics.localization._inpaint_model import infer_heatmap

        pixels = np.asarray(image.rgb.convert("RGB") if image.rgb.mode != "RGB" else image.rgb)
        mean, std = self._normalization
        heatmap, tiles = infer_heatmap(
            self._backbone,
            self._head,
            pixels,
            self.meta.config(),
            device=self.device,
            mean=mean,
            std=std,
        )
        score = top_fraction_score(heatmap, TOP_FRACTION)

        weights_label = WEIGHTS_FILENAME
        if self._digest:
            weights_label = f"{weights_label} ({self._digest[:_SHA_PREFIX_LENGTH]})"

        details: dict[str, Any] = {
            "weights": weights_label,
            "checkpoint": f"{self.checkpoint_dir} ({self.meta.created})",
            "backbone": self.meta.backbone,
            "layers": list(self.meta.layers),
            "tiles": tiles,
            "crop_size": self.meta.crop_size,
            "stride": self.meta.stride,
            "max_prob": round(float(heatmap.max()), 4),
            "mean_prob": round(float(heatmap.mean()), 4),
            f"area_fraction_above_{MASK_THRESHOLD}": round(
                float((heatmap > MASK_THRESHOLD).mean()), 4
            ),
            "lora_rank": None if self.meta.lora is None else self.meta.lora.rank,
            "device": self.device,
        }
        return DetectionResult(
            detector=self.name,
            score=score,
            label=label_from_score(score),
            heatmap=heatmap,
            details=details,
        )
