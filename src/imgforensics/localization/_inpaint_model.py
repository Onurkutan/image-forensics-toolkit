"""The torch half of ``dino_inpaint``: the patch head, its LoRA option, and inference.

Split from :mod:`imgforensics.localization.dino_inpaint` for the reason every
localizer in this package is split that way -- the wrapper is imported on
every CLI invocation for its registration side effect, and importing ``torch``
there would cost seconds on commands that never run a model. Everything here
declares or drives an :class:`torch.nn.Module`, so this module needs the
optional ``ml`` extra and is imported only from inside functions that do.

Three pieces:

- :class:`PatchHead` -- the trained part. Per-source-block LayerNorm, a 1x1
  projection of the concatenated blocks, one optional 3x3 convolution, a 1x1
  classifier; about 1.19 M parameters, fully convolutional, so the same
  weights run on any patch grid the backbone is fed.
- :class:`LoRALinear` / :func:`attach_lora` -- stage 2's way of letting the
  backbone move without training (or saving) 86 M parameters. A rank-8 update
  on each block's fused ``attn.qkv`` is 24,576 numbers per block; the base
  weight is never touched, never saved, and stays frozen.
- :func:`load_patch_backbone`, :func:`patch_logits`, :func:`infer_heatmap` --
  the forward path, shared by training's validation pass and by the
  localizer's ``predict`` so that the two cannot disagree about what the model
  does to an image.

**Why not** :func:`imgforensics.detectors.backbones.extract`: it runs under
``torch.inference_mode`` and reads CLS tokens. This model needs the patch
tokens as a feature map (``output_fmt="NCHW"``) and, in stage 2, gradients
flowing back into the backbone, which an inference-mode tensor forbids
forever. The backbone is still loaded through
:func:`~imgforensics.detectors.backbones.load_backbone`, with an overridden
:class:`~imgforensics.detectors.backbones.BackboneSpec` rather than a new
registry entry: ``input_size`` is part of the feature cache key and of every
shipped ``head.json``, so changing the registered ``dinov2_vitb14`` to 448 px
would silently invalidate both.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from imgforensics.detectors.backbones import (
    BackboneSpec,
    get_backbone,
    load_backbone,
    normalization_for,
    resolve_device,
)
from imgforensics.localization.dino_inpaint import (
    InpaintCheckpointMeta,
    InpaintConfig,
    LoraConfig,
    PatchHeadConfig,
    pad_for_tiles,
    stitch_tiles,
    tile_origins,
)

#: Tiles pushed through the backbone at once during inference. Four 448 px
#: crops is about 0.4 GB of activations under float16 autocast, which leaves
#: the 6 GB card room for whatever else is running.
INFERENCE_BATCH = 4

#: Attention submodule LoRA is attached to: the fused query/key/value
#: projection, ``Linear(768, 2304)`` in a ViT-B. Attaching to the fused layer
#: adapts all three projections with one pair of factors.
LORA_TARGET = "attn.qkv"

#: Prefixes the checkpoint's tensors are stored under, so the head's weights
#: and the backbone's deltas can be told apart without a second file.
HEAD_PREFIX = "head."
LORA_PREFIX = "lora."

_MAX_PIXEL = 255.0


class PatchHead(nn.Module):
    """Fully convolutional classifier over a concatenated multi-block patch grid.

    Input is ``(B, n_layers * dim, G, G)`` -- the selected blocks' patch-token
    feature maps concatenated along channels, as
    :func:`patch_features` produces them -- and the output is ``(B, 1, G, G)``
    raw logits, positive meaning "inpainted".

    Each source block is normalized on its own before the projection sees it,
    for the same reason :class:`imgforensics.detectors.head.MultiLayerHead`
    normalizes each depth separately: block 5 and block 11 of a ViT have
    different activation scales, and one shared normalization would let the
    loudest of them dominate the projection. Normalization is over the channel
    axis of each patch (a :class:`torch.nn.LayerNorm` over ``dim``), which is
    what "LayerNorm" means for a token and not what it would mean applied to
    an NCHW map's spatial axes.

    Size, at the default shape (three blocks of 768, 256 wide): 1,185,025
    parameters, of which the two 256-channel convolutions are 1.18 M. With
    ``context_kernel = 1`` the 3x3 is dropped entirely and the head is a
    per-patch MLP, which is the ablation that says whether neighbourhood
    context is doing any work.
    """

    def __init__(self, config: PatchHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.norms = nn.ModuleList([nn.LayerNorm(config.dim) for _ in range(config.n_layers)])
        self.project = nn.Conv2d(config.n_layers * config.dim, config.proj_dim, kernel_size=1)
        self.context = (
            nn.Conv2d(
                config.proj_dim,
                config.proj_dim,
                kernel_size=config.context_kernel,
                padding=config.context_kernel // 2,
            )
            if config.context_kernel > 1
            else None
        )
        self.classify = nn.Conv2d(config.proj_dim, 1, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Logits for a ``(B, n_layers * dim, G, G)`` feature map: ``(B, 1, G, G)``.

        Raises:
            ValueError: ``features`` is not 4-D, or its channel count does not
                match the configured ``n_layers * dim``.
        """
        expected = self.config.n_layers * self.config.dim
        if features.ndim != 4 or features.shape[1] != expected:
            raise ValueError(
                f"expected features of shape (B, {expected}, G, G), got {tuple(features.shape)}"
            )

        normalized = [
            norm(chunk.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            for norm, chunk in zip(
                self.norms, features.chunk(self.config.n_layers, dim=1), strict=True
            )
        ]
        hidden = self.activation(self.project(torch.cat(normalized, dim=1)))
        if self.context is not None:
            hidden = self.activation(self.context(hidden))
        return self.classify(hidden)


class LoRALinear(nn.Module):
    """A frozen :class:`torch.nn.Linear` plus a trainable rank-``rank`` update.

    ``y = base(x) + (dropout(x) @ A.T) @ B.T * (alpha / rank)``, the update
    from "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al.,
    2021). ``B`` is initialised to zeros, so a freshly attached adapter is
    *exactly* the layer it wraps -- stage 2 therefore starts from stage 1's
    model rather than from a perturbed one -- and ``A`` gets the same
    Kaiming-uniform initialisation a Linear's weight would.

    The base layer keeps its own parameters and they stay frozen; only ``A``
    and ``B`` are ever saved, which is what keeps a stage-2 checkpoint a few
    hundred kilobytes instead of a third of a gigabyte.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The base layer's output plus this adapter's low-rank update."""
        update = nn.functional.linear(self.dropout(x), self.lora_a)
        return self.base(x) + nn.functional.linear(update, self.lora_b) * self.scaling


def attach_lora(model: nn.Module, config: LoraConfig, target: str = LORA_TARGET) -> list[str]:
    """Wrap every ``target`` submodule of ``model`` in a :class:`LoRALinear`.

    The whole model is frozen first, so the adapters' factors are the only
    trainable tensors left -- a caller cannot accidentally hand the optimizer
    a backbone parameter by forgetting a ``requires_grad_`` somewhere.

    Args:
        model: The backbone to patch, in place.
        config: Rank, alpha and dropout of the adapters.
        target: Dotted attribute path within each block, relative to the
            module that owns it (``"attn.qkv"`` for a timm ViT).

    Returns:
        The patched modules' dotted names, in model order, for the record in
        the checkpoint.

    Raises:
        ValueError: no submodule matched ``target``, which would silently
            train nothing at all.
    """
    model.requires_grad_(False)
    suffix = f".{target}"
    patched: list[str] = []
    for name, module in list(model.named_modules()):
        matched = name == target or name.endswith(suffix)
        if not matched or not isinstance(module, nn.Linear):
            continue
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, attribute, LoRALinear(module, config.rank, config.alpha, config.dropout))
        patched.append(name)

    if not patched:
        raise ValueError(f"no {target!r} submodule found to attach LoRA to")
    return patched


def set_lora_training(model: nn.Module, training: bool) -> int:
    """Put the attached adapters in train (or eval) mode, and nothing else.

    The backbone is held in eval mode for a whole training run so that its
    normalization statistics cannot drift -- but ``eval()`` reaches every
    submodule, including the adapters' dropout, which would leave the rate
    recorded in ``inpaint.json`` describing something that never fired. This
    flips exactly the :class:`LoRALinear` modules back and leaves every other
    submodule as it was found.

    Returns:
        How many adapters were switched, so a caller can assert it patched
        the model it thought it had.
    """
    switched = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.train(training)
            switched += 1
    return switched


def backbone_spec(config: InpaintConfig) -> BackboneSpec:
    """The registered backbone, overridden to this model's resolution and blocks.

    A copy, never a mutation of :data:`imgforensics.detectors.backbones.BACKBONES`:
    that registry's ``input_size`` is baked into the feature cache key and into
    every trained ``head.json``, and this model runs the same weights at a
    different resolution for a different purpose.
    """
    return get_backbone(config.backbone).model_copy(
        update={"input_size": config.crop_size, "layers": list(config.layers)}
    )


def load_patch_backbone(
    config: InpaintConfig, device: str = "auto"
) -> tuple[nn.Module, str, tuple[float, ...], tuple[float, ...]]:
    """Load the frozen backbone this head reads, ready for variable-size input.

    ``dynamic_img_size=True`` lets one loaded model accept 448 px training
    crops and a 266 px CocoGlide tile without rebuilding: timm interpolates
    the position embeddings per forward pass instead of asserting on the
    single size the model was built at.

    Returns:
        ``(model, resolved device, normalization mean, normalization std)``.
    """
    spec = backbone_spec(config)
    resolved = resolve_device(device)
    model = load_backbone(spec, resolved, dynamic_img_size=True)
    mean, std = normalization_for(model, spec)
    return model, resolved, mean, std


def patch_features(backbone: nn.Module, batch: torch.Tensor, layers: Sequence[int]) -> torch.Tensor:
    """The selected blocks' patch tokens for a normalized batch, concatenated.

    One forward pass through timm's ``forward_intermediates(x, indices=layers,
    norm=True, output_fmt="NCHW", intermediates_only=True)``, which returns one
    ``(B, dim, G, G)`` map per requested block. Called directly rather than
    through :func:`imgforensics.detectors.backbones.extract` because that one
    runs under ``torch.inference_mode`` -- see this module's docstring.

    Returns:
        ``(B, len(layers) * dim, G, G)``, the maps concatenated in the order
        the blocks are listed.
    """
    # forward_intermediates is timm's ViT API, not torch.nn.Module's, so it is
    # called through an untyped alias rather than the Module type.
    net: Any = backbone
    maps = net.forward_intermediates(
        batch,
        indices=list(layers),
        norm=True,
        output_fmt="NCHW",
        intermediates_only=True,
    )
    return torch.cat(list(maps), dim=1)


def patch_logits(
    backbone: nn.Module, head: PatchHead, batch: torch.Tensor, layers: Sequence[int]
) -> torch.Tensor:
    """Per-patch logits for a normalized ``(B, 3, H, W)`` batch: ``(B, 1, G, G)``.

    Gradient-transparent on purpose: stage 1 wraps the backbone half of this
    in ``torch.no_grad()`` while stage 2 does not, and inference wraps the
    whole thing, so the decision belongs to the caller.
    """
    return head(patch_features(backbone, batch, layers))


def to_batch(
    windows: Sequence[np.ndarray],
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Stack HxWx3 uint8 crops into a normalized ``(B, 3, H, W)`` float32 batch."""
    stacked = np.stack([np.asarray(window, dtype=np.float32) for window in windows])
    normalized = (stacked / _MAX_PIXEL - np.asarray(mean, dtype=np.float32)) / np.asarray(
        std, dtype=np.float32
    )
    return torch.from_numpy(normalized.transpose(0, 3, 1, 2)).contiguous().to(device)


def infer_heatmap(
    backbone: nn.Module,
    head: PatchHead,
    pixels: np.ndarray,
    config: InpaintConfig,
    *,
    device: str = "cpu",
    mean: Sequence[float],
    std: Sequence[float],
) -> tuple[np.ndarray, int]:
    """The inpainting probability of every pixel of ``pixels``, and the tile count.

    Pads the image up to whole patches when it is smaller than a tile, runs
    every tile through the backbone and head in batches of
    :data:`INFERENCE_BATCH`, turns each tile's patch logits into probabilities
    and upsamples them back to the tile's own pixels, then averages the
    overlaps and crops the padding off. The image is never resized.

    Returns:
        ``(heatmap, tiles)`` -- a float32 array shaped exactly like ``pixels``
        with values in ``[0, 1]``, and how many tiles produced it.
    """
    was_training = head.training
    head.eval()
    backbone.eval()

    padded, (pad_top, pad_left) = pad_for_tiles(pixels, config.crop_size, config.patch)
    height, width = padded.shape[:2]
    tile_height = min(config.crop_size, height)
    tile_width = min(config.crop_size, width)

    boxes = [
        (top, left)
        for top in tile_origins(height, tile_height, config.stride)
        for left in tile_origins(width, tile_width, config.stride)
    ]

    torch_device = torch.device(device)
    maps: list[tuple[int, int, np.ndarray]] = []
    with torch.no_grad():
        for start in range(0, len(boxes), INFERENCE_BATCH):
            chunk = boxes[start : start + INFERENCE_BATCH]
            batch = to_batch(
                [padded[top : top + tile_height, left : left + tile_width] for top, left in chunk],
                mean,
                std,
                torch_device,
            )
            with torch.autocast(
                device_type=torch_device.type,
                dtype=torch.float16,
                enabled=torch_device.type == "cuda",
            ):
                logits = patch_logits(backbone, head, batch, config.layers)
            probabilities = nn.functional.interpolate(
                torch.sigmoid(logits.float()),
                size=(tile_height, tile_width),
                mode="bilinear",
                align_corners=False,
            )
            values = probabilities[:, 0].cpu().numpy()
            maps.extend((top, left, values[index]) for index, (top, left) in enumerate(chunk))

    head.train(was_training)
    heatmap = stitch_tiles(
        maps,
        (height, width),
        crop=(pad_top, pad_left, pixels.shape[0], pixels.shape[1]),
    )
    return heatmap, len(maps)


def checkpoint_tensors(head: PatchHead, backbone: nn.Module | None = None) -> dict[str, Any]:
    """The tensors a checkpoint file holds: the head, and any LoRA factors.

    Never a base weight. The backbone is 86 M frozen parameters that
    :func:`load_patch_backbone` downloads from the Hub anyway, so writing them
    into every checkpoint would turn a 5 MB file into a 350 MB one and record
    nothing that was learned.
    """
    tensors: dict[str, Any] = {
        f"{HEAD_PREFIX}{name}": value.detach().cpu().contiguous()
        for name, value in head.state_dict().items()
    }
    if backbone is not None:
        for name, module in backbone.named_modules():
            if isinstance(module, LoRALinear):
                tensors[f"{LORA_PREFIX}{name}.lora_a"] = module.lora_a.detach().cpu().contiguous()
                tensors[f"{LORA_PREFIX}{name}.lora_b"] = module.lora_b.detach().cpu().contiguous()
    return tensors


def load_checkpoint(
    meta: InpaintCheckpointMeta, weights_path: Path, device: str = "auto"
) -> tuple[nn.Module, PatchHead, str, tuple[tuple[float, ...], tuple[float, ...]]]:
    """Rebuild the model ``meta`` describes and fill it from ``weights_path``.

    Everything about the shape comes from the checkpoint -- backbone, blocks,
    crop size, head width, LoRA rank -- so a file trained with a different
    ablation loads as itself rather than as today's defaults.

    Returns:
        ``(backbone, head, resolved device, (mean, std))``.

    Raises:
        KeyError: the file is missing a tensor the rebuilt model needs (a
            checkpoint whose ``lora`` block disagrees with its tensors).
    """
    from safetensors.torch import load_file

    config = meta.config()
    backbone, resolved, mean, std = load_patch_backbone(config, device)
    tensors = load_file(str(weights_path))

    if meta.lora is not None:
        attach_lora(backbone, meta.lora)
        # attach_lora builds lora_a/lora_b on the CPU, wherever the layer it
        # wraps happens to live, and load_patch_backbone has already moved the
        # backbone; without this the first forward on CUDA dies with "Expected
        # all tensors to be on the same device". train_inpaint does the same
        # thing after attaching.
        backbone.to(torch.device(resolved))
        for name, module in backbone.named_modules():
            if not isinstance(module, LoRALinear):
                continue
            with torch.no_grad():
                module.lora_a.copy_(tensors[f"{LORA_PREFIX}{name}.lora_a"])
                module.lora_b.copy_(tensors[f"{LORA_PREFIX}{name}.lora_b"])
        backbone.requires_grad_(False)

    head = PatchHead(config.head_config())
    head.load_state_dict(
        {
            name[len(HEAD_PREFIX) :]: value
            for name, value in tensors.items()
            if name.startswith(HEAD_PREFIX)
        },
        strict=True,
    )
    head.eval()
    head.requires_grad_(False)
    head.to(torch.device(resolved))
    backbone.eval()
    return backbone, head, resolved, (mean, std)
