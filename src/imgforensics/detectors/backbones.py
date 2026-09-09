"""Frozen ViT backbones used as a fixed feature space for the learned detectors.

The recipe these support (``docs/ROADMAP.md``, section 3 and
``docs/research/01_ai_generated_image_detection.md``, section 7a) is a frozen
encoder plus a small trainable head reading *several* transformer blocks, not
only the last one: a self-supervised DINOv2 space separates real from
generated images better than CLIP's language-aligned space, and the
discriminative signal is spread across depth rather than concentrated at the
output. Nothing here trains: weights are loaded once, put in eval mode with
gradients disabled, and only ever read.

Two backbones are registered:

- ``dinov2_vitb14`` -- DINOv2 ViT-B/14 (the default). At 224 px its patch-14
  grid is 16x16 = 256 patch tokens plus one CLS token; 12 blocks, width 768.
- ``clip_vitl14`` -- OpenAI CLIP ViT-L/14, the comparison space; 24 blocks,
  width 1024.

Weights are downloaded from the Hugging Face Hub on first use and cached
there, never committed to this repository (``docs/ROADMAP.md``, section 7).
Set ``IMGFORENSICS_WEIGHTS_DIR`` to keep that cache inside the project's
gitignored ``weights/`` directory instead of the user-wide default.

``torch`` and ``timm`` come from the optional ``ml`` extra and are imported
inside the functions that need them, so this module stays importable without
them (see :func:`imgforensics.detectors.is_ml_available`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - import-time typing only, never at runtime
    import torch

#: Environment variable pointing at a directory to cache downloaded weights in.
WEIGHTS_DIR_ENV = "IMGFORENSICS_WEIGHTS_DIR"

# Last-resort normalization statistics, used only if a model carries no timm
# pretrained config at all; every backbone in BACKBONES ships its own.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class BackboneSpec(BaseModel):
    """A frozen backbone: which timm model, how to feed it, and what to read out of it.

    Attributes:
        name: Registry key in :data:`BACKBONES`.
        timm_id: timm model identifier, including its pretrained tag.
        input_size: Edge length of the square crops this backbone is fed.
            Also the value a :class:`~imgforensics.detectors.crops.CropPolicy`
            must use.
        mean: Channel normalization means, or ``None`` to take them from the
            loaded model's timm ``pretrained_cfg`` (the default, so the
            numbers can never drift out of sync with the weights).
        std: Channel normalization standard deviations, same convention.
        layers: Indices of the transformer blocks whose CLS token is
            extracted, in the order they appear in the feature array.
            Negative indices count from the end. Defaults to the last four
            blocks of the respective model.
        license: License of the *weights* (not of timm itself).
        weights_source: Where the weights come from, for the notices file.
    """

    name: str
    timm_id: str
    input_size: int = 224
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None
    layers: list[int] = Field(default_factory=lambda: [-4, -3, -2, -1])
    license: str = "unknown"
    weights_source: str = ""


#: The registered frozen backbones, keyed by the name used on the CLI.
BACKBONES: dict[str, BackboneSpec] = {
    "dinov2_vitb14": BackboneSpec(
        name="dinov2_vitb14",
        timm_id="vit_base_patch14_dinov2.lvd142m",
        input_size=224,
        layers=[8, 9, 10, 11],
        license="Apache-2.0",
        weights_source="facebookresearch/dinov2 via the Hugging Face Hub (timm)",
    ),
    "clip_vitl14": BackboneSpec(
        name="clip_vitl14",
        timm_id="vit_large_patch14_clip_224.openai",
        input_size=224,
        layers=[20, 21, 22, 23],
        license="MIT",
        weights_source="openai/CLIP via the Hugging Face Hub (timm)",
    ),
}


def get_backbone(name: str) -> BackboneSpec:
    """Return the :class:`BackboneSpec` registered under ``name``.

    Raises:
        KeyError: if no backbone is registered under ``name``, listing the
            known names in the error message.
    """
    try:
        return BACKBONES[name]
    except KeyError as exc:
        known = ", ".join(sorted(BACKBONES))
        raise KeyError(f"No backbone registered as {name!r}. Available: {known}") from exc


def resolve_device(device: str = "auto") -> str:
    """Turn ``"auto"`` into ``"cuda"`` when a CUDA device is visible, else ``"cpu"``.

    Any other value is passed through untouched, so an explicit ``"cuda:1"``
    or ``"cpu"`` still works.
    """
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _apply_weights_dir_env() -> Path | None:
    """Point the Hub/timm caches at ``IMGFORENSICS_WEIGHTS_DIR`` when it is set.

    ``huggingface_hub`` reads its cache location at import time, so both
    ``HF_HOME`` and ``HF_HUB_CACHE`` are set before the first ``timm`` import
    (and re-set on every load, which is harmless and makes the function
    order-independent). Returns the directory, or ``None`` when the variable
    is unset and the user-wide default cache is used.
    """
    raw = os.environ.get(WEIGHTS_DIR_ENV)
    if not raw:
        return None
    directory = Path(raw).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(directory)
    os.environ["HF_HUB_CACHE"] = str(directory / "hub")
    return directory


def load_backbone(spec: BackboneSpec, device: str = "auto") -> torch.nn.Module:
    """Load ``spec``'s pretrained weights as a frozen, eval-mode model on ``device``.

    The classifier head is dropped (``num_classes=0``), every parameter has
    ``requires_grad`` cleared, and the model is put in eval mode, so the
    returned module is inference-only: it cannot be trained by accident and
    it allocates no gradient buffers. Weights are downloaded on first use
    (see the module docstring for where they are cached).

    The model is built at ``spec.input_size`` rather than at its own default
    resolution, and timm resamples the pretrained position embeddings to the
    resulting token grid. This matters for DINOv2, whose timm entry defaults
    to 518 px and asserts on any other input size: at 224 px its patch-14
    grid is 16x16, which is the resolution the crop policy feeds it.

    Mixed precision is *not* baked into the weights here -- the model stays
    float32 and :func:`extract` runs it under ``torch.autocast`` on CUDA, so
    the same loaded model works on both CPU and GPU.
    """
    _apply_weights_dir_env()

    import timm
    import torch

    model = timm.create_model(
        spec.timm_id, pretrained=True, num_classes=0, img_size=spec.input_size
    )
    model.eval()
    model.requires_grad_(False)
    return model.to(torch.device(resolve_device(device)))


def normalization_for(
    model: torch.nn.Module, spec: BackboneSpec
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The ``(mean, std)`` a loaded backbone expects its input to be normalized with.

    ``spec.mean``/``spec.std`` win when set; otherwise the values come from
    the model's timm ``pretrained_cfg``, which ships with the weights (so
    DINOv2's ImageNet statistics and CLIP's own statistics are each used with
    their own model without being hardcoded here). Falls back to the ImageNet
    defaults if a model carries no config at all.
    """
    config: dict[str, Any] = dict(getattr(model, "pretrained_cfg", None) or {})
    mean = spec.mean if spec.mean is not None else config.get("mean", _IMAGENET_MEAN)
    std = spec.std if spec.std is not None else config.get("std", _IMAGENET_STD)
    return tuple(float(value) for value in mean), tuple(float(value) for value in std)


def _cls_token(intermediate: Any) -> torch.Tensor:
    """Pull the CLS token out of one entry of timm's ``forward_intermediates`` output.

    With ``return_prefix_tokens=True`` each entry is a
    ``(patch_tokens, prefix_tokens)`` pair, and the CLS token is the first
    prefix token: ``prefix`` is ``(B, num_prefix_tokens, D)``, whose leading
    token is the class token for every ViT in :data:`BACKBONES` (neither
    carries register tokens ahead of it). Older timm builds hand back a
    single ``(B, 1 + N, D)`` tensor instead, whose first token is the same
    CLS token -- both shapes are accepted here so the code does not pin one
    timm patch release.
    """
    prefix = intermediate[1] if isinstance(intermediate, tuple | list) else intermediate
    return prefix[:, 0]


def extract(model: torch.nn.Module, spec: BackboneSpec, batch: torch.Tensor) -> torch.Tensor:
    """Multi-layer CLS features for a normalized ``(B, 3, H, W)`` batch of crops.

    Runs one forward pass through the frozen backbone with timm's
    ``forward_intermediates(x, indices=spec.layers, return_prefix_tokens=True,
    norm=True, output_fmt="NLC", intermediates_only=False)`` and keeps, per
    selected block, that block's CLS token.

    Returns:
        A float32 tensor of shape ``(B, len(spec.layers) + 1, D)`` on the
        model's device. Row ``i`` of the middle axis is the CLS token of
        ``spec.layers[i]``, in the order they are listed in the spec; the
        final row is the backbone's own pooled output (its ``forward_head``
        pre-logits vector, i.e. what the model would normally hand a
        classifier). The pooled row comes free from the same forward pass --
        ``intermediates_only=False`` also returns the final normalized token
        sequence -- so it costs one pooling operation, not a second pass.

    On CUDA the forward pass runs under float16 autocast (roughly halving
    activation memory and time on this project's 6 GB GPU); the result is
    cast back to float32 so callers never see a half-precision array.
    """
    import torch

    device = next(model.parameters()).device
    inputs = batch.to(device, non_blocking=True)
    use_autocast = device.type == "cuda"
    # forward_intermediates/forward_head are timm's ViT API, not torch.nn.Module's,
    # so they are called through an untyped alias rather than the Module type.
    net: Any = model

    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast),
    ):
        final, intermediates = net.forward_intermediates(
            inputs,
            indices=list(spec.layers),
            return_prefix_tokens=True,
            norm=True,
            output_fmt="NLC",
            intermediates_only=False,
        )
        rows = [_cls_token(item) for item in intermediates]
        rows.append(net.forward_head(final, pre_logits=True))
        stacked = torch.stack(rows, dim=1)

    return stacked.float()
