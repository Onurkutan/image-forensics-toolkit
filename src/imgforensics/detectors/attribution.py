"""Grad-CAM attribution for the learned detector: where inside a crop it looked.

The head's own heatmap
(:meth:`imgforensics.detectors.learned.LearnedDetector._heatmap`) paints one
calibrated probability over each crop, so it says *how generated* a region
looks but nothing about *why*. At 224 px a crop is a quarter of a megapixel of
evidence reduced to a single number, and on a large photograph a whole crop is
still a large thing to point at. This module recovers the missing resolution:
one saliency map per crop, on the backbone's own patch grid, saying which
patches moved the calibrated logit.

The method is Grad-CAM (Selvaraju et al., 2017) adapted to a ViT whose
classifier reads the CLS token of *several* blocks rather than one
convolutional feature map:

- The "feature map" of a block is its patch-token output reshaped back to the
  ``g x g`` patch grid, and the "channels" are the transformer width.
- The gradient is taken of the *calibrated* logit -- the same quantity the
  reported score is a sigmoid of -- so the map explains the number a user
  actually sees rather than an intermediate one.
- Because the head reads several depths, there is one CAM per depth, and they
  are combined with the head's own learned layer weights
  (:meth:`imgforensics.detectors.head.MultiLayerHead.layer_weights`). A depth
  the head barely uses barely contributes to the picture of where it looked.

**Which block a row's CAM comes from.** The head reads the CLS token at the
*output* of block ``l``. That token cannot depend on the patch tokens at the
output of the same block -- attention mixes the tokens entering a block, not
the ones leaving it -- so its gradient with respect to those patch tokens is
exactly zero. The informative activations are the ones that *feed* block
``l``, i.e. the output of block ``l - 1``; :func:`target_blocks` does that
shift. The pooled row is read after the final block, so it is fed by block
``len(blocks) - 2``.

``torch`` is imported inside :func:`grad_cam`, so :func:`target_blocks` and
:func:`stitch_attribution` -- both pure numpy -- stay importable and testable
without the optional ``ml`` extra.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover - import-time typing only, never at runtime
    import torch

    from imgforensics.detectors.backbones import BackboneSpec
    from imgforensics.detectors.crops import CropBox
    from imgforensics.detectors.head import Calibration, MultiLayerHead


def target_blocks(
    spec_layers: Sequence[int], n_blocks: int, layer_weights: Sequence[float]
) -> dict[int, float]:
    """Map each head input row to the block whose patch tokens feed it, with its weight.

    Args:
        spec_layers: The backbone blocks the head reads a CLS token from, in
            feature-row order, as
            :attr:`~imgforensics.detectors.backbones.BackboneSpec.layers`
            lists them. Negative indices count from the end.
        n_blocks: Number of transformer blocks in the loaded model.
        layer_weights: One weight per head input row -- ``len(spec_layers) + 1``
            of them, the last being the pooled row (see
            :func:`imgforensics.detectors.backbones.extract`).

    Returns:
        ``{block index: weight}`` over the blocks whose patch tokens the CAM
        is taken from, with the weights renormalized to sum to 1. Two rows can
        resolve to the same block -- the last selected layer and the pooled
        row usually do -- in which case their weights are summed, so a block
        the head reads twice counts twice.

    A row whose feeding block would be ``-1`` (a head reading block 0, whose
    input is the patch embedding rather than another block's output) has no
    block to hook and is dropped, with the remaining weights renormalized. No
    backbone in :data:`~imgforensics.detectors.backbones.BACKBONES` selects
    block 0, so this guards a future spec rather than a live path.

    Raises:
        ValueError: ``layer_weights`` is not ``len(spec_layers) + 1`` long,
            ``n_blocks`` is not positive, or a selected layer index is out of
            range for ``n_blocks``.
    """
    expected = len(spec_layers) + 1
    if len(layer_weights) != expected:
        raise ValueError(f"expected {expected} layer weights, got {len(layer_weights)}")
    if n_blocks < 1:
        raise ValueError(f"the model reports {n_blocks} transformer blocks")

    feeding: list[int] = []
    for layer in spec_layers:
        resolved = layer + n_blocks if layer < 0 else layer
        if not 0 <= resolved < n_blocks:
            raise ValueError(f"layer index {layer} is out of range for {n_blocks} blocks")
        feeding.append(resolved - 1)
    # The pooled row is read after the last block, so the tokens feeding it
    # leave the one before it.
    feeding.append(n_blocks - 2)

    weighted: dict[int, float] = {}
    for block, weight in zip(feeding, layer_weights, strict=True):
        if block < 0:
            continue
        weighted[block] = weighted.get(block, 0.0) + float(weight)

    if not weighted:
        return {}
    total = sum(weighted.values())
    if total <= 0.0:
        # Degenerate weights (a head whose importance collapsed to zero) still
        # deserve a picture: fall back to an even split over the same blocks.
        return dict.fromkeys(weighted, 1.0 / len(weighted))
    return {block: weight / total for block, weight in weighted.items()}


def _patch_grid(n_tokens: int, n_prefix: int) -> int:
    """Edge length of the square patch grid behind ``n_tokens`` tokens.

    Derived from the token count rather than assumed, so the same code works
    for DINOv2's 16x16 patch-14 grid at 224 px and for any other backbone and
    input size.

    Raises:
        ValueError: the patch tokens do not form a square grid.
    """
    patches = n_tokens - n_prefix
    edge = int(round(patches**0.5))
    if patches <= 0 or edge * edge != patches:
        raise ValueError(f"{patches} patch tokens do not form a square grid")
    return edge


def _normalize_per_crop(cam: torch.Tensor) -> torch.Tensor:
    """Divide each crop's map by its own maximum, leaving an all-zero map at zero."""
    import torch

    peak = cam.amax(dim=(1, 2), keepdim=True)
    return cam / torch.where(peak > 0, peak, torch.ones_like(peak))


def grad_cam(
    model: torch.nn.Module,
    spec: BackboneSpec,
    head: MultiLayerHead,
    batch: torch.Tensor,
    calibration: Calibration,
) -> np.ndarray:
    """Per-crop Grad-CAM maps on the backbone's patch grid, for one batch of crops.

    Args:
        model: The frozen backbone, as
            :func:`~imgforensics.detectors.backbones.load_backbone` returns
            it. Its parameters carry no gradient, so the graph is kept alive
            by making the *input* require one instead.
        spec: The backbone's spec; its ``layers`` decide which blocks are
            explained.
        head: The trained head, in eval mode -- dropout would otherwise make
            two runs over the same crops disagree.
        batch: A normalized ``(N, 3, H, W)`` crop batch, exactly what
            :func:`~imgforensics.detectors.backbones.extract` is fed.
        calibration: The checkpoint's calibration, applied to the logits
            before the backward pass so the map explains the reported
            probability rather than the raw head output.

    Returns:
        ``(N, g, g)`` float32 maps in [0, 1], each normalized to a maximum of
        1 by itself (a crop whose CAM is everywhere non-positive stays all
        zero). ``g`` is the patch grid edge -- 16 for DINOv2 ViT-B/14 at
        224 px.

    Raises:
        ValueError: the model exposes no ``blocks``, its tokens do not form a
            square patch grid, or no block is left to explain.

    The forward pass repeats :func:`~imgforensics.detectors.backbones.extract`
    argument for argument, including the float16 autocast on CUDA, so the head
    sees exactly the features it was trained and benchmarked on; only the
    ``torch.inference_mode()`` becomes ``torch.enable_grad()``, because an
    inference-mode tensor cannot carry a graph. The CAM arithmetic itself runs
    in float32.

    One backward pass serves the whole batch: a crop's logit depends only on
    that crop's own tokens, so back-propagating the *sum* of the calibrated
    logits gives every crop its own gradient without a per-crop backward.
    """
    import torch

    from imgforensics.detectors.backbones import _cls_token

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("backbone exposes no transformer blocks to hook")

    weights = [float(value) for value in head.layer_weights().detach().cpu()]
    targets = target_blocks(spec.layers, len(blocks), weights)
    if not targets:  # pragma: no cover - only a head reading block 0 alone gets here
        raise ValueError("no block feeds any of the head's inputs")

    device = next(model.parameters()).device
    inputs = batch.to(device, non_blocking=True).float().detach().requires_grad_(True)
    use_autocast = device.type == "cuda"
    # forward_intermediates/forward_head are timm's ViT API, not torch.nn.Module's,
    # so they are called through an untyped alias rather than the Module type.
    net: Any = model

    # Hooking every block costs one dict write per block and spares the index
    # bookkeeping of hooking only the targets; the blocks nothing reads simply
    # never receive a gradient.
    activations: dict[int, torch.Tensor] = {}
    handles = []

    def _capture(index: int) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tokens = output[0] if isinstance(output, tuple | list) else output
            tokens.retain_grad()
            activations[index] = tokens

        return hook

    for index, block in enumerate(blocks):
        handles.append(block.register_forward_hook(_capture(index)))

    try:
        with (
            torch.enable_grad(),
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
            logits = head(torch.stack(rows, dim=1).float())
            calibrated = (logits + calibration.bias) / calibration.temperature
        calibrated.sum().backward()
    finally:
        for handle in handles:
            handle.remove()

    n_prefix = int(getattr(model, "num_prefix_tokens", 1))
    combined: torch.Tensor | None = None
    for index, weight in targets.items():
        tokens = activations.get(index)
        if tokens is None or tokens.grad is None:  # pragma: no cover - every target is hooked
            continue
        grid = _patch_grid(int(tokens.shape[1]), n_prefix)
        patches = tokens[:, n_prefix:, :].detach().float()
        gradients = tokens.grad[:, n_prefix:, :].detach().float()
        channel_weights = gradients.mean(dim=1)  # (N, D), pooled over the patch tokens
        cam = torch.relu((patches * channel_weights[:, None, :]).sum(-1))
        cam = _normalize_per_crop(cam.reshape(cam.shape[0], grid, grid))
        combined = cam * weight if combined is None else combined + cam * weight

    if combined is None:  # pragma: no cover - unreachable while every target is hooked
        raise ValueError("no target block produced a gradient")

    return _normalize_per_crop(combined).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)


def _crop_offset(start: int, extent: int, limit: int, crop_size: int) -> int:
    """Where a clipped box's region begins inside its own ``crop_size`` crop.

    A :class:`~imgforensics.detectors.crops.CropBox` is reported in the
    un-padded image's coordinates, so a crop overhanging the reflection
    padding has a box shorter than the crop on that axis. Which end of the
    crop the box corresponds to follows from where the box sits:

    - Full extent: the crop did not overhang, so the region starts at 0.
    - Clipped but starting past the image's near edge: the overhang is at the
      far edge, and the surviving region is still the crop's leading rows or
      columns -- offset 0.
    - Clipped at the near edge only (``start == 0``, ending before the image
      does): the crop's leading pixels fell in the padding, so the region is
      the crop's *trailing* ``extent``.
    - Clipped at both ends (the image is shorter than the crop on this axis):
      :func:`imgforensics.eval.preprocess._pad_to_at_least` centers the
      padding, so the image sits in the middle of the crop.
    """
    if extent >= crop_size:
        return 0
    if start > 0:
        return 0
    if start + extent < limit:
        return crop_size - extent
    return (crop_size - extent) // 2


def _upsampled(cam: np.ndarray, crop_size: int) -> np.ndarray:
    """Bilinearly resample one ``(g, g)`` map up to its crop's own ``crop_size``."""
    image = Image.fromarray(np.asarray(cam, dtype=np.float32), mode="F")
    resized = image.resize((crop_size, crop_size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def stitch_attribution(
    image_shape: tuple[int, int],
    boxes: Sequence[CropBox],
    per_crop_maps: Sequence[np.ndarray] | np.ndarray,
    crop_size: int,
) -> np.ndarray | None:
    """Paint per-crop patch-grid maps back onto one image-shaped map.

    Each ``(g, g)`` map is bilinearly upsampled to the full ``crop_size``
    square its crop covered, and the sub-window the crop's (possibly clipped)
    box actually reaches is written into the image. The rules mirror
    :meth:`imgforensics.detectors.learned.LearnedDetector._heatmap`: a crop
    that fell entirely in the reflection padding is skipped, pixels no crop
    covered stay 0, and pixels several crops covered -- possible in ``random``
    mode, whose crops may overlap -- get the mean.

    Args:
        image_shape: ``(height, width)`` of the source image.
        boxes: Where each crop was cut from, from
            :func:`~imgforensics.detectors.crops.crop_boxes`.
        per_crop_maps: One map per box, in the same order.
        crop_size: The crop policy's crop edge, in pixels.

    Returns:
        A float32 ``(height, width)`` map in [0, 1], or ``None`` when the
        boxes and the maps disagree in number -- the same guard the heatmap
        uses, so a mismatch degrades to "no map" rather than to a wrong one.
    """
    height, width = image_shape
    maps = list(per_crop_maps)
    if len(boxes) != len(maps):
        return None

    totals = np.zeros((height, width), dtype=np.float64)
    counts = np.zeros_like(totals)
    for box, cam in zip(boxes, maps, strict=True):
        if box.is_empty:
            continue
        full = _upsampled(cam, crop_size)
        row = _crop_offset(box.top, box.height, height, crop_size)
        column = _crop_offset(box.left, box.width, width, crop_size)
        rows = slice(box.top, box.top + box.height)
        columns = slice(box.left, box.left + box.width)
        totals[rows, columns] += full[row : row + box.height, column : column + box.width]
        counts[rows, columns] += 1.0

    covered = counts > 0
    stitched = np.zeros_like(totals)
    stitched[covered] = totals[covered] / counts[covered]
    return np.clip(stitched, 0.0, 1.0).astype(np.float32)
