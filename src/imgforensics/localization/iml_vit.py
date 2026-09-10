"""IML-ViT as a registered pixel-level localizer.

IML-ViT ("IML-ViT: Benchmarking Image Manipulation Localization by Vision
Transformer", arXiv:2307.14863, code at https://github.com/SunnyHaze/IML-ViT,
MIT) is the license-clean pretrained localizer chosen for Phase 4a
(``docs/ROADMAP.md``; ``docs/research/02_manipulation_localization.md``,
section on pretrained localizers). Sources for everything asserted below,
each read on 2026-09-10:

- License: MIT, ``LICENSE`` in the upstream repository (Copyright (c) 2023
  Xiaochen Ma), and the GitHub API's ``license.spdx_id`` for the repository.
- Architecture: ``iml_vit_model.py`` and ``modules/`` at commit
  ``07dd2be0f4ea27a5c97c9fa5ffbe236733833eac`` -- a plain ViT-B/16 encoder
  (12 blocks, width 768, 12 heads) with windowed attention (window 14) in
  every block except 2, 5, 8 and 11, which stay global, plus decomposed
  relative position embeddings; a *simple* feature pyramid (scale factors
  4, 2, 1, 0.5 plus a max-pooled sixth level) over that single-scale output;
  and a SegFormer-style decoder head that fuses the five levels into one
  logit map. Trained with an edge-supervised loss (a second BCE term
  weighted by a dilated mask boundary, ``edge_lambda=20``) -- training-only,
  and not vendored.
- Input handling: 1024x1024, produced by **zero-padding, never resizing**.
  ``utils/iml_transforms.py``'s ``get_albu_transforms("pad")`` pads with
  ``PadIfNeeded(min_height=1024, min_width=1024, border_mode=0, value=0,
  position="top_left")``, i.e. the image sits at the top-left and the padding
  is added at the bottom and right, then normalizes with the ImageNet
  statistics and crops back to 1024. ``Demo.ipynb`` confirms the other end of
  it: the prediction is read back as ``output[0, :, 0:shape[0], 0:shape[1]]``,
  the top-left crop at the original size.
- Checkpoint: ``checkpoints/ckpt_download_page.md`` -> a single Google Drive
  file (id ``1xXJGJPW1i5j9Pc1JKd7fJmIAQkvt9jY7``,
  ``iml-vit_checkpoint.pth``), a bare ``state_dict`` that
  ``Demo.ipynb`` loads with ``strict=True``. See
  :mod:`imgforensics.localization.weights` for the size and digest.
- Output: a 1024x1024 sigmoid probability mask, thresholded at 0.5 in the
  demo notebook.

**Image-level score.** The paper and the released code report pixel-level
metrics only -- ``utils/evaluation.py`` computes a pixel F1 over the
non-padded region and nothing else -- so there is no upstream rule to follow.
This detector therefore reports the **mean of the top 1% of heatmap values**
(:data:`_TOP_FRACTION`). A plain mean would scale with the manipulated
region's size and call every small edit authentic; a plain max is one pixel
of noise away from calling everything fake. The top 1% is the smallest
statistic that still needs a contiguous confident *region* rather than a
spike, and it is reported next to ``max_prob`` and ``mean_prob`` in the
details so a reader can see what it was computed from.

**Crop, never resize -- and never truncate either.** Upstream's transform
crops anything larger than 1024 px down to its top-left 1024x1024 corner,
which silently discards most of a modern photograph. This wrapper keeps the
padding rule for images that fit and, for anything larger, runs overlapping
1024 px tiles at stride :data:`_TILE_STRIDE` and averages the overlaps, so
every pixel is looked at exactly as the model expects to see it and the
returned heatmap covers the whole image. Resizing never happens in either
direction.

Weights are downloaded, never committed (``docs/ROADMAP.md``, section 7).
With none installed the detector abstains -- 0.5, ``"uncertain"``, and a
``details["reason"]`` naming the directory it looked in and the command that
fills it -- rather than failing a run, the same contract as
:mod:`imgforensics.detectors.learned`.

Like that module, this one imports no ``torch`` at module scope: it is
imported for its registration side effect by
:mod:`imgforensics.localization`, and the heavy imports sit inside
:meth:`IMLViTLocalizer.load`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from imgforensics.core.base import BaseDetector
from imgforensics.core.image import ForensicImage
from imgforensics.core.registry import register
from imgforensics.core.types import DetectionResult, label_from_score
from imgforensics.data.acquire import _sha256_of_file
from imgforensics.localization._scoring import TOP_FRACTION, top_fraction_score
from imgforensics.localization.weights import weights_file

if TYPE_CHECKING:  # pragma: no cover - import-time typing only, never at runtime
    import torch

#: Side length the model was trained at; images are padded (never resized) to it.
INPUT_SIZE = 1024

#: Step between tile origins for images larger than :data:`INPUT_SIZE`; the
#: resulting 256 px overlap is averaged, which keeps a manipulated region that
#: straddles a tile seam from being judged twice at a tile edge.
_TILE_STRIDE = 768

#: ImageNet normalization, from upstream's ``albu.Normalize`` defaults.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

#: Fraction of the heatmap the image-level score averages over (see the module
#: docstring); always at least one pixel. Shared with every other localizer
#: through :mod:`imgforensics.localization._scoring`.
_TOP_FRACTION = TOP_FRACTION

#: Probability above which a pixel counts as manipulated, matching the
#: threshold used in upstream's demo notebook and in this project's pixel
#: metrics (:func:`imgforensics.eval.metrics.pixel_f1`).
_MASK_THRESHOLD = 0.5

_ABSTAIN_SCORE = 0.5
_FETCH_HINT = "imgforensics weights fetch iml_vit --accept-license"
_SHA_PREFIX_LENGTH = 12


def _tile_origins(extent: int, tile: int = INPUT_SIZE, stride: int = _TILE_STRIDE) -> list[int]:
    """Tile start offsets covering ``extent`` pixels with ``tile``-wide windows.

    One origin at 0 when the image fits inside a tile (the padded case).
    Otherwise origins step by ``stride`` and the last one is pulled back to
    ``extent - tile``, so every tile is full-size and the final overlap is
    merely larger than the rest -- no partial tile is ever fed to the model,
    and no strip of the image is left unlooked-at.
    """
    if extent <= tile:
        return [0]
    origins = list(range(0, extent - tile, stride))
    origins.append(extent - tile)
    return origins


def _normalized_tile(pixels: np.ndarray) -> np.ndarray:
    """Pad a HxWx3 uint8 tile to 1024x1024 at the top-left, then normalize it.

    The order matters and follows upstream: zeros are written in *pixel*
    space and only then normalized, so the padding reaches the model as
    ``-mean/std`` rather than as zeros.
    """
    height, width = pixels.shape[:2]
    padded = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    padded[:height, :width] = pixels
    scaled = padded.astype(np.float32) / 255.0
    return (scaled - np.asarray(_MEAN, dtype=np.float32)) / np.asarray(_STD, dtype=np.float32)


@register("iml_vit")
class IMLViTLocalizer(BaseDetector):
    """Pixel-level manipulation localizer built on the released IML-ViT weights.

    One prediction is: cut the image into 1024 px tiles (one padded tile for
    anything that already fits), run each through the model, average the
    overlaps, and crop the result back to the image's own shape. The heatmap
    *is* the output; the image-level score is derived from it (see the module
    docstring).

    The model is loaded once per instance and reused across :meth:`predict`
    calls, so the benchmark runner -- one instance, many images -- pays the
    ~90 M-parameter load once.
    """

    name = "iml_vit"

    def __init__(self, weights_dir: str | Path | None = None, device: str = "auto") -> None:
        """Point the localizer at a weights directory (nothing is read yet).

        Args:
            weights_dir: Base directory holding ``iml_vit/<filename>``.
                ``None`` falls back to ``$IMGFORENSICS_WEIGHTS_DIR`` and then
                ``weights/`` -- resolved at :meth:`load` time, so setting the
                environment variable after constructing the detector still
                works.
            device: ``"auto"`` (CUDA when visible), ``"cpu"``, or an explicit
                device string.
        """
        self._configured_dir = weights_dir
        self._configured_device = device
        self.device = "cpu"
        self.weights_path = weights_file(self.name, weights_dir)
        self._model: torch.nn.Module | None = None
        self._digest: str | None = None
        self._attempted = False
        self._reason: str | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether the pretrained weights were found and loaded."""
        return self._model is not None

    def load(self, device: str = "auto") -> None:
        """Read the checkpoint, if there is one, and build the model on ``device``.

        Never raises for missing weights -- the localizer stays unloaded and
        :meth:`predict` abstains with a reason. ``device`` defaults to
        ``"auto"`` rather than the base class's ``"cpu"``: callers that run
        this detector at all call ``load()`` with no argument, and a 1024 px
        ViT belongs on the GPU when there is one.
        """
        self._attempted = True
        self.weights_path = weights_file(self.name, self._configured_dir)

        if not self.weights_path.is_file():
            self._reason = f"no IML-ViT weights found at {self.weights_path}; run: {_FETCH_HINT}"
            return

        import torch

        from imgforensics.detectors.backbones import resolve_device
        from imgforensics.localization._vendor.iml_vit import IMLViTModel

        requested = device if device != "auto" else self._configured_device
        self.device = resolve_device(requested)

        state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            # A checkpoint straight out of upstream's main_train.py wraps the
            # weights next to the optimizer state; the released file does not.
            state = state["model"]

        model = IMLViTModel(input_size=INPUT_SIZE)
        model.load_state_dict(state, strict=True)
        model.eval()
        model.requires_grad_(False)

        self._model = model.to(torch.device(self.device))
        self._digest = self._file_digest()
        self._reason = None

    def _file_digest(self) -> str:
        """The sha256 of the file that was actually loaded.

        Hashed here rather than copied out of the weights registry, so the
        digest reported in ``details`` describes the bytes on this machine
        even when a checkpoint was placed by hand instead of fetched. One
        pass over a 350 MB file, once per :meth:`load`.
        """
        return _sha256_of_file(self.weights_path)

    def _abstain(self) -> DetectionResult:
        """The result returned when no weights are available."""
        reason = self._reason or (
            f"no IML-ViT weights found at {self.weights_path}; run: {_FETCH_HINT}"
        )
        return DetectionResult(
            detector=self.name,
            score=_ABSTAIN_SCORE,
            label="uncertain",
            details={"reason": reason},
        )

    def _tiles(self, pixels: np.ndarray) -> list[tuple[int, int, int, int]]:
        """``(top, left, height, width)`` of every tile covering ``pixels``."""
        height, width = pixels.shape[:2]
        return [
            (top, left, min(INPUT_SIZE, height - top), min(INPUT_SIZE, width - left))
            for top in _tile_origins(height)
            for left in _tile_origins(width)
        ]

    def _infer(self, batch: np.ndarray) -> np.ndarray:
        """Run the model on a ``(N, 1024, 1024, 3)`` normalized batch -> ``(N, 1024, 1024)``."""
        import torch

        assert self._model is not None
        device = torch.device(self.device)
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().to(device)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ),
        ):
            output = self._model(tensor)
        return output.float().squeeze(1).cpu().numpy()

    def _heatmap(self, pixels: np.ndarray) -> tuple[np.ndarray, int]:
        """Probability map of ``pixels``' own shape, plus the number of tiles used.

        Each tile contributes only over the region it actually covered (its
        bottom-right padding, where present, is dropped), and overlapping
        tiles are averaged. Every pixel is covered by at least one tile by
        construction, so the sum is never divided by zero.
        """
        height, width = pixels.shape[:2]
        totals = np.zeros((height, width), dtype=np.float64)
        counts = np.zeros((height, width), dtype=np.float64)

        tiles = self._tiles(pixels)
        for top, left, tile_height, tile_width in tiles:
            window = pixels[top : top + tile_height, left : left + tile_width]
            predicted = self._infer(_normalized_tile(window)[None, ...])[0]
            totals[top : top + tile_height, left : left + tile_width] += predicted[
                :tile_height, :tile_width
            ]
            counts[top : top + tile_height, left : left + tile_width] += 1.0

        heatmap = np.clip(totals / counts, 0.0, 1.0).astype(np.float32)
        return heatmap, len(tiles)

    @staticmethod
    def _score_from(heatmap: np.ndarray) -> float:
        """Mean of the top :data:`_TOP_FRACTION` of ``heatmap``'s values."""
        return top_fraction_score(heatmap, _TOP_FRACTION)

    def predict(self, image: ForensicImage) -> DetectionResult:
        """Localize manipulated regions, or abstain when no weights are installed."""
        if not self._attempted:
            self.load()
        if self._model is None:
            return self._abstain()

        pixels = np.asarray(image.rgb.convert("RGB") if image.rgb.mode != "RGB" else image.rgb)
        heatmap, tiles = self._heatmap(pixels)
        score = self._score_from(heatmap)

        digest = self._digest
        weights_label = self.weights_path.name
        if digest:
            weights_label = f"{weights_label} ({digest[:_SHA_PREFIX_LENGTH]})"

        details: dict[str, Any] = {
            "weights": weights_label,
            "tiles": tiles,
            "max_prob": round(float(heatmap.max()), 4),
            "mean_prob": round(float(heatmap.mean()), 4),
            f"area_fraction_above_{_MASK_THRESHOLD}": round(
                float((heatmap > _MASK_THRESHOLD).mean()), 4
            ),
            "device": self.device,
        }
        return DetectionResult(
            detector=self.name,
            score=score,
            label=label_from_score(score),
            heatmap=heatmap,
            details=details,
        )


def overlay(image: Image.Image, heatmap: np.ndarray) -> Image.Image:  # pragma: no cover - helper
    """Blend ``heatmap`` over ``image`` as a red wash, for eyeballing a prediction."""
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    weight = np.clip(heatmap, 0.0, 1.0)[..., None]
    red = np.zeros_like(base)
    red[..., 0] = 255.0
    blended = base * (1.0 - 0.5 * weight) + red * (0.5 * weight)
    return Image.fromarray(blended.astype(np.uint8), mode="RGB")
