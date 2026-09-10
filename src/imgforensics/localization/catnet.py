"""CAT-Net v2 as a registered pixel-level localizer.

CAT-Net ("CAT-Net: Compression Artifact Tracing Network for Detection and
Localization of Image Splicing", WACV 2021, extended in IJCV 2022, code at
https://github.com/mjkwon2021/CAT-Net) is the second pretrained localizer of
Phase 4a (``docs/ROADMAP.md``; ``docs/research/02_manipulation_localization.md``
ranks it first among license-clean options). Where
:mod:`imgforensics.localization.iml_vit` looks only at pixels, this one also
reads the JPEG stream itself. Sources for everything asserted below, each read
on 2026-09-10 at commit ``331b8059c3f55efec1d9075de79dd153413f2061``
(2026-08-05, the commit that relicensed the project):

- License: ``README.md`` -- code Apache-2.0, weights and datasets CC-BY-4.0,
  "You are now free to use them for commercial purposes, provided you give
  proper attribution."
- Architecture: ``lib/models/network_CAT.py`` plus the hyper-parameters in
  ``experiments/CAT_full.yaml`` -- an HRNetV2-W48 RGB stream (stages 1-4)
  next to a DCT stream (a dilated 3x3 convolution over the coefficient
  volume, a 1x1 tail, a per-block reshape that turns the 8x8 within-block
  positions into 64 channels both raw and multiplied by the quantization
  table, then HRNet stages 3-4), fused branch-wise into a shared stage 5 and
  a two-class head. 114.3 M parameters.
- Checkpoint: ``README.md`` -> a Google Drive folder holding
  ``CAT_full_v2.pth.tar`` (the v2 release trained with tampCOCO and
  compRAISE). See :mod:`imgforensics.localization.weights` for the file id,
  size and digest.
- Inputs: ``tools/infer.py`` builds its dataset with ``crop_size=None,
  grid_crop=True, blocks=('RGB', 'DCTvol', 'qtable'), DCT_channels=1``, so
  inference runs at **full resolution**, padded up to the smallest 8-pixel
  grid containing the image (``Splicing/data/AbstractDataset.py``: RGB padded
  with 127.5, coefficients with 0), with no resizing anywhere. RGB is
  normalized as ``(x - 127.5) / 127.5``. The DCT volume is a 21-channel
  histogram of the *quantized luminance* coefficients: channel 0 marks
  coefficients equal to 0, channels 1-19 mark magnitude ``i``, and channel 20
  marks magnitude >= 20. The quantization table is the luminance one, passed
  separately as ``(B, 1, 8, 8)``.
- Non-JPEG input: ``Splicing/data/dataset_arbitrary.py`` re-encodes it first
  -- ``Image.open(path).convert('RGB').save(temp_jpg, quality=100,
  subsampling=0)`` -- and reads the coefficients back out of that. The model
  has no other way to be fed: its second stream is defined on a JPEG stream.
  This wrapper does the same thing in memory.
- Output: ``tools/infer.py`` -> ``F.softmax(pred, dim=0)[1]``, the
  manipulated-class probability of a two-class head, at one quarter of the
  input resolution.

**Reading the coefficients.** Upstream uses ``jpegio``; this project decodes
them with :mod:`imgforensics.localization._jpegcoef` instead, which explains
why in its own docstring.

**Image-level score.** CAT-Net reports pixel-level metrics only, so, exactly
as for IML-ViT, this detector reports the **mean of the top 1% of heatmap
values** (:data:`_TOP_FRACTION`) -- a statistic that needs a confident
*region* rather than one hot pixel, and that does not shrink as the
manipulated area does. Keeping the rule identical to the other localizer's is
deliberate: the two are meant to be compared on the same benchmark, and a
different score rule would make their image-level AUCs incomparable.

**Full resolution, then tiles.** Anything up to :data:`TILE_SIZE` on both
sides runs whole, as upstream does. Larger images are cut into
:data:`TILE_SIZE` tiles at stride :data:`_TILE_STRIDE` and the overlaps are
averaged, because a two-stream HRNet at full resolution outgrows a 6 GB card
somewhere past 1024x1024. Both numbers are multiples of 8 and every tile
origin lands on a multiple of 8, so a tile boundary never cuts a DCT block in
half -- which for this model would corrupt the very cue it is reading.

Weights are downloaded, never committed (``docs/ROADMAP.md``, section 7).
With none installed the detector abstains -- 0.5, ``"uncertain"``, and a
``details["reason"]`` naming the directory it looked in and the command that
fills it -- rather than failing a run.

Like :mod:`imgforensics.localization.iml_vit`, this module imports no
``torch`` at module scope: it is imported for its registration side effect by
:mod:`imgforensics.localization`, and the heavy imports sit inside
:meth:`CATNetLocalizer.load`.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from imgforensics.core.base import BaseDetector
from imgforensics.core.image import ForensicImage
from imgforensics.core.registry import register
from imgforensics.core.types import DetectionResult, label_from_score
from imgforensics.data.acquire import _sha256_of_file
from imgforensics.localization._jpegcoef import (
    JpegCoefficients,
    UnsupportedJpegError,
    read_luma_coefficients,
)
from imgforensics.localization.weights import weights_file

if TYPE_CHECKING:  # pragma: no cover - import-time typing only, never at runtime
    import torch

#: Side length above which an image is tiled instead of run whole.
TILE_SIZE = 1024

#: Step between tile origins. A multiple of 8, like :data:`TILE_SIZE`, so tile
#: boundaries fall on JPEG block boundaries; the 256 px overlap is averaged.
_TILE_STRIDE = 768

#: JPEG block size: every tensor fed to the model has both sides a multiple of
#: this, because the DCT stream folds each 8x8 block into 64 channels.
_BLOCK = 8

#: Largest coefficient magnitude the DCT volume distinguishes (upstream's
#: ``T``); the volume has ``T + 1`` channels.
_DCT_VOLUME_MAX = 20

#: Upstream's RGB normalization: ``(x - 127.5) / 127.5``, and the value the
#: padding is written in *before* normalizing, so padding reaches the model
#: as zero.
_RGB_OFFSET = 127.5

#: Fraction of the heatmap the image-level score averages over (see the module
#: docstring); always at least one pixel.
_TOP_FRACTION = 0.01

#: Probability above which a pixel counts as manipulated, matching this
#: project's pixel metrics (:func:`imgforensics.eval.metrics.pixel_f1`).
_MASK_THRESHOLD = 0.5

_ABSTAIN_SCORE = 0.5
_FETCH_HINT = "imgforensics weights fetch catnet_v2 --accept-license"
_SHA_PREFIX_LENGTH = 12


def _tile_origins(extent: int, tile: int, stride: int = _TILE_STRIDE) -> list[int]:
    """Tile start offsets covering ``extent`` pixels with ``tile``-wide windows.

    One origin at 0 when the image fits in a single tile. Otherwise origins
    step by ``stride`` and the last is pulled back to ``extent - tile``, so
    every tile is full-size and only the final overlap is larger than the
    rest. ``extent``, ``tile`` and ``stride`` are all multiples of 8 by
    construction, so every origin is too.
    """
    if extent <= tile:
        return [0]
    origins = list(range(0, extent - tile, stride))
    origins.append(extent - tile)
    return origins


def _dct_volume(coefficients: np.ndarray) -> np.ndarray:
    """Upstream's ``DCTvol``: a ``(21, H, W)`` histogram of coefficient magnitudes.

    Channel 0 marks a coefficient of exactly zero, channels 1 to 19 mark
    magnitude ``i``, and channel 20 marks magnitude 20 or more. Sign is
    discarded, which is upstream's encoding (``t_DCT_vol[i] += (coef == i) +
    (coef == -i)``) written as one scatter instead of twenty comparisons.
    """
    height, width = coefficients.shape
    bins = np.minimum(np.abs(coefficients), _DCT_VOLUME_MAX).astype(np.intp)
    volume = np.zeros((_DCT_VOLUME_MAX + 1, height, width), dtype=np.float32)
    np.put_along_axis(volume, bins[None, ...], 1.0, axis=0)
    return volume


def _padded_rgb(pixels: np.ndarray) -> np.ndarray:
    """Pad ``pixels`` out to whole 8x8 blocks with 127.5, as upstream does."""
    height, width = pixels.shape[:2]
    grid_height = -(-height // _BLOCK) * _BLOCK
    grid_width = -(-width // _BLOCK) * _BLOCK
    if (grid_height, grid_width) == (height, width):
        return pixels.astype(np.float32)
    padded = np.full((grid_height, grid_width, 3), _RGB_OFFSET, dtype=np.float32)
    padded[:height, :width] = pixels
    return padded


def _reencode(image: Image.Image) -> bytes:
    """Encode ``image`` as the quality-100, 4:4:4 JPEG CAT-Net expects to read."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=100, subsampling=0)
    return buffer.getvalue()


def _coefficients_for(image: ForensicImage) -> tuple[JpegCoefficients, bool]:
    """The JPEG coefficients to run on, and whether they came from the original file.

    The original stream is used only when it is a JPEG this decoder handles
    *and* its dimensions still match the pixels the rest of the pipeline sees
    -- an EXIF-rotated photograph is transposed by
    :func:`imgforensics.utils.image_io.load_image` but its coefficients are
    not, and a heatmap that disagrees with its own image is worse than one
    computed from a re-encode. Everything else goes through the quality-100
    re-encode upstream uses for non-JPEG input.
    """
    if image.raw is not None and image.format == "JPEG":
        try:
            coefficients = read_luma_coefficients(image.raw)
        except UnsupportedJpegError:
            pass
        else:
            if (coefficients.height, coefficients.width) == (image.height, image.width):
                return coefficients, True
    return read_luma_coefficients(_reencode(image.rgb)), False


def _load_checkpoint(torch: Any, path: Path) -> dict[str, Any]:
    """Read the weights out of the released ``.pth.tar``, without unpickling code.

    The file is a training checkpoint: the weights sit under ``state_dict``
    (which is the key upstream's ``infer.py`` unwraps) next to the epoch
    counter, the optimizer state and a ``best_p_mIoU`` written as a *numpy
    scalar*. That last one is why this is not a bare ``torch.load(...,
    weights_only=True)``: torch's weights-only unpickler refuses every global
    it has not been told about, and rebuilding one numpy float takes three --
    the array-scalar reconstructor and the two dtype classes it is called
    with. Naming those three, and nothing else, keeps the guarantee that
    matters: no arbitrary class from the file is instantiated.

    ``scalar`` is spelled out under the name the checkpoint pickled it as,
    because numpy 2 moved it to a private module and torch matches globals by
    their pickled path.
    """
    import numpy

    try:  # numpy >= 2; `numpy.core` still resolves there but is a deprecated shim
        from numpy._core.multiarray import scalar
    except ImportError:  # pragma: no cover - numpy 1.x
        from numpy.core.multiarray import scalar  # type: ignore[no-redef]

    allowed = [
        (scalar, "numpy.core.multiarray.scalar"),
        numpy.dtype,
        numpy.dtypes.Float64DType,
    ]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return dict(checkpoint["state_dict"])
    return dict(checkpoint)


@register("catnet_v2")
class CATNetLocalizer(BaseDetector):
    """Pixel-level manipulation localizer built on the released CAT-Net v2 weights.

    One prediction is: read (or re-create) the image's JPEG stream, build the
    RGB and DCT-volume inputs from it, run the model at full resolution or in
    1024 px tiles, average the overlaps, and crop back to the image's own
    shape. The heatmap *is* the output; the image-level score is derived from
    it (see the module docstring).

    The model is loaded once per instance and reused across :meth:`predict`
    calls, so the benchmark runner -- one instance, many images -- pays the
    114 M-parameter load once.
    """

    name = "catnet_v2"

    def __init__(self, weights_dir: str | Path | None = None, device: str = "auto") -> None:
        """Point the localizer at a weights directory (nothing is read yet).

        Args:
            weights_dir: Base directory holding ``catnet_v2/<filename>``.
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
        this detector at all call ``load()`` with no argument, and a
        two-stream HRNet belongs on the GPU when there is one.
        """
        self._attempted = True
        self.weights_path = weights_file(self.name, self._configured_dir)

        if not self.weights_path.is_file():
            self._reason = f"no CAT-Net weights found at {self.weights_path}; run: {_FETCH_HINT}"
            return

        import torch

        from imgforensics.detectors.backbones import resolve_device
        from imgforensics.localization._vendor.catnet import CATNet

        requested = device if device != "auto" else self._configured_device
        self.device = resolve_device(requested)

        state = _load_checkpoint(torch, self.weights_path)

        model = CATNet()
        model.load_state_dict(state, strict=True)
        model.eval()
        model.requires_grad_(False)

        self._model = model.to(torch.device(self.device))
        self._digest = _sha256_of_file(self.weights_path)
        self._reason = None

    def _abstain(self) -> DetectionResult:
        """The result returned when no weights are available."""
        reason = self._reason or (
            f"no CAT-Net weights found at {self.weights_path}; run: {_FETCH_HINT}"
        )
        return DetectionResult(
            detector=self.name,
            score=_ABSTAIN_SCORE,
            label="uncertain",
            details={"reason": reason},
        )

    def _infer(self, tensor: torch.Tensor, qtable: torch.Tensor) -> np.ndarray:
        """Run the model on one ``(1, 24, H, W)`` tile -> an ``(H, W)`` probability map.

        The two-class logits come out at a quarter of the input resolution;
        they are upsampled first and turned into probabilities second, which
        is the order upstream's evaluation code uses.
        """
        import torch

        assert self._model is not None
        device = torch.device(self.device)
        tensor = tensor.to(device)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ),
        ):
            logits = self._model(tensor, qtable.to(device))
            upsampled = torch.nn.functional.interpolate(
                logits.float(), size=tensor.shape[-2:], mode="bilinear", align_corners=False
            )
            probabilities = torch.softmax(upsampled, dim=1)[:, 1]
        return probabilities.float().cpu().numpy()[0]

    def _heatmap(
        self, pixels: np.ndarray, coefficients: JpegCoefficients
    ) -> tuple[np.ndarray, int]:
        """Probability map of ``pixels``' own shape, plus the number of tiles used.

        Overlapping tiles are averaged, and the 8-pixel padding added to reach
        whole blocks is cropped off at the end.
        """
        import torch

        height, width = pixels.shape[:2]
        padded = _padded_rgb(pixels)
        grid_height, grid_width = padded.shape[:2]
        normalized = (padded - _RGB_OFFSET) / _RGB_OFFSET
        blocks = coefficients.coefficients

        qtable = torch.from_numpy(
            coefficients.quantization.astype(np.float32)[None, None, ...]
        ).contiguous()

        totals = np.zeros((grid_height, grid_width), dtype=np.float64)
        counts = np.zeros((grid_height, grid_width), dtype=np.float64)

        tile_height = min(TILE_SIZE, grid_height)
        tile_width = min(TILE_SIZE, grid_width)
        tiles = 0
        for top in _tile_origins(grid_height, tile_height):
            for left in _tile_origins(grid_width, tile_width):
                bottom, right = top + tile_height, left + tile_width
                volume = _dct_volume(blocks[top:bottom, left:right])
                window = normalized[top:bottom, left:right].transpose(2, 0, 1)
                tensor = torch.from_numpy(
                    np.concatenate([window, volume], axis=0)[None, ...]
                ).contiguous()
                totals[top:bottom, left:right] += self._infer(tensor, qtable)
                counts[top:bottom, left:right] += 1.0
                tiles += 1

        heatmap = np.clip(totals / counts, 0.0, 1.0)[:height, :width]
        return heatmap.astype(np.float32), tiles

    @staticmethod
    def _score_from(heatmap: np.ndarray) -> float:
        """Mean of the top :data:`_TOP_FRACTION` of ``heatmap``'s values."""
        flat = heatmap.reshape(-1)
        if flat.size == 0:  # pragma: no cover - ForensicImage always has pixels
            return _ABSTAIN_SCORE
        keep = max(1, int(round(flat.size * _TOP_FRACTION)))
        top = np.partition(flat, flat.size - keep)[flat.size - keep :]
        return float(np.clip(top.mean(), 0.0, 1.0))

    def _quality_estimate(self, image: ForensicImage) -> int | None:
        """The input's own JPEG quality, or ``None`` when it was not a JPEG.

        Reuses :mod:`imgforensics.signals.metadata`'s estimator rather than
        reading the table this detector already decoded, so the number means
        the same thing here as in the ``metadata`` signal's details -- and so
        it describes the *input*, not the quality-100 re-encode a non-JPEG
        goes through.
        """
        from imgforensics.signals.metadata import _jpeg_quality_info

        if image.raw is None:
            return None
        with image.open_original() as original:
            estimate = _jpeg_quality_info(original)["jpeg_quality_estimate"]
        return int(estimate) if estimate is not None else None

    def predict(self, image: ForensicImage) -> DetectionResult:
        """Localize manipulated regions, or abstain when no weights are installed."""
        if not self._attempted:
            self.load()
        if self._model is None:
            return self._abstain()

        coefficients, from_original = _coefficients_for(image)
        pixels = np.asarray(image.rgb.convert("RGB") if image.rgb.mode != "RGB" else image.rgb)
        heatmap, tiles = self._heatmap(pixels, coefficients)
        score = self._score_from(heatmap)

        weights_label = self.weights_path.name
        if self._digest:
            weights_label = f"{weights_label} ({self._digest[:_SHA_PREFIX_LENGTH]})"

        details: dict[str, Any] = {
            "weights": weights_label,
            "tiles": tiles,
            "max_prob": round(float(heatmap.max()), 4),
            "mean_prob": round(float(heatmap.mean()), 4),
            f"area_fraction_above_{_MASK_THRESHOLD}": round(
                float((heatmap > _MASK_THRESHOLD).mean()), 4
            ),
            "input_was_jpeg": image.format == "JPEG",
            "jpeg_quality_estimate": self._quality_estimate(image),
            "dct_source": "original jpeg stream" if from_original else "re-encoded at quality 100",
            "device": self.device,
        }
        return DetectionResult(
            detector=self.name,
            score=score,
            label=label_from_score(score),
            heatmap=heatmap,
            details=details,
        )
