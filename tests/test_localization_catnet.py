"""Tests for the registered CAT-Net v2 localizer (optional ``ml`` extra).

Offline and CPU-only. The real CAT-Net is a 114 M-parameter two-stream HRNet
and its weights are a several-hundred-megabyte download, so every test here
swaps :class:`~imgforensics.localization._vendor.catnet.CATNet` for
:class:`_StandInModel`: one convolution with the same input and output
contract (a ``(B, 24, H, W)`` batch plus a ``(B, 1, 8, 8)`` quantization
table in, ``(B, 2, H / 4, W / 4)`` logits out). What is under test is the
wrapper -- the JPEG re-encode policy, the DCT volume, the 8-pixel-aligned
tiling, overlap averaging, the score rule, the details and the abstention --
none of which depends on the weights being the real ones.

The coefficient reader those inputs are built from has its own tests in
``test_localization_jpegcoef.py``, which need no ``ml`` extra.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import natural_like_image
from PIL import Image
from typer.testing import CliRunner

torch = pytest.importorskip("torch")

from imgforensics.cli import app  # noqa: E402
from imgforensics.core import registry  # noqa: E402
from imgforensics.core.image import ForensicImage  # noqa: E402
from imgforensics.localization._jpegcoef import read_luma_coefficients  # noqa: E402
from imgforensics.localization._vendor import catnet as vendored  # noqa: E402
from imgforensics.localization.catnet import (  # noqa: E402
    TILE_SIZE,
    CATNetLocalizer,
    _dct_volume,
    _tile_origins,
)
from imgforensics.localization.weights import WEIGHTS  # noqa: E402

pytestmark = pytest.mark.ml

runner = CliRunner()

_SMALL_SIZE = (300, 200)  # (width, height): one tile, padded to whole blocks
_LARGE_SIZE = (1100, 1050)  # (width, height): 2 x 2 tiles once padded to 1104 x 1056
_INPUT_CHANNELS = 3 + vendored.DCT_VOLUME_CHANNELS


class _StandInModel(torch.nn.Module):
    """A single random convolution with CAT-Net's input/output contract.

    A stride-4 convolution takes the input down to the quarter resolution the
    real two-stream network's head predicts at, with the same two channels.
    The quantization table is folded into the output so that a wrongly shaped
    or missing table would fail loudly rather than be silently ignored.
    """

    def __init__(self) -> None:
        super().__init__()
        self.body = torch.nn.Conv2d(_INPUT_CHANNELS, 2, kernel_size=4, stride=4)

    def forward(self, x: torch.Tensor, qtable: torch.Tensor) -> torch.Tensor:
        assert qtable.shape[1:] == (1, 8, 8), qtable.shape
        return self.body(x) * (1.0 + qtable.mean())


@pytest.fixture
def weights_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A weights directory holding a stand-in checkpoint, with the model swapped in.

    The checkpoint is wrapped in the ``{"epoch", "state_dict"}`` envelope the
    released ``.pth.tar`` uses, so the unwrapping in ``load`` is exercised.
    CUDA is hidden as well, so ``device="auto"`` resolves to the CPU even on a
    machine that has a GPU (and, on this project's 6 GB card, so these tests
    never compete with a training run for it).
    """
    monkeypatch.setattr(vendored, "CATNet", _StandInModel)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    checkpoint_dir = tmp_path / "catnet_v2"
    checkpoint_dir.mkdir(parents=True)
    torch.save(
        {"epoch": 0, "state_dict": _StandInModel().state_dict()},
        checkpoint_dir / WEIGHTS["catnet_v2"].filename,
    )
    return tmp_path


def _forensic_image(
    size: tuple[int, int], fmt: str = "PNG", seed: int = 5, **options: Any
) -> ForensicImage:
    buffer = io.BytesIO()
    natural_like_image(size=size, seed=seed).save(buffer, format=fmt, **options)
    return ForensicImage.from_bytes(buffer.getvalue())


def _loaded(weights_dir: Path) -> CATNetLocalizer:
    localizer = CATNetLocalizer(weights_dir=weights_dir)
    localizer.load("cpu")
    assert localizer.is_loaded
    return localizer


def test_registry_exposes_the_localizer_when_torch_is_installed() -> None:
    assert "catnet_v2" in registry.available()
    assert registry.get("catnet_v2") is CATNetLocalizer


def test_tile_origins_stay_on_block_boundaries_and_cover_the_extent() -> None:
    assert _tile_origins(1024, 1024) == [0]
    assert _tile_origins(200, 200) == [0]
    # Above one tile, origins step by the stride and the last is pulled back
    # so the final tile still ends exactly at the edge.
    assert _tile_origins(1056, TILE_SIZE) == [0, 32]
    assert _tile_origins(3000, TILE_SIZE) == [0, 768, 1536, 1976]
    for extent in (1032, 1056, 1104, 3000):
        origins = _tile_origins(extent, TILE_SIZE)
        assert origins[-1] + TILE_SIZE == extent
        # A tile boundary must never cut a JPEG block in half.
        assert all(origin % 8 == 0 for origin in origins)


def test_the_dct_volume_bins_coefficients_by_magnitude() -> None:
    coefficients = np.array([[0, 1, -1, 19, -19, 20, -400]], dtype=np.int32)

    volume = _dct_volume(coefficients)

    assert volume.shape == (vendored.DCT_VOLUME_CHANNELS, 1, 7)
    assert volume.dtype == np.float32
    # Exactly one channel is set per coefficient, and sign is discarded.
    assert volume.sum(axis=0).tolist() == [[1.0] * 7]
    assert volume[:, 0, :].argmax(axis=0).tolist() == [0, 1, 1, 19, 19, 20, 20]


def test_a_small_png_is_padded_to_whole_blocks_and_cropped_back(weights_dir: Path) -> None:
    image = _forensic_image(_SMALL_SIZE)
    result = _loaded(weights_dir).predict(image)

    assert result.detector == "catnet_v2"
    assert result.details["tiles"] == 1

    heatmap = result.heatmap
    assert heatmap is not None
    # PIL sizes are (width, height); a heatmap is indexed (rows, columns).
    assert heatmap.shape == (image.height, image.width) == (200, 300)
    assert heatmap.dtype == np.float32
    assert 0.0 <= float(heatmap.min()) <= float(heatmap.max()) <= 1.0


def test_a_jpeg_keeps_its_own_coefficients_and_its_shape(weights_dir: Path) -> None:
    image = _forensic_image(_SMALL_SIZE, fmt="JPEG", quality=82)
    result = _loaded(weights_dir).predict(image)

    heatmap = result.heatmap
    assert heatmap is not None
    assert heatmap.shape == (200, 300)
    assert result.details["input_was_jpeg"] is True
    assert result.details["dct_source"] == "original jpeg stream"
    # The metadata signal's estimator, on the file's own quantization table.
    assert result.details["jpeg_quality_estimate"] == pytest.approx(82, abs=2)


def test_a_png_is_re_encoded_at_quality_100_the_way_upstream_does(weights_dir: Path) -> None:
    image = _forensic_image(_SMALL_SIZE)
    result = _loaded(weights_dir).predict(image)

    assert result.details["input_was_jpeg"] is False
    assert result.details["jpeg_quality_estimate"] is None
    assert result.details["dct_source"] == "re-encoded at quality 100"


def test_a_progressive_jpeg_falls_back_to_the_re_encode(weights_dir: Path) -> None:
    image = _forensic_image(_SMALL_SIZE, fmt="JPEG", quality=85, progressive=True)
    result = _loaded(weights_dir).predict(image)

    assert result.heatmap is not None
    assert result.details["input_was_jpeg"] is True
    assert result.details["dct_source"] == "re-encoded at quality 100"


def test_a_large_image_is_tiled_and_still_covers_every_pixel(weights_dir: Path) -> None:
    image = _forensic_image(_LARGE_SIZE, seed=9)
    result = _loaded(weights_dir).predict(image)

    # 1050 px -> padded 1056 -> origins [0, 32]; 1100 px -> padded 1104 ->
    # origins [0, 80]; 2 x 2 tiles of 1024.
    assert result.details["tiles"] == 4

    heatmap = result.heatmap
    assert heatmap is not None
    assert heatmap.shape == (1050, 1100)
    assert 0.0 <= float(heatmap.min()) <= float(heatmap.max()) <= 1.0
    # Averaging overlaps never leaves an uncovered (identically zero) pixel:
    # a softmax over a random convolution is strictly positive everywhere.
    assert float(heatmap.min()) > 0.0


def test_the_score_is_the_mean_of_the_top_one_percent(weights_dir: Path) -> None:
    result = _loaded(weights_dir).predict(_forensic_image(_SMALL_SIZE))

    heatmap = result.heatmap
    assert heatmap is not None
    flat = np.sort(heatmap.reshape(-1))
    keep = max(1, int(round(flat.size * 0.01)))
    assert result.score == pytest.approx(float(flat[-keep:].mean()), abs=1e-6)
    assert 0.0 <= result.score <= 1.0
    assert result.label in {"real", "fake", "uncertain"}


def test_details_report_the_weights_and_the_heatmap_summary(weights_dir: Path) -> None:
    result = _loaded(weights_dir).predict(_forensic_image(_SMALL_SIZE))

    heatmap = result.heatmap
    assert heatmap is not None
    details = result.details
    assert details["weights"].startswith(WEIGHTS["catnet_v2"].filename)
    assert "(" in details["weights"]  # the sha256 prefix of the file actually loaded
    assert details["max_prob"] == pytest.approx(float(heatmap.max()), abs=1e-4)
    assert details["mean_prob"] == pytest.approx(float(heatmap.mean()), abs=1e-4)
    assert details["area_fraction_above_0.5"] == pytest.approx(
        float((heatmap > 0.5).mean()), abs=1e-4
    )
    assert details["device"] == "cpu"


def test_an_exif_rotated_jpeg_uses_a_re_encode_so_pixels_and_coefficients_agree(
    weights_dir: Path,
) -> None:
    # Orientation 6 means "rotate 90 degrees clockwise on display", so the
    # decoded pixels are 300x200 while the stored coefficients are 200x300.
    buffer = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6
    natural_like_image(size=(200, 300), seed=4).save(buffer, format="JPEG", quality=90, exif=exif)
    image = ForensicImage.from_bytes(buffer.getvalue())
    assert (image.width, image.height) == (300, 200)
    assert read_luma_coefficients(buffer.getvalue()).width == 200

    result = _loaded(weights_dir).predict(image)

    assert result.heatmap is not None
    assert result.heatmap.shape == (200, 300)
    assert result.details["dct_source"] == "re-encoded at quality 100"


def test_without_weights_the_localizer_abstains_with_a_reason(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere"
    localizer = CATNetLocalizer(weights_dir=missing)
    localizer.load("cpu")

    result = localizer.predict(_forensic_image((64, 64)))

    assert localizer.is_loaded is False
    assert result.score == 0.5
    assert result.label == "uncertain"
    assert result.heatmap is None
    reason = result.details["reason"]
    assert str(missing) in reason
    assert "imgforensics weights fetch catnet_v2 --accept-license" in reason


def test_predict_loads_lazily_when_load_was_never_called(tmp_path: Path) -> None:
    result = CATNetLocalizer(weights_dir=tmp_path / "nowhere").predict(_forensic_image((64, 64)))

    assert result.score == 0.5
    assert "no CAT-Net weights found" in result.details["reason"]


def test_cli_analyze_runs_the_localizer_from_the_env_var(
    weights_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMGFORENSICS_WEIGHTS_DIR", str(weights_dir))
    image_path = tmp_path / "sample.jpg"
    natural_like_image(size=_SMALL_SIZE, seed=11).save(image_path, format="JPEG", quality=90)
    heatmap_dir = tmp_path / "heatmaps"

    result = runner.invoke(
        app,
        [
            "analyze",
            str(image_path),
            "--json",
            "--detector",
            "catnet_v2",
            "--save-heatmaps",
            str(heatmap_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout
    entry: dict[str, Any] = json.loads(result.stdout)["results"][0]
    assert entry["detector"] == "catnet_v2"
    assert 0.0 <= entry["score"] <= 1.0
    assert entry["details"]["tiles"] == 1
    assert entry["heatmap"] == str(heatmap_dir / "sample_catnet_v2.png")
    assert (heatmap_dir / "sample_catnet_v2.png").is_file()


def test_cli_analyze_reports_the_abstention_when_no_weights_are_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMGFORENSICS_WEIGHTS_DIR", str(tmp_path / "nowhere"))
    image_path = tmp_path / "sample.png"
    natural_like_image(size=(64, 64), seed=2).save(image_path, format="PNG")

    result = runner.invoke(app, ["analyze", str(image_path), "--json", "--detector", "catnet_v2"])

    assert result.exit_code == 0, result.stdout
    entry = json.loads(result.stdout)["results"][0]
    assert entry["score"] == 0.5
    assert entry["label"] == "uncertain"
    assert "no CAT-Net weights found" in entry["details"]["reason"]
