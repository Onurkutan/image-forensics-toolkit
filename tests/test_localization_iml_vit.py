"""Tests for the registered IML-ViT localizer (optional ``ml`` extra).

Offline and CPU-only. The real IML-ViT is a 92 M-parameter ViT at 1024 px and
its weights are a 350 MB download, so every test here swaps
:class:`~imgforensics.localization._vendor.iml_vit.IMLViTModel` for
:class:`_StandInModel`: a randomly initialised convolutional stack with the
same input and output contract (a ``(B, 3, 1024, 1024)`` batch in, a
``(B, 1, 1024, 1024)`` probability map out). What is under test is the
wrapper -- padding, tiling, overlap averaging, cropping back, the score rule,
the abstention and the CLI path -- none of which depends on the weights being
the real ones.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import natural_like_image
from typer.testing import CliRunner

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from imgforensics.cli import app  # noqa: E402
from imgforensics.core import registry  # noqa: E402
from imgforensics.core.image import ForensicImage  # noqa: E402
from imgforensics.localization._vendor import iml_vit as vendored  # noqa: E402
from imgforensics.localization.iml_vit import (  # noqa: E402
    INPUT_SIZE,
    IMLViTLocalizer,
    _tile_origins,
)
from imgforensics.localization.weights import WEIGHTS  # noqa: E402

pytestmark = pytest.mark.ml

runner = CliRunner()

_SMALL_SIZE = (300, 200)  # (width, height): fits inside one padded tile
_LARGE_SIZE = (1500, 1100)  # (width, height): 2 x 2 tiles at stride 768


class _StandInModel(torch.nn.Module):
    """A tiny random conv net with IML-ViT's input/output contract.

    Strided convolutions take the 1024 px input down to a 64x64 logit map --
    the same token grid the real patch-16 encoder produces -- which is then
    bilinearly upsampled back to 1024 and squashed with a sigmoid, exactly as
    the vendored model does. The weights are random, so the values mean
    nothing; the shapes, the range and the spatial correspondence are what
    the tests read.
    """

    def __init__(self, input_size: int = INPUT_SIZE) -> None:
        super().__init__()
        self.input_size = input_size
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=8, stride=8),
            torch.nn.GELU(),
            torch.nn.Conv2d(4, 1, kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.body(x)
        upsampled = torch.nn.functional.interpolate(
            logits, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False
        )
        return torch.sigmoid(upsampled)


@pytest.fixture
def weights_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A weights directory holding a stand-in checkpoint, with the model swapped in.

    Also hides CUDA, so ``device="auto"`` resolves to the CPU even on a
    machine that has a GPU (and, on this project's 6 GB card, so these tests
    never compete with a training run for it).
    """
    monkeypatch.setattr(vendored, "IMLViTModel", _StandInModel)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    checkpoint_dir = tmp_path / "iml_vit"
    checkpoint_dir.mkdir(parents=True)
    torch.save(_StandInModel().state_dict(), checkpoint_dir / WEIGHTS["iml_vit"].filename)
    return tmp_path


def _forensic_image(size: tuple[int, int], seed: int = 5) -> ForensicImage:
    buffer = io.BytesIO()
    natural_like_image(size=size, seed=seed).save(buffer, format="PNG")
    return ForensicImage.from_bytes(buffer.getvalue())


def _loaded(weights_dir: Path) -> IMLViTLocalizer:
    localizer = IMLViTLocalizer(weights_dir=weights_dir)
    localizer.load("cpu")
    assert localizer.is_loaded
    return localizer


def test_registry_exposes_the_localizer_when_torch_is_installed() -> None:
    assert "iml_vit" in registry.available()
    assert registry.get("iml_vit") is IMLViTLocalizer


def test_tile_origins_cover_the_extent_with_full_tiles() -> None:
    # Anything at or below the tile size is one padded tile.
    assert _tile_origins(200) == [0]
    assert _tile_origins(INPUT_SIZE) == [0]
    # Above it, origins step by the stride and the last is pulled back so the
    # final tile still ends exactly at the edge.
    assert _tile_origins(1100) == [0, 76]
    assert _tile_origins(1500) == [0, 476]
    assert _tile_origins(3000) == [0, 768, 1536, 1976]
    for extent in (1025, 1100, 1500, 3000):
        origins = _tile_origins(extent)
        assert origins[-1] + INPUT_SIZE == extent
        assert all(0 <= origin <= extent - INPUT_SIZE for origin in origins)


def test_a_small_image_is_padded_into_one_tile_and_cropped_back(weights_dir: Path) -> None:
    image = _forensic_image(_SMALL_SIZE)
    result = _loaded(weights_dir).predict(image)

    assert result.detector == "iml_vit"
    assert result.details["tiles"] == 1

    heatmap = result.heatmap
    assert heatmap is not None
    # PIL sizes are (width, height); a heatmap is indexed (rows, columns).
    assert heatmap.shape == (image.height, image.width) == (200, 300)
    assert heatmap.dtype == np.float32
    assert 0.0 <= float(heatmap.min()) <= float(heatmap.max()) <= 1.0


def test_a_large_image_is_tiled_and_still_covers_every_pixel(weights_dir: Path) -> None:
    image = _forensic_image(_LARGE_SIZE)
    result = _loaded(weights_dir).predict(image)

    # 1500 px -> origins [0, 476]; 1100 px -> origins [0, 76]; 2 x 2 tiles.
    assert result.details["tiles"] == 4

    heatmap = result.heatmap
    assert heatmap is not None
    assert heatmap.shape == (1100, 1500)
    assert 0.0 <= float(heatmap.min()) <= float(heatmap.max()) <= 1.0
    # Averaging overlaps never leaves an uncovered (identically zero) pixel:
    # a random conv net's sigmoid is strictly positive everywhere.
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
    localizer = _loaded(weights_dir)
    result = localizer.predict(_forensic_image(_SMALL_SIZE))

    heatmap = result.heatmap
    assert heatmap is not None
    details = result.details
    assert details["weights"].startswith(WEIGHTS["iml_vit"].filename)
    assert "(" in details["weights"]  # the sha256 prefix of the file actually loaded
    assert details["max_prob"] == pytest.approx(float(heatmap.max()), abs=1e-4)
    assert details["mean_prob"] == pytest.approx(float(heatmap.mean()), abs=1e-4)
    assert details["area_fraction_above_0.5"] == pytest.approx(
        float((heatmap > 0.5).mean()), abs=1e-4
    )
    assert details["device"] == "cpu"


def test_without_weights_the_localizer_abstains_with_a_reason(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere"
    localizer = IMLViTLocalizer(weights_dir=missing)
    localizer.load("cpu")

    result = localizer.predict(_forensic_image((64, 64)))

    assert localizer.is_loaded is False
    assert result.score == 0.5
    assert result.label == "uncertain"
    assert result.heatmap is None
    reason = result.details["reason"]
    assert str(missing) in reason
    assert "imgforensics weights fetch iml_vit --accept-license" in reason


def test_predict_loads_lazily_when_load_was_never_called(tmp_path: Path) -> None:
    result = IMLViTLocalizer(weights_dir=tmp_path / "nowhere").predict(_forensic_image((64, 64)))

    assert result.score == 0.5
    assert "no IML-ViT weights found" in result.details["reason"]


def test_cli_analyze_runs_the_localizer_from_the_env_var(
    weights_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMGFORENSICS_WEIGHTS_DIR", str(weights_dir))
    image_path = tmp_path / "sample.png"
    natural_like_image(size=_SMALL_SIZE, seed=11).save(image_path, format="PNG")
    heatmap_dir = tmp_path / "heatmaps"

    result = runner.invoke(
        app,
        [
            "analyze",
            str(image_path),
            "--json",
            "--detector",
            "iml_vit",
            "--save-heatmaps",
            str(heatmap_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout
    entry: dict[str, Any] = json.loads(result.stdout)["results"][0]
    assert entry["detector"] == "iml_vit"
    assert 0.0 <= entry["score"] <= 1.0
    assert entry["details"]["tiles"] == 1
    assert entry["heatmap"] == str(heatmap_dir / "sample_iml_vit.png")
    assert (heatmap_dir / "sample_iml_vit.png").is_file()


def test_cli_analyze_reports_the_abstention_when_no_weights_are_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMGFORENSICS_WEIGHTS_DIR", str(tmp_path / "nowhere"))
    image_path = tmp_path / "sample.png"
    natural_like_image(size=(64, 64), seed=2).save(image_path, format="PNG")

    result = runner.invoke(app, ["analyze", str(image_path), "--json", "--detector", "iml_vit"])

    assert result.exit_code == 0, result.stdout
    entry = json.loads(result.stdout)["results"][0]
    assert entry["score"] == 0.5
    assert entry["label"] == "uncertain"
    assert "no IML-ViT weights found" in entry["details"]["reason"]
