"""Tests for the registered learned detector (optional ``ml`` extra).

Offline and CPU-only, like the feature tests: ``load_backbone`` is replaced by
a randomly initialised ``vit_tiny_patch16_224`` (no download, 192-dimensional
features) and CUDA is hidden, so ``device="auto"`` resolves to the CPU. A real
head is trained on that tiny backbone's features inside the fixture, because
the point of these tests is the whole path -- checkpoint on disk, backbone,
head, calibration, heatmap, CLI -- not the quality of the predictions, which a
random backbone cannot have.
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
timm = pytest.importorskip("timm")
pytest.importorskip("safetensors")

from imgforensics.cli import app  # noqa: E402
from imgforensics.core import registry  # noqa: E402
from imgforensics.core.image import ForensicImage  # noqa: E402
from imgforensics.data.manifest import build_manifest, label_from_parent_folder  # noqa: E402
from imgforensics.detectors import attribution as attribution_module  # noqa: E402
from imgforensics.detectors import backbones  # noqa: E402
from imgforensics.detectors.crops import CropPolicy  # noqa: E402
from imgforensics.detectors.head import HeadOptions  # noqa: E402
from imgforensics.detectors.learned import (  # noqa: E402
    HEAD_ATTRIBUTION_ENV,
    HEAD_DIR_ENV,
    LearnedDetector,
    resolve_checkpoint_dir,
)
from imgforensics.detectors.train import TrainConfig, train_head  # noqa: E402

pytestmark = pytest.mark.ml

runner = CliRunner()

_TINY_MODEL = "vit_tiny_patch16_224"
_TINY_DIM = 192
_CROP_SIZE = 224
_TRAIN_SIZE = (320, 320)  # one 224 grid tile
_PREDICT_SIZE = (480, 480)  # four 224 grid tiles, covering [0:448, 0:448]
_POLICY = CropPolicy(size=_CROP_SIZE, mode="grid", max_crops=4)


def _tiny_backbone(spec: Any, device: str = "auto") -> Any:
    model = timm.create_model(
        _TINY_MODEL, pretrained=False, num_classes=0, img_size=spec.input_size
    )
    model.eval()
    model.requires_grad_(False)
    return model.to("cpu")


@pytest.fixture
def cpu_tiny_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """No weight download, and no CUDA -- so ``device="auto"`` means the CPU."""
    monkeypatch.setattr(backbones, "load_backbone", _tiny_backbone)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _forensic_image(size: tuple[int, int] = _PREDICT_SIZE, seed: int = 3) -> ForensicImage:
    buffer = io.BytesIO()
    natural_like_image(size=size, seed=seed).save(buffer, format="PNG")
    return ForensicImage.from_bytes(buffer.getvalue())


@pytest.fixture
def checkpoint(tmp_path: Path, cpu_tiny_backbone: None) -> Path:
    """A real checkpoint: a head trained for three epochs on the tiny backbone."""
    root = tmp_path / "images"
    for index in range(8):
        folder = root / ("fake" if index % 2 else "real")
        folder.mkdir(parents=True, exist_ok=True)
        natural_like_image(size=_TRAIN_SIZE, seed=200 + index).save(
            folder / f"{index}.png", format="PNG"
        )
    manifest, _ = build_manifest(
        root,
        dataset="tiny",
        label_of=label_from_parent_folder,
        license="MIT",
        commercial_ok=True,
        progress=False,
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest.save(manifest_path)

    config = TrainConfig(
        train_manifest=manifest_path,
        val_manifest=manifest_path,
        cache_dir=tmp_path / "features",
        crop=_POLICY,
        head=HeadOptions(proj_dim=16, hidden_dim=16, dropout=0.0),
        epochs=3,
        batch_size=8,
        out_dir=tmp_path / "head",
        device="cpu",
    )
    report = train_head(config, progress=False)
    assert report.meta.head.dim == _TINY_DIM
    return Path(report.out_dir)


def test_registry_exposes_the_learned_detector_when_torch_is_installed() -> None:
    assert "dinov2_head" in registry.available()
    assert registry.get("dinov2_head") is LearnedDetector


def test_predict_returns_a_probability_per_crop_details_and_a_heatmap(checkpoint: Path) -> None:
    detector = LearnedDetector(checkpoint_dir=checkpoint)
    detector.load("cpu")
    assert detector.is_loaded

    image = _forensic_image()
    result = detector.predict(image)

    assert result.detector == "dinov2_head"
    assert 0.0 <= result.score <= 1.0
    assert result.label in {"real", "fake", "uncertain"}
    assert result.details["backbone"] == "dinov2_vitb14"
    assert result.details["n_crops"] == 4
    assert len(result.details["per_crop"]) == 4
    assert all(0.0 <= value <= 1.0 for value in result.details["per_crop"])
    assert result.details["temperature"] > 0.0
    assert checkpoint.name in result.details["checkpoint"]

    # The score is the mean of the per-crop probabilities (up to the rounding
    # applied to the reported ones).
    assert result.score == pytest.approx(float(np.mean(result.details["per_crop"])), abs=1e-3)


def test_the_heatmap_fills_the_crop_regions_and_leaves_the_rest_at_zero(
    checkpoint: Path,
) -> None:
    detector = LearnedDetector(checkpoint_dir=checkpoint)
    detector.load("cpu")

    image = _forensic_image()
    result = detector.predict(image)

    heatmap = result.heatmap
    assert heatmap is not None
    assert heatmap.shape == (image.height, image.width)
    assert heatmap.dtype == np.float32

    # Four non-overlapping 224 tiles cover [0:448, 0:448]; the trailing strip
    # no crop reached stays at zero.
    covered = heatmap[:448, :448]
    assert np.all(covered > 0.0)
    assert np.all(heatmap[448:, :] == 0.0)
    assert np.all(heatmap[:, 448:] == 0.0)
    tiles = [
        heatmap[rows, columns]
        for rows in (slice(0, 224), slice(224, 448))
        for columns in (slice(0, 224), slice(224, 448))
    ]
    for tile in tiles:
        # One crop, one constant region -- these four tiles do not overlap.
        assert float(tile.min()) == pytest.approx(float(tile.max()))
    assert sorted(float(tile.max()) for tile in tiles) == pytest.approx(
        sorted(result.details["per_crop"]), abs=1e-3
    )


def test_random_mode_averages_overlapping_crops(checkpoint: Path, tmp_path: Path) -> None:
    """A checkpoint whose crop policy overlaps still yields a valid heatmap.

    Re-uses the trained weights under a rewritten crop policy: only the
    geometry changes, which is exactly the code path under test.
    """
    overlapping = tmp_path / "head-random"
    overlapping.mkdir()
    document = json.loads((checkpoint / "head.json").read_text(encoding="utf-8"))
    document["crop"] = {**document["crop"], "mode": "random", "max_crops": 6}
    (overlapping / "head.json").write_text(json.dumps(document), encoding="utf-8")
    (overlapping / "head.safetensors").write_bytes((checkpoint / "head.safetensors").read_bytes())

    detector = LearnedDetector(checkpoint_dir=overlapping)
    detector.load("cpu")
    result = detector.predict(_forensic_image())

    assert result.details["n_crops"] == 6
    heatmap = result.heatmap
    assert heatmap is not None
    assert heatmap.shape == _PREDICT_SIZE
    assert 0.0 <= float(heatmap.min()) <= float(heatmap.max()) <= 1.0
    assert float(heatmap.max()) > 0.0


def _fixed_maps(grid: int = 4) -> Any:
    """A stand-in for ``grad_cam``: one all-ones map per crop, on a tiny grid.

    The real Grad-CAM pass has its own tests
    (``tests/test_detectors_attribution.py``); what these tests are about is
    the wiring around it -- that the per-crop maps reach the result, the
    details describe them, and a failure stays contained.
    """

    def fake(model: Any, spec: Any, head: Any, batch: Any, calibration: Any) -> np.ndarray:
        return np.ones((int(batch.shape[0]), grid, grid), dtype=np.float32)

    return fake


def test_predict_attaches_the_attribution_map_and_describes_it(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(attribution_module, "grad_cam", _fixed_maps())
    detector = LearnedDetector(checkpoint_dir=checkpoint)
    detector.load("cpu")

    image = _forensic_image()
    result = detector.predict(image)

    attribution = result.attribution
    assert attribution is not None
    assert attribution.shape == (image.height, image.width)
    # The same four 224 tiles the heatmap covers, and nothing outside them.
    assert np.allclose(attribution[:448, :448], 1.0)
    assert np.all(attribution[448:, :] == 0.0)

    details = result.details["attribution"]
    assert details["method"] == "grad-cam"
    assert details["grid"] == 4
    # DINOv2's spec selects blocks 8-11, each explained by the block feeding
    # it; the pooled row shares block 10 with the last selected layer.
    assert details["target_blocks"] == [7, 8, 9, 10]


def test_the_attribution_env_var_switches_the_map_off(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(HEAD_ATTRIBUTION_ENV, "0")
    detector = LearnedDetector(checkpoint_dir=checkpoint)
    detector.load("cpu")
    assert detector.attribution is False

    result = detector.predict(_forensic_image())

    assert result.attribution is None
    assert "attribution" not in result.details
    assert result.heatmap is not None  # the heatmap is a separate thing


def test_a_failing_attribution_pass_never_fails_the_prediction(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*_args: Any, **_kwargs: Any) -> np.ndarray:
        raise RuntimeError("no blocks to hook")

    monkeypatch.setattr(attribution_module, "grad_cam", explode)
    detector = LearnedDetector(checkpoint_dir=checkpoint)
    detector.load("cpu")

    result = detector.predict(_forensic_image())

    assert result.attribution is None
    assert 0.0 <= result.score <= 1.0
    assert result.heatmap is not None
    assert "no blocks to hook" in result.details["attribution"]["error"]


def test_without_a_checkpoint_the_detector_abstains_with_a_reason(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere"
    detector = LearnedDetector(checkpoint_dir=missing)
    detector.load("cpu")

    result = detector.predict(_forensic_image(size=(64, 64)))

    assert detector.is_loaded is False
    assert result.score == 0.5
    assert result.label == "uncertain"
    assert result.heatmap is None
    reason = result.details["reason"]
    assert str(missing) in reason
    assert "imgforensics train head --config configs/head_dinov2.yaml" in reason


def test_predict_loads_lazily_when_load_was_never_called(tmp_path: Path) -> None:
    result = LearnedDetector(checkpoint_dir=tmp_path / "nowhere").predict(
        _forensic_image(size=(64, 64))
    )

    assert result.score == 0.5
    assert "no trained head found" in result.details["reason"]


def test_checkpoint_directory_resolution_prefers_argument_then_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(HEAD_DIR_ENV, raising=False)
    assert resolve_checkpoint_dir() == Path("weights/dinov2_head")

    monkeypatch.setenv(HEAD_DIR_ENV, str(tmp_path / "from-env"))
    assert resolve_checkpoint_dir() == tmp_path / "from-env"
    assert resolve_checkpoint_dir(tmp_path / "explicit") == tmp_path / "explicit"


def test_cli_analyze_runs_the_learned_detector_from_the_env_var(
    checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(HEAD_DIR_ENV, str(checkpoint))
    image_path = tmp_path / "sample.png"
    natural_like_image(size=_PREDICT_SIZE, seed=42).save(image_path, format="PNG")
    heatmap_dir = tmp_path / "heatmaps"

    result = runner.invoke(
        app,
        [
            "analyze",
            str(image_path),
            "--json",
            "--detector",
            "dinov2_head",
            "--save-heatmaps",
            str(heatmap_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout
    document = json.loads(result.stdout)
    assert len(document["results"]) == 1
    entry = document["results"][0]
    assert entry["detector"] == "dinov2_head"
    assert 0.0 <= entry["score"] <= 1.0
    assert entry["details"]["n_crops"] == 4
    assert entry["heatmap"] == str(heatmap_dir / "sample_dinov2_head.png")
    assert (heatmap_dir / "sample_dinov2_head.png").is_file()
    # The Grad-CAM pass runs for real here, on the tiny backbone.
    attribution_png = heatmap_dir / "sample_dinov2_head_attribution.png"
    assert entry["attribution"] == str(attribution_png)
    assert attribution_png.is_file()
    assert entry["details"]["attribution"]["method"] == "grad-cam"


def test_cli_analyze_reports_the_abstention_when_no_head_is_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(HEAD_DIR_ENV, str(tmp_path / "nowhere"))
    image_path = tmp_path / "sample.png"
    natural_like_image(size=(64, 64), seed=1).save(image_path, format="PNG")

    result = runner.invoke(app, ["analyze", str(image_path), "--json", "--detector", "dinov2_head"])

    assert result.exit_code == 0, result.stdout
    entry = json.loads(result.stdout)["results"][0]
    assert entry["score"] == 0.5
    assert entry["label"] == "uncertain"
    assert "no trained head found" in entry["details"]["reason"]
