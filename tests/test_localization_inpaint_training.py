"""Tests for the ``dino_inpaint`` training loop (optional ``ml`` extra, CPU only).

One tiny epoch of the real :func:`~imgforensics.localization.train_inpaint.train_inpaint`
over a handful of synthetic 84 px images, with the backbone swapped for a
stand-in and the model shrunk to a 32-channel, two-block, 56 px configuration.
That keeps a whole run -- crop sampling, augmentation, the loss, the gradient
scaler, validation through the inference path, the checkpoint and the log --
inside a couple of CPU seconds, so the code path the GPU runs is covered by
something that runs in CI.

The scaler is disabled on the CPU (there is no float16 autocast to protect
against), which is itself part of the contract: ``amp_scale`` is 1.0 in the
log rather than missing, so the field means the same thing on both devices.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from imgforensics.core.image import ForensicImage  # noqa: E402
from imgforensics.data.manifest import Manifest, ManifestEntry, ManifestMeta  # noqa: E402
from imgforensics.localization import _inpaint_model  # noqa: E402
from imgforensics.localization import train_inpaint as training  # noqa: E402
from imgforensics.localization._inpaint_model import LoRALinear, PatchHead  # noqa: E402
from imgforensics.localization.dino_inpaint import (  # noqa: E402
    LOG_FILENAME,
    METADATA_FILENAME,
    WEIGHTS_FILENAME,
    DinoInpaintLocalizer,
    InpaintConfig,
    LoraConfig,
    PatchHeadConfig,
)
from imgforensics.localization.train_inpaint import (  # noqa: E402
    CropDataset,
    CropSample,
    InpaintTrainConfig,
    inpaint_loss,
    train_inpaint,
)

pytestmark = pytest.mark.ml

_DIM = 32
_PATCH = 14
_CROP = 56
_SIDE = 84


class _StandInBackbone(torch.nn.Module):
    """A patch-embedding convolution and one ``attn.qkv`` projection, nothing else.

    Small enough to train on a CPU, and shaped like the real thing where it
    matters: ``forward_intermediates`` returns one ``(B, dim, G, G)`` map per
    requested block, and the ``attn.qkv`` linear layer sits *on the path* to
    that output, so a LoRA adapter attached to it actually receives gradients
    instead of quietly staying at its zero-initialised value.
    """

    def __init__(self, dim: int = _DIM, patch: int = _PATCH) -> None:
        super().__init__()
        self.dim = dim
        self.body = torch.nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.attn = torch.nn.Module()
        self.attn.qkv = torch.nn.Linear(dim, 3 * dim)  # type: ignore[assignment]

    def forward_intermediates(
        self,
        x: torch.Tensor,
        indices: list[int],
        norm: bool = True,
        output_fmt: str = "NCHW",
        intermediates_only: bool = True,
    ) -> list[torch.Tensor]:
        assert output_fmt == "NCHW" and intermediates_only and norm
        features = self.body(x)
        tokens = features.flatten(2).transpose(1, 2)
        mixed = self.attn.qkv(tokens)[..., : self.dim]
        features = mixed.transpose(1, 2).reshape(features.shape)
        return [features * (1.0 + index) for index in indices]


class _RecordingHead(PatchHead):
    """A :class:`PatchHead` that remembers what it looked like before training."""

    initial: dict[str, torch.Tensor] = {}

    def __init__(self, config: PatchHeadConfig) -> None:
        super().__init__(config)
        type(self).initial = {
            name: value.detach().clone() for name, value in self.state_dict().items()
        }


def _write_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 256, (_SIDE, _SIDE, 3), dtype=np.uint8), mode="RGB").save(
        path, format="PNG"
    )


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((_SIDE, _SIDE), dtype=np.uint8)
    mask[14:56, 14:56] = 255
    Image.fromarray(mask, mode="L").save(path, format="PNG")


def _entry(path: str, label: str, index: int, **overrides: Any) -> ManifestEntry:
    return ManifestEntry(
        path=path,
        label=label,  # type: ignore[arg-type]
        source="stand-in",
        split="train",
        sha256=f"{index:064d}",
        width=_SIDE,
        height=_SIDE,
        format="PNG",
        **overrides,
    )


def _meta(root: Path) -> ManifestMeta:
    return ManifestMeta(
        dataset="stand-in",
        root=str(root),
        license="CC BY-SA 4.0 (COCO-derived)",
        commercial_ok=False,
        created="2026-09-21",
    )


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Four fakes with masks, four reals, and the three manifests over them."""
    for index in range(4):
        _write_image(tmp_path / "fake" / f"{index}.png", seed=index)
        _write_mask(tmp_path / "masks" / f"{index}.png")
        _write_image(tmp_path / "real" / f"{index}.png", seed=100 + index)

    fakes = [
        _entry(
            f"fake/{index}.png",
            "fake",
            index,
            generator="sd2-fr",
            mask_path=f"masks/{index}.png",
        )
        for index in range(4)
    ]
    reals = [_entry(f"real/{index}.png", "real", 100 + index) for index in range(4)]

    Manifest(meta=_meta(tmp_path), entries=fakes).save(tmp_path / "train_fake.jsonl")
    Manifest(meta=_meta(tmp_path), entries=reals).save(tmp_path / "train_real.jsonl")
    # Validation needs one of each: a fake carries the pixel metrics, a real
    # carries the false-positive area they cannot show.
    Manifest(meta=_meta(tmp_path), entries=[fakes[0], reals[0]]).save(tmp_path / "val.jsonl")

    augment = tmp_path / "augment.yaml"
    augment.write_text("jpeg_quality: [60, 95]\np: 0.5\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def tiny_config(dataset_root: Path) -> InpaintTrainConfig:
    return InpaintTrainConfig(
        train_fake_manifest=dataset_root / "train_fake.jsonl",
        train_real_manifest=dataset_root / "train_real.jsonl",
        val_manifest=dataset_root / "val.jsonl",
        model=InpaintConfig(
            layers=[0, 1],
            crop_size=_CROP,
            stride=42,
            patch=_PATCH,
            head=PatchHeadConfig(dim=_DIM, proj_dim=8),
        ),
        sampling={"real": 0.4, "sd2-fr": 0.6},
        augment=str(dataset_root / "augment.yaml"),
        epochs=1,
        crops_per_epoch=4,
        batch_size=2,
        num_workers=0,
        device="cpu",
        out_dir=dataset_root / "out",
    )


@pytest.fixture
def stand_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _inpaint_model,
        "load_backbone",
        lambda spec, device, dynamic_img_size=False: _StandInBackbone(),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(training, "PatchHead", _RecordingHead)


# --- CropDataset -------------------------------------------------------------

_MASK_BOX = (14, 56)  # top/left and bottom/right of the painted region


@pytest.fixture
def aligned_sample(tmp_path: Path) -> CropDataset:
    """A dataset over one white image whose masked region is painted black.

    That makes crop/mask alignment checkable from the outside: a patch the
    target calls fully inpainted must be fully black in the crop it came
    from, so a one-row offset between the two crops fails the test instead of
    quietly training the head against the wrong pixels.
    """
    low, high = _MASK_BOX
    pixels = np.full((_SIDE, _SIDE, 3), 255, dtype=np.uint8)
    pixels[low:high, low:high] = 0
    Image.fromarray(pixels, mode="RGB").save(tmp_path / "image.png", format="PNG")

    mask = np.zeros((_SIDE, _SIDE), dtype=np.uint8)
    mask[low:high, low:high] = 255
    Image.fromarray(mask, mode="L").save(tmp_path / "mask.png", format="PNG")

    return CropDataset(
        [CropSample("image.png", "mask.png", 0)],
        roots=[str(tmp_path)],
        probabilities=[1.0],
        crop_size=_CROP,
        patch=_PATCH,
        patch_target="soft",
        positive_crop_fraction=1.0,
        flip_probability=0.0,
        augmentation=None,
        # Identity normalization, so the tensor is the image divided by 255
        # and "black" is exactly 0.
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        crops_per_epoch=16,
        seed=0,
    )


def test_a_crop_and_its_patch_target_describe_the_same_pixels(
    aligned_sample: CropDataset,
) -> None:
    grid = _CROP // _PATCH
    seen_full, seen_empty = 0, 0

    for index in range(len(aligned_sample)):
        crop, target = aligned_sample[index]
        assert crop.shape == (3, _CROP, _CROP)
        assert target.shape == (1, grid, grid)

        for row in range(grid):
            for column in range(grid):
                block = crop[
                    :,
                    row * _PATCH : (row + 1) * _PATCH,
                    column * _PATCH : (column + 1) * _PATCH,
                ]
                value = float(target[0, row, column])
                if value == 1.0:
                    assert float(block.max()) == 0.0, (index, row, column)
                    seen_full += 1
                elif value == 0.0:
                    assert float(block.min()) == 1.0, (index, row, column)
                    seen_empty += 1
                else:  # a boundary patch: its black share is its target
                    assert float((block[0] == 0.0).float().mean()) == pytest.approx(value, abs=1e-6)

    # A guided draw always covers the mask, so both kinds really did occur.
    assert seen_full > 0 and seen_empty > 0


def test_a_flip_flips_the_crop_and_its_target_together(aligned_sample: CropDataset) -> None:
    flipped = CropDataset(
        [CropSample("image.png", "mask.png", 0)],
        roots=aligned_sample.roots,
        probabilities=[1.0],
        crop_size=_CROP,
        patch=_PATCH,
        patch_target="soft",
        positive_crop_fraction=1.0,
        flip_probability=1.0,
        augmentation=None,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        crops_per_epoch=16,
        seed=0,
    )

    for index in range(4):
        crop, target = aligned_sample[index]
        crop_flipped, target_flipped = flipped[index]

        # The flip decision costs one draw either way, so the two datasets
        # pick the same image and the same crop box for a given index.
        assert torch.equal(crop_flipped, torch.flip(crop, dims=[2]))
        assert torch.equal(target_flipped, torch.flip(target, dims=[2]))


# --- inpaint_loss ------------------------------------------------------------


def test_the_positive_weight_applies_to_the_inpainted_class_only() -> None:
    logits = torch.zeros(1, 1, 4, 4)
    inpainted = torch.ones(1, 1, 4, 4)
    authentic = torch.zeros(1, 1, 4, 4)
    chance = float(np.log(2.0))

    # sigmoid(0) = 0.5, so every term is log(2) before weighting.
    assert float(inpaint_loss(logits, inpainted, 1.0, 0.0)) == pytest.approx(chance, abs=1e-6)
    assert float(inpaint_loss(logits, inpainted, 3.0, 0.0)) == pytest.approx(3 * chance, abs=1e-6)
    # The weight must not touch the authentic class, or the head would be
    # pushed toward predicting nothing exactly as hard.
    assert float(inpaint_loss(logits, authentic, 3.0, 0.0)) == pytest.approx(chance, abs=1e-6)


def test_the_dice_term_scores_probabilities_against_the_soft_target() -> None:
    logits = torch.tensor([[[[2.0, -1.0], [0.5, -3.0]]]])
    targets = torch.tensor([[[[1.0, 0.0], [0.5, 0.25]]]])

    total = float(inpaint_loss(logits, targets, 2.0, 1.0))
    bce = float(inpaint_loss(logits, targets, 2.0, 0.0))

    probabilities = torch.sigmoid(logits).flatten()
    flat = targets.flatten()
    intersection = float((probabilities * flat).sum())
    union = float(probabilities.sum() + flat.sum())
    expected_dice = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)

    assert total - bce == pytest.approx(expected_dice, abs=1e-6)
    # Half the weight, half the term: the two parts are simply added.
    half = float(inpaint_loss(logits, targets, 2.0, 0.5))
    assert half - bce == pytest.approx(0.5 * expected_dice, abs=1e-6)


def test_a_perfect_prediction_costs_almost_nothing() -> None:
    targets = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    logits = torch.where(targets > 0.5, 20.0, -20.0)

    assert float(inpaint_loss(logits, targets, 3.0, 0.5)) < 1e-3
    # And a confidently wrong one costs a great deal more.
    assert float(inpaint_loss(-logits, targets, 3.0, 0.5)) > 10.0


# --- train_inpaint() ---------------------------------------------------------


def test_one_epoch_trains_the_head_through_the_gradient_scaler(
    tiny_config: InpaintTrainConfig, stand_in: None
) -> None:
    report = train_inpaint(tiny_config, progress=False)

    assert len(report.epochs) == 1
    record = report.epochs[0]
    assert np.isfinite(record.train_loss) and record.train_loss > 0.0
    # The scaler is disabled on the CPU and reports the neutral factor, so the
    # field means the same thing in a CPU log as in a CUDA one.
    assert record.amp_scale == 1.0
    assert report.lora_parameters == 0
    assert report.head_parameters > 0

    # The weights moved: scaler.step() really did step the optimizer.
    trained = PatchHead(tiny_config.model.head_config())
    from safetensors.torch import load_file

    tensors = load_file(str(report.weights_path))
    trained.load_state_dict({name[len("head.") :]: value for name, value in tensors.items()})
    changed = [
        name
        for name, value in trained.state_dict().items()
        if not torch.equal(value, _RecordingHead.initial[name])
    ]
    assert changed, "no head parameter changed in the epoch"


def test_every_step_goes_through_the_scaler_not_around_it(
    tiny_config: InpaintTrainConfig, stand_in: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disabled scaler behaves like no scaler, so the wiring is asserted directly.

    On the CPU ``scaler.step(optimizer)`` and a bare ``optimizer.step()``
    produce identical weights, which means no CPU test can tell a regression
    to the latter apart from the intended code -- and that regression would
    only show up on CUDA, as a run that silently learns nothing. Counting the
    calls is what closes that gap.
    """
    calls: list[str] = []
    real_scaler = torch.amp.GradScaler

    class _CountingScaler(real_scaler):  # type: ignore[misc, valid-type]
        def scale(self, outputs: Any) -> Any:
            calls.append("scale")
            return super().scale(outputs)

        def step(self, optimizer: Any, *args: Any, **kwargs: Any) -> Any:
            calls.append("step")
            return super().step(optimizer, *args, **kwargs)

        def update(self, new_scale: Any = None) -> None:
            calls.append("update")
            super().update(new_scale)

    monkeypatch.setattr(torch.amp, "GradScaler", _CountingScaler)

    train_inpaint(tiny_config, progress=False)

    # Four crops at batch size two: two optimizer steps, each scaled, stepped
    # and followed by an update, in that order.
    assert calls == ["scale", "step", "update"] * 2


def test_each_epoch_persists_its_log_and_an_improving_epoch_its_checkpoint(
    tiny_config: InpaintTrainConfig, stand_in: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that dies at hour three must leave its history and its best model.

    Both files used to be written only after the last epoch. This watches the
    directory from *inside* the second epoch's validation -- the earliest
    point at which epoch 1's writes must already be on disk -- and keeps a
    copy of what it found, so the mid-run checkpoint can be loaded after the
    run like any other.
    """
    config = tiny_config.model_copy(update={"epochs": 2})
    original = training._validation_metrics
    seen: list[dict[str, Any]] = []
    snapshot = Path(config.out_dir).parent / "snapshot"

    def _spy(*args: Any, **kwargs: Any) -> Any:
        if seen:  # the second epoch: epoch 1 has been written by now
            out_dir = Path(config.out_dir)
            log = out_dir / LOG_FILENAME
            seen.append(
                {
                    "log_lines": log.read_text(encoding="utf-8").splitlines()
                    if log.is_file()
                    else [],
                    "weights": (out_dir / WEIGHTS_FILENAME).is_file(),
                    "meta": (out_dir / METADATA_FILENAME).is_file(),
                }
            )
            shutil.copytree(out_dir, snapshot, dirs_exist_ok=True)
        else:
            seen.append({})
        return original(*args, **kwargs)

    monkeypatch.setattr(training, "_validation_metrics", _spy)

    report = train_inpaint(config, progress=False)

    assert len(seen) == 2, "the run did not reach a second epoch"
    mid_run = seen[1]
    # Epoch 1 always improves (the best score starts at -inf), so all three
    # files exist before epoch 2 finishes.
    assert len(mid_run["log_lines"]) == 1
    assert json.loads(mid_run["log_lines"][0])["epoch"] == 1
    assert mid_run["weights"] and mid_run["meta"]

    # ...and the copy taken mid-run is a checkpoint, not a half-written file.
    saved = json.loads((snapshot / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert saved["best_epoch"] == 1
    assert saved["epochs_run"] == 1
    assert saved["val"]["best_f1"] == pytest.approx(report.epochs[0].val_best_f1)

    localizer = DinoInpaintLocalizer(checkpoint_dir=snapshot)
    localizer.load("cpu")
    assert localizer.is_loaded

    probe = snapshot / "probe.png"
    _write_image(probe, seed=77)
    assert localizer.predict(ForensicImage.from_bytes(probe.read_bytes())).heatmap is not None

    # The finished run still reports the whole history and the best epoch.
    assert len(report.epochs) == 2
    assert (Path(config.out_dir) / LOG_FILENAME).read_text(encoding="utf-8").count("\n") == 2
    assert report.meta.epochs_run == 2


def test_the_progress_line_reports_the_whole_epoch_on_one_line(
    tiny_config: InpaintTrainConfig, stand_in: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """The only test that runs with ``progress=True``, which is how a real run runs.

    Every other test here silences the output, so a broken format string in
    the per-epoch print would reach the GPU untested.
    """
    report = train_inpaint(tiny_config, progress=True)

    lines = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("[train] epoch")
    ]

    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("[train] epoch 1/1 ")
    for field in ("loss=", "best_f1=", "ap=", "f1@0.5=", "real_area=", "crops/s", "amp_scale="):
        assert field in line, field
    assert f"{report.epochs[0].val_best_f1:.4f}" in line
    assert line.endswith(" *")  # the first epoch is always an improvement


def test_the_epoch_log_records_the_scaler_factor(
    tiny_config: InpaintTrainConfig, stand_in: None
) -> None:
    report = train_inpaint(tiny_config, progress=False)

    lines = (report.out_dir / LOG_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged["amp_scale"] == 1.0
    assert logged["epoch"] == 1
    assert "val_best_f1" in logged


def test_the_run_writes_a_checkpoint_the_localizer_can_load(
    tiny_config: InpaintTrainConfig, stand_in: None
) -> None:
    report = train_inpaint(tiny_config, progress=False)

    for filename in (WEIGHTS_FILENAME, METADATA_FILENAME, LOG_FILENAME):
        assert (report.out_dir / filename).is_file()
    assert report.meta.commercial_ok is False
    assert report.meta.val.fake_entries == 1
    assert report.meta.val.real_entries == 1

    localizer = DinoInpaintLocalizer(checkpoint_dir=report.out_dir)
    localizer.load("cpu")
    assert localizer.is_loaded

    probe = report.out_dir / "probe.png"
    _write_image(probe, seed=42)
    result = localizer.predict(ForensicImage.from_bytes(probe.read_bytes()))
    assert result.heatmap is not None
    assert result.heatmap.shape == (_SIDE, _SIDE)
    # 84 px at tile 56, stride 42 -> origins [0, 28] on both axes.
    assert result.details["tiles"] == 4
    assert result.details["lora_rank"] is None


def test_validation_splits_on_the_label_and_skips_a_fake_with_no_mask(
    tiny_config: InpaintTrainConfig, dataset_root: Path, stand_in: None
) -> None:
    """A maskless fake is a broken manifest, not an authentic image.

    Reading "real" off a missing mask would move it into the false-positive
    average, where a confidently wrong prediction on a fake would flatter the
    run instead of being reported.
    """
    masked = _entry("fake/0.png", "fake", 0, generator="sd2-fr", mask_path="masks/0.png")
    maskless = _entry("fake/1.png", "fake", 1, generator="sd2-fr")
    real = _entry("real/0.png", "real", 100)
    val_path = dataset_root / "val_broken.jsonl"
    Manifest(meta=_meta(dataset_root), entries=[masked, maskless, real]).save(val_path)

    report = train_inpaint(
        tiny_config.model_copy(update={"val_manifest": val_path}), progress=False
    )

    assert report.val_images == 3
    assert report.meta.val.fake_entries == 1
    assert report.meta.val.real_entries == 1


def test_stage_two_runs_the_adapters_in_train_mode_so_their_dropout_fires(
    tiny_config: InpaintTrainConfig, stand_in: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recorded dropout rate has to be the one that actually ran.

    The backbone is held in ``eval()`` for the whole run so its normalization
    statistics cannot drift, and that reaches the adapters' dropout too --
    which would leave ``inpaint.json`` recording 0.05 for a layer that never
    dropped anything. Validation is the other half of the contract: there the
    adapters *must* be back in eval, or two runs of the same image would
    disagree.
    """
    modes: list[bool] = []

    class _RecordingLoRA(_inpaint_model.LoRALinear):
        def forward(self, x: Any) -> Any:
            modes.append(self.training)
            return super().forward(x)

    # attach_lora instantiates the module-global name, so patching it here is
    # what puts the recording subclass into the backbone.
    monkeypatch.setattr(_inpaint_model, "LoRALinear", _RecordingLoRA)
    config = tiny_config.model_copy(update={"lora": LoraConfig(rank=4, alpha=8.0, dropout=0.05)})

    train_inpaint(config, progress=False)

    assert modes, "the adapters were never reached by a forward pass"
    # Training first (dropout on), then the validation pass (dropout off).
    assert modes[0] is True
    assert modes[-1] is False
    assert any(modes) and not all(modes)


def test_stage_two_steps_the_lora_factors_through_the_same_scaler(
    tiny_config: InpaintTrainConfig, stand_in: None
) -> None:
    config = tiny_config.model_copy(update={"lora": LoraConfig(rank=4, alpha=8.0)})

    report = train_inpaint(config, progress=False)

    assert report.meta.lora is not None
    assert report.meta.lora.modules == ["attn.qkv"]
    assert report.meta.lora.dropout == 0.05
    # rank 4 x (32 in + 96 out) on the one projection the stand-in has.
    assert report.lora_parameters == 4 * _DIM + 3 * _DIM * 4
    assert report.epochs[0].amp_scale == 1.0

    from safetensors.torch import load_file

    tensors = load_file(str(report.weights_path))
    assert sorted(name for name in tensors if name.startswith("lora.")) == [
        "lora.attn.qkv.lora_a",
        "lora.attn.qkv.lora_b",
    ]
    assert not [name for name in tensors if "base" in name]
    # B starts at zero; a non-zero B is the proof the adapters were stepped.
    assert float(tensors["lora.attn.qkv.lora_b"].abs().max()) > 0.0

    localizer = DinoInpaintLocalizer(checkpoint_dir=report.out_dir)
    localizer.load("cpu")
    assert localizer.is_loaded
    adapters = [
        module for module in localizer._backbone.modules() if isinstance(module, LoRALinear)
    ]
    assert len(adapters) == 1
    assert torch.equal(adapters[0].lora_b.detach().cpu(), tensors["lora.attn.qkv.lora_b"])
