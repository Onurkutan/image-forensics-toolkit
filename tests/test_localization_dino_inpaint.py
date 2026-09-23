"""Tests for the registered ``dino_inpaint`` localizer (optional ``ml`` extra).

Offline and CPU-only. The real backbone is a frozen DINOv2 ViT-B/14 whose
weights are a 330 MB download, so every test here swaps it for
:class:`_StandInBackbone`: one convolution with the same contract
(``forward_intermediates`` returning one ``(B, 768, G, G)`` map per requested
block). What is under test is everything around it -- the head's shape and
size, the LoRA adapters, the checkpoint round trip, the tiling, the score rule
and the abstention -- none of which depends on the features being real ones.

The tiling arithmetic and the crop sampler have their own tests in
``test_localization_inpaint_stitch.py`` and
``test_localization_inpaint_sampling.py``, which need no ``ml`` extra at all.
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

from imgforensics.cli import app  # noqa: E402
from imgforensics.core import registry  # noqa: E402
from imgforensics.core.image import ForensicImage  # noqa: E402
from imgforensics.localization import _inpaint_model  # noqa: E402
from imgforensics.localization._inpaint_model import (  # noqa: E402
    HEAD_PREFIX,
    LoRALinear,
    PatchHead,
    attach_lora,
    checkpoint_tensors,
    set_lora_training,
)
from imgforensics.localization._scoring import TOP_FRACTION, top_fraction_score  # noqa: E402
from imgforensics.localization.dino_inpaint import (  # noqa: E402
    CROP_SIZE,
    METADATA_FILENAME,
    PATCH,
    WEIGHTS_FILENAME,
    DinoInpaintLocalizer,
    InpaintCheckpointMeta,
    InpaintConfig,
    InpaintValMetrics,
    LoraConfig,
    ManifestRef,
    PatchHeadConfig,
    TrainEcho,
)

pytestmark = pytest.mark.ml

runner = CliRunner()

_DIM = 768
_GRID = CROP_SIZE // PATCH
_HEAD_PARAMETER_BUDGET = 1_300_000


class _StandInBackbone(torch.nn.Module):
    """One patch-embedding convolution with timm's ``forward_intermediates`` contract.

    A stride-14 convolution turns a ``(B, 3, H, W)`` batch into the
    ``(B, 768, H / 14, W / 14)`` patch map a ViT-B block would emit. Each
    requested block gets a differently scaled copy, so a wrapper that asked
    for the wrong number of blocks -- or silently dropped one -- fails loudly
    rather than being averaged away.

    It also carries the one submodule LoRA is attached to, ``attn.qkv``, so
    the stage-2 load path has something to patch. It is not on the forward
    path here -- a test that needs the adapters to receive gradients uses the
    training suite's stand-in instead.
    """

    def __init__(self, dim: int = _DIM, patch: int = PATCH) -> None:
        super().__init__()
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
        return [features * (1.0 + index) for index in indices]


def _lora_backbone() -> _StandInBackbone:
    """A stand-in backbone to attach LoRA to; it carries exactly one ``attn.qkv``."""
    return _StandInBackbone()


def _config(**overrides: Any) -> InpaintConfig:
    return InpaintConfig(layers=[5, 8, 11], **overrides)


def _meta(config: InpaintConfig | None = None, **overrides: Any) -> InpaintCheckpointMeta:
    resolved = config or _config()
    defaults: dict[str, Any] = {
        "package_version": "0.1.0",
        "created": "2026-09-21T00:00:00+00:00",
        "backbone": resolved.backbone,
        "layers": list(resolved.layers),
        "crop_size": resolved.crop_size,
        "stride": resolved.stride,
        "patch": resolved.patch,
        "head": resolved.head_config(),
        "train": TrainEcho(augment_hash="none", crops_per_epoch=64, batch_size=8, lr=3e-4, seed=0),
        "best_epoch": 1,
        "epochs_run": 1,
        "val": InpaintValMetrics(best_f1=0.5, ap=0.4, f1_at_threshold=0.3, iou=0.2),
        "manifests": [
            ManifestRef(
                role="train_fake",
                path="data/manifests/tgif_train_fr.jsonl",
                dataset="TGIF",
                entries=29280,
                sha256="a" * 64,
                license="CC BY-SA 4.0 (COCO-derived)",
                commercial_ok=False,
            )
        ],
        "commercial_ok": False,
        "licenses": ["CC BY-SA 4.0 (COCO-derived)"],
    }
    return InpaintCheckpointMeta(**{**defaults, **overrides})


def _write_checkpoint(
    directory: Path, head: PatchHead, meta: InpaintCheckpointMeta, backbone: Any = None
) -> None:
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    save_file(checkpoint_tensors(head, backbone), str(directory / WEIGHTS_FILENAME))
    (directory / METADATA_FILENAME).write_text(
        json.dumps(meta.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
    )


@pytest.fixture
def stand_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the Hub download for the stand-in and hide CUDA.

    The backbone is replaced at the seam ``load_patch_backbone`` loads through,
    so everything above it -- the spec override, the device resolution, the
    normalization lookup -- runs exactly as it does in production. CUDA is
    hidden so ``device="auto"`` resolves to the CPU even on a machine that has
    a GPU, and so these tests never compete with a training run for it.
    """
    monkeypatch.setattr(
        _inpaint_model,
        "load_backbone",
        lambda spec, device, dynamic_img_size=False: _StandInBackbone(),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def checkpoint_dir(tmp_path: Path, stand_in: None) -> Path:
    """A checkpoint directory holding a randomly initialised head and its metadata."""
    config = _config()
    torch.manual_seed(0)
    _write_checkpoint(tmp_path / "dino_inpaint", PatchHead(config.head_config()), _meta(config))
    return tmp_path / "dino_inpaint"


def _forensic_image(size: tuple[int, int], seed: int = 5) -> ForensicImage:
    buffer = io.BytesIO()
    natural_like_image(size=size, seed=seed).save(buffer, format="PNG")
    return ForensicImage.from_bytes(buffer.getvalue())


def _loaded(checkpoint_dir: Path) -> DinoInpaintLocalizer:
    localizer = DinoInpaintLocalizer(checkpoint_dir=checkpoint_dir)
    localizer.load("cpu")
    assert localizer.is_loaded
    return localizer


# --- PatchHead ---------------------------------------------------------------


def test_the_head_maps_a_concatenated_patch_grid_to_one_logit_per_patch() -> None:
    head = PatchHead(PatchHeadConfig(n_layers=3))

    logits = head(torch.randn(2, 3 * _DIM, _GRID, _GRID))

    assert logits.shape == (2, 1, _GRID, _GRID)
    assert sum(parameter.numel() for parameter in head.parameters()) < _HEAD_PARAMETER_BUDGET


def test_the_head_is_fully_convolutional_so_any_grid_works() -> None:
    head = PatchHead(PatchHeadConfig(n_layers=3))

    assert head(torch.randn(1, 3 * _DIM, 19, 19)).shape == (1, 1, 19, 19)


def test_context_kernel_one_drops_the_neighbourhood_convolution() -> None:
    with_context = PatchHead(PatchHeadConfig(n_layers=3))
    without_context = PatchHead(PatchHeadConfig(n_layers=3, context_kernel=1))

    assert with_context.context is not None
    assert without_context.context is None
    assert without_context(torch.randn(1, 3 * _DIM, _GRID, _GRID)).shape == (1, 1, _GRID, _GRID)
    # The 3x3 is the bulk of the head; dropping it halves it.
    assert sum(p.numel() for p in without_context.parameters()) < 0.55 * sum(
        p.numel() for p in with_context.parameters()
    )


def test_the_head_rejects_a_feature_map_of_the_wrong_width() -> None:
    head = PatchHead(PatchHeadConfig(n_layers=3))

    with pytest.raises(ValueError, match="expected features of shape"):
        head(torch.randn(1, 2 * _DIM, _GRID, _GRID))


# --- LoRA --------------------------------------------------------------------


def test_a_fresh_lora_layer_is_exactly_the_layer_it_wraps() -> None:
    base = torch.nn.Linear(16, 32)
    inputs = torch.randn(4, 16)
    expected = base(inputs)

    wrapped = LoRALinear(base, rank=8, alpha=16.0, dropout=0.05)

    assert torch.allclose(wrapped(inputs), expected, atol=1e-6)


def test_only_the_lora_factors_of_a_wrapped_layer_train() -> None:
    wrapped = LoRALinear(torch.nn.Linear(16, 32), rank=8, alpha=16.0)

    trainable = {name for name, p in wrapped.named_parameters() if p.requires_grad}

    assert trainable == {"lora_a", "lora_b"}


def test_attach_lora_patches_every_block_and_freezes_everything_else() -> None:
    class _Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv = torch.nn.Linear(_DIM, 3 * _DIM)

    class _Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = _Attention()

    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([_Block() for _ in range(12)])  # type: ignore[assignment]

    modules = attach_lora(model, LoraConfig())

    assert len(modules) == 12
    assert modules[0] == "blocks.0.attn.qkv"
    trainable = [p for p in model.parameters() if p.requires_grad]
    # 12 blocks x (8 x 768 + 2304 x 8) trainable numbers, and nothing else.
    assert sum(p.numel() for p in trainable) == 294_912
    adapters = [module for module in model.modules() if isinstance(module, LoRALinear)]
    assert adapters and all(not module.base.weight.requires_grad for module in adapters)


def test_attach_lora_leaves_its_factors_where_the_layer_it_wraps_lives() -> None:
    """The adapters must end up on the model's device, not on the CPU by default.

    :func:`attach_lora` builds ``lora_a``/``lora_b`` wherever the process is,
    so attaching to a backbone that has already been moved leaves the factors
    behind and the first forward pass dies on a device mismatch. Both callers
    therefore ``.to(device)`` the backbone again afterwards; this pins the
    invariant they are restoring.
    """
    model = torch.nn.Module()
    model.attn = torch.nn.Module()  # type: ignore[assignment]
    model.attn.qkv = torch.nn.Linear(_DIM, 3 * _DIM)  # type: ignore[assignment]
    device = next(model.parameters()).device

    attach_lora(model, LoraConfig())
    model.to(device)

    adapters = [module for module in model.modules() if isinstance(module, LoRALinear)]
    assert len(adapters) == 1
    for adapter in adapters:
        assert adapter.lora_a.device == adapter.base.weight.device
        assert adapter.lora_b.device == adapter.base.weight.device
    # And the forward pass the mismatch would have broken still runs.
    assert model.attn.qkv(torch.randn(2, _DIM)).shape == (2, 3 * _DIM)


def test_the_load_path_moves_the_adapters_onto_the_resolved_device(
    tmp_path: Path, stand_in: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``load_checkpoint`` must move the backbone again after attaching LoRA.

    Asserted by watching the call, because the bug it guards against only
    shows itself on a machine with a second device: on a CPU-only run every
    tensor is on the CPU whether or not the move happened.
    """
    moved: list[Any] = []
    original = torch.nn.Module.to

    def _recording_to(self: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(self, _StandInBackbone):
            moved.append(args[0] if args else kwargs.get("device"))
        return original(self, *args, **kwargs)

    config = _config()
    torch.manual_seed(3)
    head = PatchHead(config.head_config())
    backbone = _lora_backbone()
    attach_lora(backbone, LoraConfig())
    _write_checkpoint(
        tmp_path / "run",
        head,
        _meta(config, lora=LoraConfig(modules=["attn.qkv"])),
        backbone=backbone,
    )

    monkeypatch.setattr(torch.nn.Module, "to", _recording_to)
    localizer = _loaded(tmp_path / "run")

    assert moved, "the backbone was never moved onto the resolved device"
    assert str(moved[-1]) == localizer.device
    adapters = [
        module for module in localizer._backbone.modules() if isinstance(module, LoRALinear)
    ]
    assert adapters
    for adapter in adapters:
        assert adapter.lora_a.device == adapter.base.weight.device


def test_set_lora_training_flips_the_adapters_and_leaves_the_backbone_in_eval() -> None:
    backbone = _lora_backbone()
    attach_lora(backbone, LoraConfig())
    backbone.eval()  # what a training run does to keep the frozen half frozen

    switched = set_lora_training(backbone, True)

    assert switched == 1
    adapter = next(m for m in backbone.modules() if isinstance(m, LoRALinear))
    assert adapter.training
    # The dropout the checkpoint records only fires in train mode, which is
    # the whole point of switching them back.
    assert adapter.dropout.training
    assert not backbone.body.training

    assert set_lora_training(backbone, False) == 1
    assert not adapter.dropout.training


def test_attach_lora_refuses_a_model_with_no_target() -> None:
    with pytest.raises(ValueError, match="no 'attn.qkv' submodule"):
        attach_lora(torch.nn.Linear(4, 4), LoraConfig())


# --- the checkpoint ----------------------------------------------------------


def test_a_checkpoint_round_trips_through_the_localizer(tmp_path: Path, stand_in: None) -> None:
    config = _config()
    torch.manual_seed(1)
    head = PatchHead(config.head_config())
    meta = _meta(config)
    _write_checkpoint(tmp_path / "run", head, meta)

    localizer = _loaded(tmp_path / "run")

    assert localizer.meta == meta
    assert localizer.meta is not None and localizer.meta.commercial_ok is False
    assert localizer.meta.licenses == ["CC BY-SA 4.0 (COCO-derived)"]
    loaded_state = localizer._head.state_dict()  # type: ignore[union-attr]
    for name, value in head.state_dict().items():
        assert torch.equal(loaded_state[name], value), name


def test_a_checkpoint_holds_the_head_and_the_lora_deltas_and_no_base_weights() -> None:
    torch.manual_seed(2)
    head = PatchHead(PatchHeadConfig(n_layers=3))
    backbone = _lora_backbone()
    attach_lora(backbone, LoraConfig())

    tensors = checkpoint_tensors(head, backbone)

    assert {name for name in tensors if name.startswith(HEAD_PREFIX)}
    assert sorted(name for name in tensors if name.startswith("lora.")) == [
        "lora.attn.qkv.lora_a",
        "lora.attn.qkv.lora_b",
    ]
    assert not [name for name in tensors if "base" in name]


# --- the localizer -----------------------------------------------------------


def test_registry_exposes_the_localizer_when_torch_is_installed() -> None:
    assert "dino_inpaint" in registry.available()
    assert registry.get("dino_inpaint") is DinoInpaintLocalizer


def test_without_a_checkpoint_the_localizer_abstains_with_a_reason(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere"
    localizer = DinoInpaintLocalizer(checkpoint_dir=missing)
    localizer.load("cpu")

    result = localizer.predict(_forensic_image((64, 64)))

    assert localizer.is_loaded is False
    assert result.score == 0.5
    assert result.label == "uncertain"
    assert result.heatmap is None
    reason = result.details["reason"]
    assert str(missing) in reason
    assert "imgforensics train localizer" in reason


def test_predict_loads_lazily_when_load_was_never_called(tmp_path: Path) -> None:
    result = DinoInpaintLocalizer(checkpoint_dir=tmp_path / "nowhere").predict(
        _forensic_image((64, 64))
    )

    assert result.score == 0.5
    assert "no trained inpainting localizer" in result.details["reason"]


def test_a_tiled_image_gets_a_heatmap_of_its_own_shape(checkpoint_dir: Path) -> None:
    image = _forensic_image((600, 512))  # (width, height)
    result = _loaded(checkpoint_dir).predict(image)

    heatmap = result.heatmap
    assert heatmap is not None
    # 512 px -> origins [0, 64]; 600 px -> origins [0, 152]; 2 x 2 tiles.
    assert result.details["tiles"] == 4
    assert heatmap.shape == (image.height, image.width) == (512, 600)
    assert heatmap.dtype == np.float32
    assert 0.0 <= float(heatmap.min()) <= float(heatmap.max()) <= 1.0


def test_an_image_smaller_than_a_tile_is_padded_and_cropped_back(checkpoint_dir: Path) -> None:
    result = _loaded(checkpoint_dir).predict(_forensic_image((256, 256), seed=7))

    heatmap = result.heatmap
    assert heatmap is not None
    assert result.details["tiles"] == 1
    assert heatmap.shape == (256, 256)


def test_the_score_is_the_shared_top_fraction_rule(checkpoint_dir: Path) -> None:
    result = _loaded(checkpoint_dir).predict(_forensic_image((512, 512)))

    heatmap = result.heatmap
    assert heatmap is not None
    assert result.score == pytest.approx(top_fraction_score(heatmap, TOP_FRACTION), abs=1e-6)
    assert result.label in {"real", "fake", "uncertain"}


def test_details_report_the_checkpoint_and_the_heatmap_summary(checkpoint_dir: Path) -> None:
    result = _loaded(checkpoint_dir).predict(_forensic_image((512, 512)))

    heatmap = result.heatmap
    assert heatmap is not None
    details = result.details
    assert details["weights"].startswith(WEIGHTS_FILENAME)
    assert "(" in details["weights"]  # the sha256 prefix of the file actually loaded
    assert str(checkpoint_dir) in details["checkpoint"]
    assert details["backbone"] == "dinov2_vitb14"
    assert details["layers"] == [5, 8, 11]
    assert details["crop_size"] == CROP_SIZE
    assert details["lora_rank"] is None
    assert details["max_prob"] == pytest.approx(float(heatmap.max()), abs=1e-4)
    assert details["mean_prob"] == pytest.approx(float(heatmap.mean()), abs=1e-4)
    assert details["area_fraction_above_0.5"] == pytest.approx(
        float((heatmap > 0.5).mean()), abs=1e-4
    )
    assert details["device"] == "cpu"


def test_cli_analyze_runs_the_localizer_from_the_env_var(
    checkpoint_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMGFORENSICS_INPAINT_DIR", str(checkpoint_dir))
    image_path = tmp_path / "sample.png"
    natural_like_image(size=(300, 260), seed=11).save(image_path, format="PNG")

    result = runner.invoke(
        app, ["analyze", str(image_path), "--json", "--detector", "dino_inpaint"]
    )

    assert result.exit_code == 0, result.stdout
    entry: dict[str, Any] = json.loads(result.stdout)["results"][0]
    assert entry["detector"] == "dino_inpaint"
    assert 0.0 <= entry["score"] <= 1.0
    assert entry["details"]["tiles"] == 1
