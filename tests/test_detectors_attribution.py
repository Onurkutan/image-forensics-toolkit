"""Tests for the Grad-CAM attribution map (optional ``ml`` extra).

Offline and CPU-only, like the other learned-detector tests: the backbone is a
randomly initialised ``vit_tiny_patch16_224`` (no download, 192-dimensional
features, a 14x14 patch grid at 224 px) carrying a head with random weights,
and CUDA is hidden so a GPU on the machine is never touched. A random backbone
cannot produce a *meaningful* saliency map, which is fine -- what is under test
is the plumbing: the grid the map lands on, its normalization, its
determinism, which blocks it is taken from, and how the per-crop maps are
painted back onto the image.

:func:`~imgforensics.detectors.attribution.target_blocks` and
:func:`~imgforensics.detectors.attribution.stitch_attribution` are pure numpy
and are exercised directly, without a model.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from imgforensics.core.types import DetectionResult  # noqa: E402
from imgforensics.detectors.attribution import (  # noqa: E402
    _crop_offset,
    _upsampled,
    grad_cam,
    stitch_attribution,
    target_blocks,
)
from imgforensics.detectors.backbones import BackboneSpec  # noqa: E402
from imgforensics.detectors.crops import CropBox, CropPolicy, crop_boxes  # noqa: E402
from imgforensics.detectors.head import Calibration, HeadOptions, MultiLayerHead  # noqa: E402

pytestmark = pytest.mark.ml

_TINY_MODEL = "vit_tiny_patch16_224"
_TINY_DIM = 192
_INPUT_SIZE = 224
_TINY_GRID = 14  # 224 / 16
_LAYERS = [-2, -1]


@pytest.fixture
def no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide CUDA, so a GPU on this machine is never pulled into these tests."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def tiny_stack(no_cuda: None) -> tuple[torch.nn.Module, BackboneSpec, MultiLayerHead]:
    """A random tiny ViT, its spec, and a random head reading two blocks plus the pooled row."""
    torch.manual_seed(11)
    model = timm.create_model(_TINY_MODEL, pretrained=False, num_classes=0)
    model.eval()
    model.requires_grad_(False)

    spec = BackboneSpec(
        name="tiny", timm_id=_TINY_MODEL, input_size=_INPUT_SIZE, layers=list(_LAYERS)
    )
    options = HeadOptions(proj_dim=16, hidden_dim=16, dropout=0.0)
    head = MultiLayerHead(options.with_shape(len(_LAYERS) + 1, _TINY_DIM))
    head.eval()
    head.requires_grad_(False)
    return model, spec, head


def _batch(seed: int, n: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, _INPUT_SIZE, _INPUT_SIZE, generator=generator)


# --------------------------------------------------------------------------
# grad_cam
# --------------------------------------------------------------------------


def test_grad_cam_returns_one_normalized_patch_grid_map_per_crop(
    tiny_stack: tuple[torch.nn.Module, BackboneSpec, MultiLayerHead],
) -> None:
    model, spec, head = tiny_stack
    maps = grad_cam(model, spec, head, _batch(1), Calibration(temperature=1.5, bias=0.25))

    assert maps.shape == (2, _TINY_GRID, _TINY_GRID)
    assert maps.dtype == np.float32
    assert 0.0 <= float(maps.min()) <= float(maps.max()) <= 1.0
    # A map that is everywhere zero carries no explanation; at least one crop
    # has to have something to show.
    assert float(maps.max()) > 0.0


def test_grad_cam_is_deterministic_for_the_same_input(
    tiny_stack: tuple[torch.nn.Module, BackboneSpec, MultiLayerHead],
) -> None:
    model, spec, head = tiny_stack
    batch = _batch(2)
    first = grad_cam(model, spec, head, batch, Calibration())
    second = grad_cam(model, spec, head, batch, Calibration())

    assert np.array_equal(first, second)


def test_grad_cam_differs_between_two_different_inputs(
    tiny_stack: tuple[torch.nn.Module, BackboneSpec, MultiLayerHead],
) -> None:
    model, spec, head = tiny_stack
    first = grad_cam(model, spec, head, _batch(3), Calibration())
    second = grad_cam(model, spec, head, _batch(4), Calibration())

    assert not np.allclose(first, second)


def test_grad_cam_leaves_the_backbone_free_of_gradients(
    tiny_stack: tuple[torch.nn.Module, BackboneSpec, MultiLayerHead],
) -> None:
    """The backward pass reaches the input, never the frozen parameters."""
    model, spec, head = tiny_stack
    grad_cam(model, spec, head, _batch(5), Calibration())

    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in head.parameters())


def test_grad_cam_on_a_backbone_without_blocks_raises(
    tiny_stack: tuple[torch.nn.Module, BackboneSpec, MultiLayerHead],
) -> None:
    _, spec, head = tiny_stack
    with pytest.raises(ValueError, match="blocks"):
        grad_cam(torch.nn.Linear(4, 4), spec, head, _batch(6), Calibration())


# --------------------------------------------------------------------------
# target_blocks
# --------------------------------------------------------------------------


def test_target_blocks_shifts_each_row_to_the_block_that_feeds_it() -> None:
    # The DINOv2 ViT-B/14 shape: four selected blocks plus the pooled row.
    blocks = target_blocks([8, 9, 10, 11], 12, [0.1, 0.2, 0.3, 0.2, 0.2])

    # Block 11's CLS token is explained by the tokens leaving block 10, and the
    # pooled row -- read after block 11 -- by the same ones.
    assert blocks == pytest.approx({7: 0.1, 8: 0.2, 9: 0.3, 10: 0.4})
    assert sum(blocks.values()) == pytest.approx(1.0)


def test_target_blocks_resolves_negative_layer_indices() -> None:
    assert target_blocks([-2, -1], 12, [0.2, 0.3, 0.5]) == pytest.approx({9: 0.2, 10: 0.8})


def test_target_blocks_sums_the_weights_of_two_rows_sharing_one_block() -> None:
    # The last selected layer and the pooled row both resolve to block 10.
    blocks = target_blocks([11], 12, [0.25, 0.75])

    assert set(blocks) == {10}
    assert blocks[10] == pytest.approx(1.0)


def test_target_blocks_skips_a_row_that_no_block_feeds_and_renormalizes() -> None:
    # Block 0's input is the patch embedding, not another block's output, so
    # that row has nothing to hook and its weight is redistributed.
    blocks = target_blocks([0, 5], 12, [0.5, 0.2, 0.3])

    assert set(blocks) == {4, 10}
    assert blocks[4] == pytest.approx(0.4)
    assert blocks[10] == pytest.approx(0.6)


def test_target_blocks_splits_evenly_when_every_weight_is_zero() -> None:
    even = 1.0 / 3.0
    assert target_blocks([9, 10], 12, [0.0, 0.0, 0.0]) == pytest.approx(
        {8: even, 9: even, 10: even}
    )


def test_target_blocks_rejects_a_wrong_number_of_weights() -> None:
    with pytest.raises(ValueError, match="3 layer weights"):
        target_blocks([9, 10], 12, [0.5, 0.5])


def test_target_blocks_rejects_a_layer_index_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        target_blocks([12], 12, [0.5, 0.5])


# --------------------------------------------------------------------------
# stitch_attribution
# --------------------------------------------------------------------------


def _ramp(grid: int = 4) -> np.ndarray:
    """A deterministic ``(grid, grid)`` map rising from 0 to 1 top-left to bottom-right."""
    return np.linspace(0.0, 1.0, grid * grid, dtype=np.float32).reshape(grid, grid)


def test_stitch_attribution_paints_a_box_and_leaves_the_rest_at_zero() -> None:
    boxes = [CropBox(top=4, left=6, height=16, width=16)]
    stitched = stitch_attribution((40, 40), boxes, [_ramp()], 16)

    assert stitched is not None
    assert stitched.shape == (40, 40)
    assert stitched.dtype == np.float32
    assert float(stitched[4:20, 6:22].max()) > 0.0
    assert float(stitched[:4, :].max()) == 0.0
    assert float(stitched[20:, :].max()) == 0.0
    assert float(stitched[:, 22:].max()) == 0.0


def test_stitch_attribution_averages_two_overlapping_boxes() -> None:
    ones = np.ones((4, 4), dtype=np.float32)
    zeros = np.zeros((4, 4), dtype=np.float32)
    boxes = [CropBox(top=0, left=0, height=8, width=8), CropBox(top=0, left=4, height=8, width=8)]
    stitched = stitch_attribution((8, 12), boxes, [ones, zeros], 8)

    assert stitched is not None
    assert float(stitched[0, 0]) == pytest.approx(1.0)  # first box only
    assert float(stitched[0, 5]) == pytest.approx(0.5)  # both boxes, one of them zero
    assert float(stitched[0, 10]) == pytest.approx(0.0)  # second box only


def test_stitch_attribution_skips_an_empty_box() -> None:
    boxes = [CropBox(top=0, left=0, height=0, width=0)]
    stitched = stitch_attribution((8, 8), boxes, [np.ones((4, 4), dtype=np.float32)], 8)

    assert stitched is not None
    assert float(stitched.max()) == 0.0


def test_stitch_attribution_returns_none_when_the_counts_disagree() -> None:
    boxes = [CropBox(top=0, left=0, height=8, width=8)]
    assert stitch_attribution((8, 8), boxes, [], 8) is None


def test_stitch_attribution_takes_the_centered_window_of_an_overhanging_crop() -> None:
    """A crop larger than the image sees reflection padding on all four sides.

    ``crop_boxes`` reports the box in the un-padded image's coordinates, so it
    is smaller than the crop and the map has to be read from the middle of the
    crop -- where the real pixels sat -- rather than from its top-left corner.
    """
    policy = CropPolicy(size=32, mode="center")
    image = Image.new("RGB", (20, 20), color=(10, 20, 30))
    boxes = crop_boxes(image, policy)
    assert boxes == [CropBox(top=0, left=0, height=20, width=20)]

    cam = _ramp()
    stitched = stitch_attribution((20, 20), boxes, [cam], policy.size)

    assert stitched is not None
    expected = _upsampled(cam, policy.size)[6:26, 6:26]  # (32 - 20) // 2 on both axes
    assert np.allclose(stitched, expected)


def test_crop_offset_covers_every_way_a_box_can_be_clipped() -> None:
    # Not clipped at all: the box is the whole crop.
    assert _crop_offset(10, 32, 100, 32) == 0
    # Clipped at the far edge: the surviving pixels are the crop's leading ones.
    assert _crop_offset(80, 20, 100, 32) == 0
    # Clipped at the near edge: the surviving pixels are the crop's trailing ones.
    assert _crop_offset(0, 20, 100, 32) == 12
    # Clipped at both edges (image shorter than the crop): centered padding.
    assert _crop_offset(0, 20, 20, 32) == 6


# --------------------------------------------------------------------------
# DetectionResult.attribution
# --------------------------------------------------------------------------


def test_detection_result_accepts_a_two_dimensional_attribution() -> None:
    result = DetectionResult(
        detector="x", score=0.5, label="uncertain", attribution=np.zeros((4, 4), dtype=np.float32)
    )
    assert result.attribution is not None
    assert result.attribution.shape == (4, 4)


def test_detection_result_rejects_a_three_dimensional_attribution() -> None:
    with pytest.raises(ValueError, match="attribution must be 2-D"):
        DetectionResult(
            detector="x",
            score=0.5,
            label="uncertain",
            attribution=np.zeros((2, 2, 2), dtype=np.float32),
        )


def test_detection_result_rejects_attribution_values_above_one() -> None:
    with pytest.raises(ValueError, match=r"attribution values must be within"):
        DetectionResult(
            detector="x",
            score=0.5,
            label="uncertain",
            attribution=np.full((2, 2), 1.5, dtype=np.float32),
        )
