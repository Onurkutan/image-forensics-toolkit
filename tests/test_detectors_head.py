"""Tests for the trainable multi-layer head (optional ``ml`` extra).

Pure tensor plumbing: shapes, the parameter budget, and the layer weighting.
No backbone is involved -- the head only ever sees the cached feature arrays,
so these run in milliseconds on the CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from imgforensics.detectors.head import (  # noqa: E402
    Calibration,
    HeadConfig,
    HeadOptions,
    MultiLayerHead,
    aggregate_crops,
)

pytestmark = pytest.mark.ml

# The DINOv2 ViT-B/14 shape: four selected blocks plus the pooled output.
_N_LAYERS = 5
_DIM = 768
_PARAMETER_BUDGET = 1_100_000


def _head(**overrides: object) -> MultiLayerHead:
    config = HeadConfig(n_layers=_N_LAYERS, dim=_DIM, **overrides)  # type: ignore[arg-type]
    return MultiLayerHead(config)


def test_head_stays_within_its_parameter_budget() -> None:
    model = _head()
    total = sum(parameter.numel() for parameter in model.parameters())

    # 5 x (768 -> 256) projections dominate at 984 K; the norms, layer logits
    # and the whole MLP add under 74 K.
    projections = _N_LAYERS * (_DIM * 256 + 256)
    assert total == projections + _N_LAYERS * 2 * _DIM + _N_LAYERS + (256 * 256 + 256) + 257
    assert total < _PARAMETER_BUDGET


def test_forward_maps_crop_features_to_one_logit_per_crop() -> None:
    model = _head()
    features = torch.randn(7, _N_LAYERS, _DIM)

    logits = model(features)

    assert logits.shape == (7,)
    assert logits.dtype == torch.float32


def test_forward_rejects_features_of_the_wrong_shape() -> None:
    model = _head()

    with pytest.raises(ValueError, match="expected features of shape"):
        model(torch.randn(3, _N_LAYERS + 1, _DIM))
    with pytest.raises(ValueError, match="expected features of shape"):
        model(torch.randn(3, _DIM))


def test_softmax_layer_weights_are_positive_and_sum_to_one() -> None:
    model = _head()
    with torch.no_grad():
        model.layer_logits.copy_(torch.tensor([2.0, -1.0, 0.5, 0.0, 3.0]))

    weights = model.layer_weights().detach()

    assert weights.shape == (_N_LAYERS,)
    assert float(weights.sum()) == pytest.approx(1.0)
    assert bool((weights > 0).all())
    # The largest logit must take the largest share.
    assert int(weights.argmax()) == _N_LAYERS - 1


def test_mean_weighting_ignores_the_learned_logits() -> None:
    model = _head(layer_weighting="mean")
    with torch.no_grad():
        model.layer_logits.copy_(torch.tensor([9.0, 0.0, 0.0, 0.0, -9.0]))

    weights = model.layer_weights().detach()

    assert float(weights.sum()) == pytest.approx(1.0)
    assert torch.allclose(weights, torch.full((_N_LAYERS,), 1.0 / _N_LAYERS))


def test_head_options_infer_the_shape_from_the_features() -> None:
    options = HeadOptions(proj_dim=32, hidden_dim=16, dropout=0.0, layer_weighting="mean")

    config = options.with_shape(n_layers=3, dim=64)

    assert (config.n_layers, config.dim) == (3, 64)
    assert (config.proj_dim, config.hidden_dim) == (32, 16)
    assert config.layer_weighting == "mean"


def test_aggregate_crops_averages_or_maximizes_probabilities() -> None:
    logits = np.array([-2.0, 0.0, 2.0])
    probabilities = 1.0 / (1.0 + np.exp(-logits))

    assert aggregate_crops(logits) == pytest.approx(float(probabilities.mean()))
    assert aggregate_crops(logits, "max_prob") == pytest.approx(float(probabilities.max()))
    assert aggregate_crops(torch.tensor([0.0])) == pytest.approx(0.5)

    with pytest.raises(ValueError, match="at least one crop"):
        aggregate_crops([])
    with pytest.raises(ValueError, match="Unknown aggregation mode"):
        aggregate_crops(logits, "median")  # type: ignore[arg-type]


def test_calibration_shifts_and_softens_the_logits() -> None:
    identity = Calibration()
    softened = Calibration(temperature=2.0, bias=1.0)
    logits = np.array([-4.0, 0.0, 4.0])

    assert np.allclose(identity.apply(logits), 1.0 / (1.0 + np.exp(-logits)))
    assert np.allclose(softened.apply(logits), 1.0 / (1.0 + np.exp(-(logits + 1.0) / 2.0)))
