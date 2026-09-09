"""Tests for the frozen-backbone feature extractor and its cache (optional ``ml`` extra).

These are marked ``ml`` and skipped unless ``torch`` and ``timm`` are
installed. They must also stay offline: downloading DINOv2 or CLIP weights
would make the suite depend on the Hugging Face Hub and on 350 MB of network
per run. Every test therefore monkeypatches
:func:`imgforensics.detectors.backbones.load_backbone` to build a randomly
initialised ``vit_tiny_patch16_224`` instead -- same architecture family, same
``forward_intermediates`` contract, 192-dimensional features, no download.
What is under test is the plumbing (shapes, batching, caching, the CLI), not
the numeric content of pretrained features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import natural_like_image
from typer.testing import CliRunner

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from imgforensics.cli import app  # noqa: E402
from imgforensics.core.image import ForensicImage  # noqa: E402
from imgforensics.data.manifest import build_manifest, label_from_parent_folder  # noqa: E402
from imgforensics.detectors import backbones  # noqa: E402
from imgforensics.detectors.backbones import BACKBONES, extract, normalization_for  # noqa: E402
from imgforensics.detectors.crops import CropPolicy  # noqa: E402
from imgforensics.detectors.features import FeatureCache, FeatureExtractor  # noqa: E402

pytestmark = pytest.mark.ml

runner = CliRunner()

_TINY_DIM = 192
_TINY_MODEL = "vit_tiny_patch16_224"
_IMAGE_SIZE = (320, 320)


def _tiny_backbone(spec: Any, device: str = "auto") -> Any:
    """Stand-in for ``load_backbone``: a random tiny ViT on the CPU, no download.

    Built the same way the real loader builds its models -- explicit
    ``img_size``, no classifier, eval mode, gradients off -- so the plumbing
    under test sees a model with the same contract.
    """
    model = timm.create_model(
        _TINY_MODEL, pretrained=False, num_classes=0, img_size=spec.input_size
    )
    model.eval()
    model.requires_grad_(False)
    return model.to("cpu")


@pytest.fixture
def tiny_backbone(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(backbones, "load_backbone", _tiny_backbone)
    return _tiny_backbone


def _write_images(root: Path, count: int) -> None:
    (root / "real").mkdir(parents=True, exist_ok=True)
    (root / "fake").mkdir(parents=True, exist_ok=True)
    for index in range(count):
        folder = "real" if index % 2 == 0 else "fake"
        image = natural_like_image(size=_IMAGE_SIZE, seed=100 + index)
        image.save(root / folder / f"{index}.png", format="PNG")


def _forensic_image(seed: int = 0) -> ForensicImage:
    import io

    buffer = io.BytesIO()
    natural_like_image(size=_IMAGE_SIZE, seed=seed).save(buffer, format="PNG")
    return ForensicImage.from_bytes(buffer.getvalue())


def test_extract_returns_one_cls_row_per_layer_plus_the_pooled_output() -> None:
    spec = BACKBONES["dinov2_vitb14"]
    model = _tiny_backbone(spec)
    batch = torch.randn(2, 3, spec.input_size, spec.input_size)

    features = extract(model, spec, batch)

    assert features.shape == (2, len(spec.layers) + 1, _TINY_DIM)
    assert features.dtype == torch.float32
    # Different blocks must not collapse to the same vector.
    assert not torch.allclose(features[:, 0], features[:, 1])


def test_normalization_comes_from_the_timm_pretrained_config() -> None:
    spec = BACKBONES["dinov2_vitb14"]
    model = _tiny_backbone(spec)

    mean, std = normalization_for(model, spec)
    assert mean == tuple(model.pretrained_cfg["mean"])
    assert std == tuple(model.pretrained_cfg["std"])

    override = spec.model_copy(update={"mean": (0.1, 0.2, 0.3), "std": (0.4, 0.5, 0.6)})
    assert normalization_for(model, override) == ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))


def test_features_for_image_has_shape_crops_layers_dim(tiny_backbone: Any) -> None:
    spec = BACKBONES["dinov2_vitb14"]
    extractor = FeatureExtractor(
        backbone="dinov2_vitb14",
        device="cpu",
        crop_policy=CropPolicy(size=spec.input_size, mode="grid", max_crops=2),
    )

    features = extractor.features_for_image(_forensic_image(seed=1))

    assert features.shape == (1, len(spec.layers) + 1, _TINY_DIM)
    assert features.dtype == np.float32


def test_cache_round_trip_stores_float16_and_returns_float32(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path / "features")
    policy = CropPolicy(size=224, mode="grid", max_crops=2)
    key = cache.key_for("a" * 64, "dinov2_vitb14", policy)
    array = np.arange(2 * 5 * 8, dtype=np.float32).reshape(2, 5, 8)

    assert cache.get(key) is None
    path = cache.put(key, array, layers=[8, 9, 10, 11])

    assert path.parent.name == "aa"
    assert path.name.startswith("a" * 64 + "_dinov2_vitb14_")
    with np.load(path, allow_pickle=False) as payload:
        assert payload["features"].dtype == np.float16
        assert list(payload["layers"]) == [8, 9, 10, 11]
        assert str(payload["backbone"]) == "dinov2_vitb14"
        assert json.loads(str(payload["crop_policy"]))["max_crops"] == 2
        assert str(payload["created"])

    loaded = cache.get(key)
    assert loaded is not None
    assert loaded.dtype == np.float32
    assert np.allclose(loaded, array, atol=0.5)

    assert cache.stats() == {"dinov2_vitb14": 1}
    assert cache.size_bytes() > 0


def test_cache_key_changes_with_image_backbone_and_policy(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path)
    policy = CropPolicy()
    base = cache.key_for("a" * 64, "dinov2_vitb14", policy)

    assert base.filename != cache.key_for("b" * 64, "dinov2_vitb14", policy).filename
    assert base.filename != cache.key_for("a" * 64, "clip_vitl14", policy).filename
    assert (
        base.filename != cache.key_for("a" * 64, "dinov2_vitb14", CropPolicy(max_crops=9)).filename
    )


def test_features_for_paths_caches_and_then_skips_the_backbone(
    tmp_path: Path, tiny_backbone: Any
) -> None:
    root = tmp_path / "images"
    _write_images(root, 4)
    paths = sorted(root.rglob("*.png"))
    cache = FeatureCache(tmp_path / "features")
    policy = CropPolicy(size=224, mode="grid", max_crops=2)

    first = FeatureExtractor(device="cpu", crop_policy=policy, batch_size=3)
    computed = list(first.features_for_paths(paths, cache=cache, progress=False))

    assert [path for path, _, _ in computed] == paths
    assert [view for _, view, _ in computed] == [0] * len(paths)
    assert first.cache_hits == 0
    assert len(list(cache.entries())) == 4

    second = FeatureExtractor(device="cpu", crop_policy=policy, batch_size=3)

    def _fail(*args: Any, **kwargs: Any) -> np.ndarray:
        raise AssertionError("a fully cached run must not touch the backbone")

    second._forward = _fail  # type: ignore[method-assign]
    reused = list(second.features_for_paths(paths, cache=cache, progress=False))

    assert second.cache_hits == len(paths)
    assert [path for path, _, _ in reused] == paths
    for (_, _, fresh), (_, _, cached) in zip(computed, reused, strict=True):
        assert np.allclose(fresh, cached, atol=0.05)


def test_features_for_paths_preserves_order_with_a_partly_warm_cache(
    tmp_path: Path, tiny_backbone: Any
) -> None:
    root = tmp_path / "images"
    _write_images(root, 4)
    paths = sorted(root.rglob("*.png"))
    cache = FeatureCache(tmp_path / "features")
    policy = CropPolicy(size=224, mode="grid", max_crops=2)

    warm = FeatureExtractor(device="cpu", crop_policy=policy, batch_size=2)
    list(warm.features_for_paths(paths[:1], cache=cache, progress=False))

    mixed = FeatureExtractor(device="cpu", crop_policy=policy, batch_size=2)
    results = list(mixed.features_for_paths(paths, cache=cache, progress=False))

    assert [path for path, _, _ in results] == paths
    assert mixed.cache_hits == 1


def test_cli_features_extract_and_info(tmp_path: Path, tiny_backbone: Any) -> None:
    root = tmp_path / "images"
    _write_images(root, 4)
    manifest, _ = build_manifest(
        root, dataset="tiny", label_of=label_from_parent_folder, progress=False
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest.save(manifest_path)
    cache_dir = tmp_path / "cache"

    result = runner.invoke(
        app,
        [
            "features",
            "extract",
            str(manifest_path),
            "--cache-dir",
            str(cache_dir),
            "--device",
            "cpu",
            "--max-crops",
            "2",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "images/s" in result.stdout
    assert len(list(FeatureCache(cache_dir).entries())) == 4
    assert (cache_dir / "index.json").is_file()

    info = runner.invoke(app, ["features", "info", "--cache-dir", str(cache_dir)])
    assert info.exit_code == 0, info.stdout
    assert "dinov2_vitb14" in info.stdout


def test_cli_features_extract_rejects_unknown_backbone_and_crop_mode(tmp_path: Path) -> None:
    root = tmp_path / "images"
    _write_images(root, 2)
    manifest, _ = build_manifest(
        root, dataset="tiny", label_of=label_from_parent_folder, progress=False
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest.save(manifest_path)

    unknown_backbone = runner.invoke(
        app, ["features", "extract", str(manifest_path), "--backbone", "nope"]
    )
    unknown_mode = runner.invoke(
        app, ["features", "extract", str(manifest_path), "--crop-mode", "diagonal"]
    )

    assert unknown_backbone.exit_code != 0
    assert unknown_mode.exit_code != 0


def _augmentation() -> Any:
    """A cheap but visible augmentation: always resize down-and-up, nothing else."""
    from imgforensics.eval.preprocess import AugmentationConfig

    return AugmentationConfig(downscale_upscale=(0.5, 0.6), p=1.0)


def test_cache_key_depends_on_the_augmentation_and_the_view(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path)
    policy = CropPolicy()
    plain = cache.key_for("a" * 64, "dinov2_vitb14", policy)
    augmented = cache.key_for("a" * 64, "dinov2_vitb14", policy, augment=_augmentation())
    second_view = cache.key_for("a" * 64, "dinov2_vitb14", policy, augment=_augmentation(), view=1)

    # View 0 with no augmentation keeps the pre-augmentation file name, so a
    # cache filled by an older run still answers.
    assert plain.filename == f"{'a' * 64}_dinov2_vitb14_{policy.fingerprint()}.npz"
    assert plain.filename != augmented.filename
    assert augmented.filename != second_view.filename
    assert second_view.filename.endswith("_v1.npz")


def test_stats_reads_the_backbone_name_from_both_file_name_forms(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path / "features")
    policy = CropPolicy()
    array = np.zeros((1, 5, 8), dtype=np.float32)
    cache.put(cache.key_for("a" * 64, "dinov2_vitb14", policy), array)
    cache.put(
        cache.key_for("a" * 64, "dinov2_vitb14", policy, augment=_augmentation(), view=1), array
    )
    cache.put(
        cache.key_for("b" * 64, "clip_vitl14", policy, augment=_augmentation(), view=2), array
    )

    assert cache.stats() == {"dinov2_vitb14": 2, "clip_vitl14": 1}


def test_views_produce_one_result_per_view_and_augmented_views_differ(
    tmp_path: Path, tiny_backbone: Any
) -> None:
    root = tmp_path / "images"
    _write_images(root, 2)
    paths = sorted(root.rglob("*.png"))
    cache = FeatureCache(tmp_path / "features")
    policy = CropPolicy(size=224, mode="grid", max_crops=1)

    extractor = FeatureExtractor(
        device="cpu", crop_policy=policy, batch_size=4, augment=_augmentation(), views=3
    )
    results = list(extractor.features_for_paths(paths, cache=cache, progress=False))

    assert [(path, view) for path, view, _ in results] == [
        (path, view) for path in paths for view in range(3)
    ]
    assert len(list(cache.entries())) == len(paths) * 3

    by_view = {view: features for path, view, features in results if path == paths[0]}
    # View 0 is the image untouched; every augmented view differs from it and
    # from the other augmented views (each has its own seed).
    assert not np.allclose(by_view[0], by_view[1], atol=1e-3)
    assert not np.allclose(by_view[1], by_view[2], atol=1e-3)


def test_view_zero_is_shared_with_an_un_augmented_extraction(
    tmp_path: Path, tiny_backbone: Any
) -> None:
    root = tmp_path / "images"
    _write_images(root, 2)
    paths = sorted(root.rglob("*.png"))
    cache = FeatureCache(tmp_path / "features")
    policy = CropPolicy(size=224, mode="grid", max_crops=1)

    plain = FeatureExtractor(device="cpu", crop_policy=policy)
    list(plain.features_for_paths(paths, cache=cache, progress=False))

    augmented = FeatureExtractor(device="cpu", crop_policy=policy, augment=_augmentation(), views=2)
    results = list(augmented.features_for_paths(paths, cache=cache, progress=False))

    # View 0 is un-augmented by construction, so it re-uses what the plain run
    # cached; only the augmented view had to be computed.
    assert augmented.cache_hits == len(paths)
    assert [view for _, view, _ in results] == [0, 1, 0, 1]


def test_augmented_views_are_reproducible_and_seeded_per_view() -> None:
    """Two extractors must cut identical pixels for a view; only the seed decides.

    Checked on the crops rather than the features because the stand-in
    backbone is randomly initialised per instance -- reproducibility of the
    *augmentation* is what this pins.
    """
    policy = CropPolicy(size=224, mode="grid", max_crops=1)
    image = _forensic_image(seed=11)

    def crops(view: int, seed: int = 0) -> np.ndarray:
        extractor = FeatureExtractor(
            device="cpu",
            crop_policy=policy.model_copy(update={"seed": seed}),
            augment=_augmentation(),
            views=3,
        )
        return np.asarray(extractor._crops_for_view(image, view)[0])

    assert np.array_equal(crops(1), crops(1))
    assert not np.array_equal(crops(1), crops(2))
    assert not np.array_equal(crops(1), crops(1, seed=7))
    assert np.array_equal(crops(0), np.asarray(_forensic_image(seed=11).rgb.crop((0, 0, 224, 224))))


def test_cli_features_extract_with_views_and_augment(tmp_path: Path, tiny_backbone: Any) -> None:
    root = tmp_path / "images"
    _write_images(root, 2)
    manifest, _ = build_manifest(
        root, dataset="tiny", label_of=label_from_parent_folder, progress=False
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest.save(manifest_path)
    cache_dir = tmp_path / "cache"

    result = runner.invoke(
        app,
        [
            "features",
            "extract",
            str(manifest_path),
            "--cache-dir",
            str(cache_dir),
            "--device",
            "cpu",
            "--max-crops",
            "1",
            "--views",
            "2",
            "--augment",
            "configs/augment_default.yaml",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert len(list(FeatureCache(cache_dir).entries())) == 4

    without_augment = runner.invoke(
        app, ["features", "extract", str(manifest_path), "--views", "2"]
    )
    assert without_augment.exit_code != 0
