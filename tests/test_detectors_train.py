"""Tests for the head training loop (optional ``ml`` extra).

No backbone runs here. The feature cache is filled directly with synthetic
features -- two Gaussian classes separated along a random subspace -- and the
images on disk exist only so a manifest can be built and hashed. Every cache
lookup therefore hits, ``FeatureExtractor`` never decodes an image or loads a
model, and the test measures exactly what it means to: the training loop, the
model selection, the calibration and the checkpoint it writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from imgforensics.data.manifest import (  # noqa: E402
    Manifest,
    build_manifest,
    label_from_parent_folder,
)
from imgforensics.detectors.crops import CropPolicy  # noqa: E402
from imgforensics.detectors.features import FeatureCache  # noqa: E402
from imgforensics.detectors.head import HeadOptions  # noqa: E402
from imgforensics.detectors.train import TrainConfig, train_head  # noqa: E402

pytestmark = pytest.mark.ml

_BACKBONE = "dinov2_vitb14"
_N_LAYERS = 5
_DIM = 32
_CROPS_PER_IMAGE = 4
_SIGNAL_DIMENSIONS = 6
# Tuned so the task is learnable but not instantly: the best epoch lands after
# the first (model selection is exercised) and the calibration fit still finds
# something to correct.
_SEPARATION = 0.6

_POLICY = CropPolicy(size=224, mode="grid", max_crops=_CROPS_PER_IMAGE)


def _write_images(root: Path, count: int, seed: int) -> None:
    """Tiny PNGs with distinct bytes, so every manifest entry has its own sha256."""
    rng = np.random.default_rng(seed)
    for index in range(count):
        folder = root / ("fake" if index % 2 else "real")
        folder.mkdir(parents=True, exist_ok=True)
        pixels = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
        Image.fromarray(pixels, mode="RGB").save(folder / f"{index:04d}.png", format="PNG")


def _class_features(label: float, rng: np.random.Generator, direction: np.ndarray) -> np.ndarray:
    """``(n_crops, n_layers, dim)`` noise, shifted along ``direction`` for the fake class."""
    block = rng.normal(0.0, 1.0, size=(_CROPS_PER_IMAGE, _N_LAYERS, _DIM))
    if label > 0.5:
        block += _SEPARATION * direction
    return block.astype(np.float32)


def _fill_cache(
    manifest: Manifest,
    manifest_root: Path,
    cache: FeatureCache,
    views: int,
    rng: np.random.Generator,
    direction: np.ndarray,
) -> None:
    """Write synthetic features for every (entry, view) the extractor will ask for."""
    import hashlib

    for entry in manifest.entries:
        digest = hashlib.sha256((manifest_root / entry.path).read_bytes()).hexdigest()
        label = 1.0 if entry.label == "fake" else 0.0
        for view in range(views):
            # View 0 is keyed without an augmentation, exactly as the extractor
            # keys it (view 0 is un-augmented by construction).
            key = cache.key_for(digest, _BACKBONE, _POLICY, view=view)
            cache.put(key, _class_features(label, rng, direction), layers=[8, 9, 10, 11])


def _build_split(
    tmp_path: Path,
    name: str,
    count: int,
    views: int,
    cache: FeatureCache,
    rng: np.random.Generator,
    direction: np.ndarray,
) -> Path:
    """Write images, build a manifest, and fill the cache for it. Returns the manifest path."""
    root = tmp_path / name
    _write_images(root, count, seed=abs(hash(name)) % 1000)
    manifest, _ = build_manifest(
        root,
        dataset=f"synthetic-{name}",
        label_of=label_from_parent_folder,
        license="CC-BY-4.0",
        commercial_ok=True,
        progress=False,
    )
    manifest_path = tmp_path / f"{name}.jsonl"
    manifest.save(manifest_path)
    _fill_cache(manifest, root, cache, views, rng, direction)
    return manifest_path


@pytest.fixture
def trained(tmp_path: Path) -> tuple[TrainConfig, Path]:
    """A ready-to-run config over a filled cache: 64 train x 2 views, 32 val."""
    rng = np.random.default_rng(0)
    direction = rng.normal(0.0, 1.0, size=(_N_LAYERS, _DIM))
    direction[:, _SIGNAL_DIMENSIONS:] = 0.0  # the signal lives in a small subspace

    cache = FeatureCache(tmp_path / "features")
    train_manifest = _build_split(tmp_path, "train", 64, 2, cache, rng, direction)
    val_manifest = _build_split(tmp_path, "val", 32, 1, cache, rng, direction)

    config = TrainConfig(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        cache_dir=tmp_path / "features",
        backbone=_BACKBONE,
        crop=_POLICY,
        views=2,
        head=HeadOptions(proj_dim=32, hidden_dim=32, dropout=0.1),
        epochs=8,
        batch_size=64,
        lr=3e-3,
        early_stopping_patience=8,
        seed=0,
        out_dir=tmp_path / "checkpoint",
        device="cpu",
    )
    return config, tmp_path


def test_train_head_separates_the_two_classes_and_writes_a_checkpoint(
    trained: tuple[TrainConfig, Path],
) -> None:
    config, _ = trained

    report = train_head(config, progress=False)

    assert report.meta.val.auc > 0.95
    assert report.train_crops == 64 * 2 * _CROPS_PER_IMAGE
    assert report.train_images == 64
    assert report.val_crops == 32 * _CROPS_PER_IMAGE
    assert report.val_images == 32
    assert 1 < report.meta.best_epoch <= report.meta.epochs_run <= config.epochs

    out_dir = Path(config.out_dir)
    assert (out_dir / "head.safetensors").is_file()
    assert (out_dir / "head.json").is_file()
    assert (out_dir / "training_log.jsonl").is_file()

    log_lines = (out_dir / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == report.meta.epochs_run
    first = json.loads(log_lines[0])
    assert set(first) == {
        "epoch",
        "train_loss",
        "val_auc",
        "val_balanced_accuracy",
        "lr",
        "seconds",
        "best",
    }


def test_checkpoint_json_describes_the_run_end_to_end(trained: tuple[TrainConfig, Path]) -> None:
    config, _ = trained

    report = train_head(config, progress=False)
    document = json.loads(Path(report.metadata_path).read_text(encoding="utf-8"))

    assert document["package_version"]
    assert document["created"]
    assert document["backbone"] == _BACKBONE
    assert document["layers"] == [8, 9, 10, 11]
    assert document["crop"]["max_crops"] == _CROPS_PER_IMAGE
    assert document["views"] == 2
    assert document["augment_hash"] == "none"  # this config sets no augmentation
    assert document["head"]["n_layers"] == _N_LAYERS
    assert document["head"]["dim"] == _DIM
    assert set(document["calibration"]) == {"temperature", "bias"}
    assert document["best_epoch"] >= 1
    assert set(document["val"]) == {
        "auc",
        "balanced_accuracy",
        "balanced_accuracy_tuned",
        "threshold",
        "ece_before",
        "ece_after",
    }

    roles = {entry["role"]: entry for entry in document["manifests"]}
    assert set(roles) == {"train", "val"}
    assert roles["train"]["entries"] == 64
    assert roles["val"]["entries"] == 32
    assert len(roles["train"]["sha256"]) == 64
    assert roles["train"]["dataset"] == "synthetic-train"

    # Both manifests are CC-BY-4.0 and commercially usable, so the head is too.
    assert document["commercial_ok"] is True
    assert document["licenses"] == ["CC-BY-4.0"]

    # Default config trains on every view; recorded in the checkpoint.
    assert document["train_views"] == "all"


def test_calibration_does_not_make_the_probabilities_worse(
    trained: tuple[TrainConfig, Path],
) -> None:
    config, _ = trained

    report = train_head(config, progress=False)

    assert report.meta.val.ece_after <= report.meta.val.ece_before + 0.01
    assert report.meta.calibration.temperature > 0.0
    # On this fixture the fit has real work to do, so it must be applied
    # rather than rejected in favour of the identity.
    assert report.meta.val.ece_after < report.meta.val.ece_before
    assert report.meta.calibration.temperature != pytest.approx(1.0)


def test_the_same_seed_reproduces_the_same_run(trained: tuple[TrainConfig, Path]) -> None:
    config, tmp_path = trained

    first = train_head(config, progress=False)
    second = train_head(
        config.model_copy(update={"out_dir": tmp_path / "checkpoint-again"}), progress=False
    )

    assert first.meta.best_epoch == second.meta.best_epoch
    assert first.meta.val.auc == pytest.approx(second.meta.val.auc, abs=1e-6)
    assert [record.val_auc for record in first.epochs] == pytest.approx(
        [record.val_auc for record in second.epochs], abs=1e-6
    )


def test_augmented_only_trains_on_views_1_and_up(trained: tuple[TrainConfig, Path]) -> None:
    config, tmp_path = trained
    config = config.model_copy(
        update={"train_views": "augmented_only", "out_dir": tmp_path / "checkpoint-augmented"}
    )

    report = train_head(config, progress=False)

    # views=2 in the fixture, so only view 1 remains: images x (views-1) x crops.
    assert report.train_crops == 64 * (2 - 1) * _CROPS_PER_IMAGE
    assert report.train_images == 64
    assert report.meta.train_views == "augmented_only"

    document = json.loads(Path(report.metadata_path).read_text(encoding="utf-8"))
    assert document["train_views"] == "augmented_only"


def test_augmented_only_requires_at_least_two_views(trained: tuple[TrainConfig, Path]) -> None:
    config, _ = trained
    config = config.model_copy(update={"train_views": "augmented_only", "views": 1})

    with pytest.raises(ValueError, match="augmented_only"):
        train_head(config, progress=False)


def test_a_single_class_training_manifest_is_rejected(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    direction = np.zeros((_N_LAYERS, _DIM))
    cache = FeatureCache(tmp_path / "features")

    root = tmp_path / "reals"
    (root / "real").mkdir(parents=True)
    for index in range(4):
        Image.fromarray(rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8), mode="RGB").save(
            root / "real" / f"{index}.png", format="PNG"
        )
    manifest, _ = build_manifest(
        root, dataset="reals", label_of=label_from_parent_folder, progress=False
    )
    manifest_path = tmp_path / "reals.jsonl"
    manifest.save(manifest_path)
    _fill_cache(manifest, root, cache, 1, rng, direction)

    config = TrainConfig(
        train_manifest=manifest_path,
        val_manifest=manifest_path,
        cache_dir=tmp_path / "features",
        crop=_POLICY,
        epochs=1,
        out_dir=tmp_path / "checkpoint",
        device="cpu",
    )

    with pytest.raises(ValueError, match="only one class"):
        train_head(config, progress=False)
