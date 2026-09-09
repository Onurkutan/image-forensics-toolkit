"""Frozen-backbone feature extraction and its on-disk cache.

A frozen backbone is run once per image and never again: the head that will
sit on top of these features (``docs/ROADMAP.md``, section 5, Phase 3) trains
in minutes, so the backbone forward pass -- not the training -- dominates the
cost of a sweep. :class:`FeatureCache` therefore keys features on everything
that could change them (the image's own sha256, the backbone name, and a hash
of the crop policy) and stores one small ``.npz`` per image, so re-running an
experiment with a different head, or after adding images to a manifest, reads
from disk instead of touching the GPU.

Features are stored as float16 (half the disk, and well inside the precision
of activations that were computed under float16 autocast on CUDA anyway) and
returned as float32.

``torch`` is only needed by the extractor, and only inside its methods, so
this module -- including the whole cache -- imports and works without the
optional ``ml`` extra installed.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from imgforensics.core.image import ForensicImage
from imgforensics.detectors.backbones import (
    BackboneSpec,
    get_backbone,
    normalization_for,
    resolve_device,
)
from imgforensics.detectors.crops import CropPolicy, crops_for, to_array

if TYPE_CHECKING:  # pragma: no cover - import-time typing only, never at runtime
    import torch

_DEFAULT_BATCH_SIZE = 32
_SHARD_PREFIX_LENGTH = 2
_FILENAME_PARTS = 3
_PROGRESS_EVERY = 50


@dataclass(frozen=True)
class CacheKey:
    """Everything a cached feature array depends on, plus its file name.

    ``image_sha256`` identifies the image by content rather than by path, so
    the same file cached under two manifests is computed once, and an edited
    file can never silently reuse the old features.
    """

    image_sha256: str
    backbone: str
    policy: CropPolicy
    policy_hash: str

    @property
    def filename(self) -> str:
        """``<sha256>_<backbone>_<policy hash>.npz`` -- the three key parts, in order.

        Neither the sha256 nor the policy hash contains an underscore, so a
        backbone name that does (``dinov2_vitb14``) is still recoverable by
        splitting on the first and last separator (see
        :meth:`FeatureCache.stats`).
        """
        return f"{self.image_sha256}_{self.backbone}_{self.policy_hash}.npz"

    @property
    def shard(self) -> str:
        """Subdirectory name (the sha256's first two characters).

        Sharding keeps any single directory to roughly 1/256th of the cache,
        which matters on Windows once a manifest reaches six figures.
        """
        return self.image_sha256[:_SHARD_PREFIX_LENGTH]


class FeatureCache:
    """A directory of per-image ``.npz`` feature files.

    Each file holds ``features`` (float16, ``(n_crops, n_layers, D)``),
    ``layers`` (the block indices behind the middle axis), ``crop_policy``
    (the policy as a JSON string), ``backbone`` (its registry name) and
    ``created`` (an ISO-8601 UTC timestamp).
    """

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)

    def key_for(self, image_sha256: str, backbone: str, policy: CropPolicy) -> CacheKey:
        """Build the cache key for one (image, backbone, crop policy) combination."""
        return CacheKey(
            image_sha256=image_sha256,
            backbone=backbone,
            policy=policy,
            policy_hash=policy.fingerprint(),
        )

    def path_for(self, key: CacheKey) -> Path:
        """Absolute path of the ``.npz`` file ``key`` maps to (it need not exist)."""
        return self.dir / key.shard / key.filename

    def get(self, key: CacheKey) -> np.ndarray | None:
        """Return the cached float32 feature array, or ``None`` on a miss.

        A file that exists but cannot be read (truncated by an interrupted
        run, say) counts as a miss rather than an error, so a damaged cache
        entry is simply recomputed and overwritten.
        """
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as payload:
                return np.asarray(payload["features"], dtype=np.float32)
        except (OSError, ValueError, KeyError):
            return None

    def put(self, key: CacheKey, array: np.ndarray, *, layers: Sequence[int] | None = None) -> Path:
        """Write ``array`` (cast to float16) to the cache and return its path."""
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=np.asarray(array, dtype=np.float16),
            layers=np.asarray(list(layers) if layers is not None else [], dtype=np.int32),
            crop_policy=np.asarray(key.policy.model_dump_json()),
            backbone=np.asarray(key.backbone),
            created=np.asarray(datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        return path

    def entries(self) -> Iterator[Path]:
        """Yield every cached ``.npz`` path, in sorted order."""
        if not self.dir.is_dir():
            return
        yield from sorted(self.dir.rglob("*.npz"))

    def size_bytes(self) -> int:
        """Total size on disk of every cached file."""
        return sum(path.stat().st_size for path in self.entries())

    def stats(self) -> dict[str, int]:
        """Count cached files per backbone name, read from the file names.

        Names that do not follow :attr:`CacheKey.filename` are counted under
        ``"unknown"`` rather than skipped, so a stray file in the cache
        directory is visible instead of silently ignored.
        """
        counts: dict[str, int] = {}
        for path in self.entries():
            parts = path.stem.split("_")
            name = "_".join(parts[1:-1]) if len(parts) >= _FILENAME_PARTS else "unknown"
            counts[name or "unknown"] = counts.get(name or "unknown", 0) + 1
        return counts

    def write_index(self) -> Path:
        """Write a small ``index.json`` summarizing the cache, and return its path.

        Purely informational (counts, total size, and when it was written);
        nothing reads it back, so a stale index can never corrupt a lookup.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / "index.json"
        payload = {
            "backbones": self.stats(),
            "files": sum(self.stats().values()),
            "size_bytes": self.size_bytes(),
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path


@dataclass
class _Pending:
    """One image queued for a batched forward pass (or already answered by the cache)."""

    path: Path
    key: CacheKey | None
    features: np.ndarray | None = None
    crops: list[Image.Image] = field(default_factory=list)


class FeatureExtractor:
    """Turns images into frozen-backbone features, batching crops across images.

    The backbone is loaded lazily on first use, so constructing an extractor
    is free and a CLI can validate its arguments (and fail on an unknown
    backbone name) before spending time on a download.
    """

    def __init__(
        self,
        backbone: str = "dinov2_vitb14",
        device: str = "auto",
        crop_policy: CropPolicy | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        """Build an extractor (nothing is loaded or downloaded yet).

        Args:
            backbone: Registry name from
                :data:`~imgforensics.detectors.backbones.BACKBONES`.
            device: ``"auto"`` (CUDA when available), ``"cpu"``, or an
                explicit torch device string.
            crop_policy: Crop policy; defaults to
                :class:`~imgforensics.detectors.crops.CropPolicy` with the
                backbone's own ``input_size`` as the crop size.
            batch_size: Maximum number of *crops* (not images) per forward
                pass.

        Raises:
            KeyError: ``backbone`` is not a registered backbone name.
        """
        self.spec: BackboneSpec = get_backbone(backbone)
        self.device = device
        self.crop_policy = crop_policy or CropPolicy(size=self.spec.input_size)
        self.batch_size = max(1, int(batch_size))
        self._model: torch.nn.Module | None = None
        self._normalization: tuple[tuple[float, ...], tuple[float, ...]] | None = None
        #: Cache hits during the most recent :meth:`features_for_paths` call.
        self.cache_hits = 0

    @property
    def model(self) -> torch.nn.Module:
        """The loaded frozen backbone, loading (and downloading) it on first access."""
        if self._model is None:
            from imgforensics.detectors.backbones import load_backbone

            self.device = resolve_device(self.device)
            self._model = load_backbone(self.spec, self.device)
        return self._model

    @property
    def normalization(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """The ``(mean, std)`` this backbone's weights were trained with."""
        if self._normalization is None:
            self._normalization = normalization_for(self.model, self.spec)
        return self._normalization

    def _forward(self, crops: Sequence[Image.Image]) -> np.ndarray:
        """Run ``crops`` through the backbone in ``batch_size`` chunks."""
        import torch

        from imgforensics.detectors.backbones import extract

        mean, std = self.normalization
        outputs: list[np.ndarray] = []
        for start in range(0, len(crops), self.batch_size):
            chunk = crops[start : start + self.batch_size]
            batch = torch.from_numpy(to_array(chunk, mean, std))
            outputs.append(extract(self.model, self.spec, batch).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def features_for_image(self, image: ForensicImage) -> np.ndarray:
        """Features for one image: ``(n_crops, n_layers, D)`` float32.

        ``n_crops`` is what the crop policy produced for this image (so it
        can be below ``max_crops`` for a small image in ``grid`` mode), and
        ``n_layers`` is ``len(spec.layers) + 1`` -- the selected blocks' CLS
        tokens followed by the pooled output (see
        :func:`imgforensics.detectors.backbones.extract`).
        """
        crops = crops_for(image.rgb, self.crop_policy, seed_material=image.raw)
        return self._forward(crops)

    def features_for_paths(
        self,
        paths: Iterable[str | Path],
        cache: FeatureCache | None = None,
        progress: bool = True,
    ) -> Iterator[tuple[Path, np.ndarray]]:
        """Yield ``(path, features)`` for every path, in input order.

        Crops are accumulated across images and forwarded together, so a
        batch stays full even when each image contributes only a few crops.
        When a ``cache`` is given, an image whose features are already
        cached is never decoded twice: it is read back from disk, and newly
        computed features are written before they are yielded.

        Args:
            paths: Image paths to process, in the order results are wanted.
            cache: Optional :class:`FeatureCache` to read from and write to.
            progress: Print a one-line progress count every 50 images (plain
                ``print``, as in :func:`imgforensics.data.manifest.build_manifest`).
        """
        queue: list[_Pending] = []
        queued_crops = 0
        done = 0
        reported = 0
        self.cache_hits = 0
        all_paths = [Path(path) for path in paths]

        for path in all_paths:
            pending = self._prepare(path, cache)
            if pending.features is not None:
                self.cache_hits += 1
            queue.append(pending)
            queued_crops += len(pending.crops)
            if queued_crops >= self.batch_size:
                yield from self._flush(queue, cache)
                done += len(queue)
                queue, queued_crops = [], 0
                if progress and done - reported >= _PROGRESS_EVERY:
                    print(f"[features] {done}/{len(all_paths)} images")
                    reported = done

        if queue:
            yield from self._flush(queue, cache)
            done += len(queue)
        if progress:
            print(f"[features] {done}/{len(all_paths)} images")

    def _prepare(self, path: Path, cache: FeatureCache | None) -> _Pending:
        """Read one image, answering from ``cache`` when possible.

        On a cache hit only the file's bytes are hashed -- the image is
        never decoded and no crop is cut.
        """
        data = path.read_bytes()
        key: CacheKey | None = None
        if cache is not None:
            key = cache.key_for(hashlib.sha256(data).hexdigest(), self.spec.name, self.crop_policy)
            cached = cache.get(key)
            if cached is not None:
                return _Pending(path=path, key=key, features=cached)

        image = ForensicImage.from_bytes(data, path=path)
        crops = crops_for(image.rgb, self.crop_policy, seed_material=image.raw)
        return _Pending(path=path, key=key, crops=list(crops))

    def _flush(
        self, queue: Sequence[_Pending], cache: FeatureCache | None
    ) -> Iterator[tuple[Path, np.ndarray]]:
        """Forward every queued image's crops in one go and yield results in order."""
        batch_crops = [crop for pending in queue for crop in pending.crops]
        stacked = self._forward(batch_crops) if batch_crops else None

        offset = 0
        for pending in queue:
            if pending.features is not None:
                yield pending.path, pending.features
                continue
            count = len(pending.crops)
            if stacked is None:  # pragma: no cover - a queued miss always has crops
                continue
            features = stacked[offset : offset + count]
            offset += count
            if cache is not None and pending.key is not None:
                cache.put(pending.key, features, layers=self.spec.layers)
            yield pending.path, features


def extract_to_cache(
    paths: Sequence[str | Path],
    cache: FeatureCache,
    extractor: FeatureExtractor,
    progress: bool = True,
) -> dict[str, float | int | str]:
    """Extract and cache features for ``paths``, returning a small run summary.

    Returns a mapping with ``images``, ``cache_hits`` (how many of them were
    already cached, and so skipped the backbone), ``elapsed_s``,
    ``images_per_s``, ``cache_bytes`` and ``feature_shape`` (as a string),
    which is what the CLI prints.

    ``elapsed_s`` covers the whole call, including the one-off backbone load
    (and, on the very first run, the weight download) that the first uncached
    image triggers -- so the reported throughput of a short run understates
    the steady-state rate. A fully cached run loads no backbone at all.
    """
    start = time.perf_counter()
    shape: tuple[int, ...] | None = None
    count = 0
    for _, features in extractor.features_for_paths(paths, cache=cache, progress=progress):
        shape = features.shape
        count += 1
    elapsed = time.perf_counter() - start
    return {
        "images": count,
        "cache_hits": extractor.cache_hits,
        "elapsed_s": elapsed,
        "images_per_s": (count / elapsed) if elapsed > 0 else 0.0,
        "cache_bytes": cache.size_bytes(),
        "feature_shape": str(shape) if shape is not None else "-",
    }
