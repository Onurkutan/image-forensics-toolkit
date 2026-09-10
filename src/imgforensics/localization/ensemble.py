"""An ensemble that combines the registered localizers into one heatmap.

``iml_vit`` and ``catnet_v2`` read different evidence -- one looks only at
pixels, the other also at the JPEG stream's quantized coefficients (see
:mod:`imgforensics.localization.iml_vit` and
:mod:`imgforensics.localization.catnet`) -- and on CocoGlide they disagree
about *where* an edit is often enough that averaging their maps is worth
measuring. This detector does exactly that and nothing more: it owns one
instance of each member, runs them, combines their heatmaps pixelwise, and
derives an image-level score from the result with the same top-1% rule the
members use (:func:`imgforensics.localization._scoring.top_fraction_score`),
so its score sits in the same benchmark column as theirs.

**The three modes**, chosen with ``mode=`` or the
:data:`MODE_ENV` environment variable:

- ``"mean"`` (the default): the pixelwise mean. Both members emit calibrated
  probabilities, so their mean is still a probability, and a fixed 0.5
  threshold keeps meaning what it means for either member alone.
- ``"max"``: the pixelwise maximum. Takes whichever member is more confident
  at each pixel, which finds a region one member missed entirely at the cost
  of inheriting the other's false positives.
- ``"rank_mean"``: each member's map is first replaced by its own per-image
  percentile rank, then averaged. This equalizes members whose probabilities
  live on different scales -- useful when one member is systematically
  under-confident -- but it **discards the absolute calibration of both**: a
  rank map's values are, by construction, spread uniformly over [0, 1], so
  thresholding one at 0.5 marks the upper half of the image no matter what
  the image contains. Pixel AP and best-F1 stay meaningful under it (they
  only need the ranking); F1@0.5 and IoU@0.5 do not. That is why it is not
  the default.

**Memory.** The ensemble builds its own member instances rather than sharing
whatever the caller already holds, so a ``benchmark`` run that lists
``iml_vit``, ``catnet_v2`` *and* ``localizer_ensemble`` together loads each
model twice and holds both copies for the whole run. On a 6 GB card, run the
ensemble in a separate ``benchmark`` invocation from its members.

Members that have no weights installed are dropped at :meth:`load` time and
those that abstain at prediction time are dropped from the combination, so
the ensemble degrades to whichever members are actually available. With none
of them available it abstains itself -- 0.5, ``"uncertain"``, and a
``details["reason"]`` naming the members that were missing -- rather than
failing a run, the same contract the members have.

This module imports no ``torch`` at all, not even inside a function: it only
ever calls its members, and each of them owns its own heavy imports. So
registering it costs nothing at startup, exactly as for the two wrappers it
builds on.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Literal, cast, get_args

import numpy as np
from PIL import Image

from imgforensics.core import registry
from imgforensics.core.base import BaseDetector
from imgforensics.core.image import ForensicImage
from imgforensics.core.registry import register
from imgforensics.core.types import DetectionResult, label_from_score
from imgforensics.localization._scoring import top_fraction_score

#: How the member heatmaps are combined. See the module docstring.
Mode = Literal["mean", "max", "rank_mean"]

#: The valid ``mode`` values, in the order they are documented.
MODES: tuple[str, ...] = get_args(Mode)

#: Environment variable choosing the default combination mode, so a whole
#: benchmark run can be switched without editing any call site.
MODE_ENV = "IMGFORENSICS_LOCALIZER_ENSEMBLE_MODE"

#: The mode used when neither the constructor nor the environment says.
DEFAULT_MODE: Mode = "mean"

#: Registry names of the members combined by default, in the order they run.
DEFAULT_MEMBERS: tuple[str, ...] = ("catnet_v2", "iml_vit")

#: Probability above which a pixel counts as manipulated, matching the
#: members and this project's pixel metrics
#: (:func:`imgforensics.eval.metrics.pixel_f1`).
_MASK_THRESHOLD = 0.5

_ABSTAIN_SCORE = 0.5
_DETAIL_DECIMALS = 4
_ELAPSED_DECIMALS = 2


def resolve_mode(mode: str | None = None) -> Mode:
    """The combination mode to use: argument, then :data:`MODE_ENV`, then the default.

    Args:
        mode: An explicit mode, or ``None`` to consult the environment.

    Returns:
        One of :data:`MODES`.

    Raises:
        ValueError: if the argument or the environment variable names a mode
            that does not exist. An unusable environment variable is worth
            failing on rather than silently ignoring: a benchmark run
            configured through it would otherwise report the default mode's
            numbers under the intended mode's name.
    """
    chosen = mode if mode is not None else os.environ.get(MODE_ENV)
    if chosen is None:
        return DEFAULT_MODE
    if chosen not in MODES:
        source = f"{MODE_ENV}=" if mode is None else "mode="
        raise ValueError(f"Unknown ensemble mode {source}{chosen!r}. Expected one of: {MODES}")
    return cast(Mode, chosen)


def resize_heatmap(heatmap: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Return ``heatmap`` at ``(height, width)``, bilinearly resampled if needed.

    Both current members return a map of the image's own shape, so this is
    usually a no-op cast. It exists for a member that predicts at a reduced
    resolution: the combination is pixelwise, so every map must share one
    grid, and the image's own grid is the one the caller's mask is aligned to.
    """
    if heatmap.shape == shape:
        return heatmap.astype(np.float32, copy=False)
    resized = Image.fromarray(heatmap.astype(np.float32)).resize(
        (shape[1], shape[0]), Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.float32)


def percentile_ranks(heatmap: np.ndarray) -> np.ndarray:
    """``heatmap``'s values replaced by their percentile rank within the map, in [0, 1].

    The smallest value becomes 0.0 and the largest 1.0. Equal values share
    the average of the ranks they span (the "average rank" tie rule), so a
    map with large flat areas -- which is what a confident localizer's
    background looks like -- does not have its ties broken arbitrarily by
    pixel order. A constant map therefore becomes a constant 0.5 rather than
    a gradient, and so does a single-pixel map, which has no ordering at all.

    Computed with one stable argsort and a run-length pass over the sorted
    values, so no SciPy dependency is needed.
    """
    flat = np.asarray(heatmap, dtype=np.float64).reshape(-1)
    if flat.size <= 1:
        return np.full(heatmap.shape, 0.5, dtype=np.float64)

    order = np.argsort(flat, kind="stable")
    ordered = flat[order]

    starts_run = np.empty(ordered.size, dtype=bool)
    starts_run[0] = True
    starts_run[1:] = ordered[1:] != ordered[:-1]

    run_starts = np.flatnonzero(starts_run)
    run_lengths = np.diff(np.append(run_starts, ordered.size))
    # The mean sorted position of each run of equal values, spread back over
    # every member of that run.
    mean_positions = run_starts + (run_lengths - 1) / 2.0
    ranked = np.empty(ordered.size, dtype=np.float64)
    ranked[order] = mean_positions[np.cumsum(starts_run) - 1]
    return (ranked / (ordered.size - 1)).reshape(heatmap.shape)


def combine(heatmaps: Sequence[np.ndarray], mode: str) -> np.ndarray:
    """Combine same-shaped member heatmaps into one float32 map in [0, 1].

    Args:
        heatmaps: At least one map, all of the same shape.
        mode: One of :data:`MODES`; see the module docstring for what each
            one trades away.

    Returns:
        The combined map, clipped to [0, 1].

    Raises:
        ValueError: if ``heatmaps`` is empty (callers abstain instead) or
            ``mode`` is not one of :data:`MODES`.
    """
    if not heatmaps:
        raise ValueError("combine() needs at least one heatmap")
    if mode == "rank_mean":
        stack = np.stack([percentile_ranks(heatmap) for heatmap in heatmaps])
        combined = stack.mean(axis=0)
    elif mode == "mean":
        combined = np.stack(heatmaps).mean(axis=0)
    elif mode == "max":
        combined = np.stack(heatmaps).max(axis=0)
    else:
        raise ValueError(f"Unknown ensemble mode {mode!r}. Expected one of: {MODES}")
    return np.clip(combined, 0.0, 1.0).astype(np.float32)


@register("localizer_ensemble")
class LocalizerEnsemble(BaseDetector):
    """Pixel-level localizer that combines the other registered localizers' heatmaps.

    One prediction is: run every member that has its weights installed,
    resample any map that does not already match the image's shape, combine
    them pixelwise under :attr:`mode`, and score the result with the members'
    own top-1% rule. The heatmap *is* the output; the score is derived from
    it.

    Members are built once per instance at :meth:`load` time and reused
    across :meth:`predict` calls, so a benchmark run pays each model's load
    once -- but see the module docstring on running this alongside its own
    members.
    """

    name = "localizer_ensemble"

    def __init__(
        self,
        members: Sequence[str] | None = None,
        mode: str | None = None,
        device: str = "auto",
    ) -> None:
        """Choose the members and the combination rule (nothing is built yet).

        Args:
            members: Registry names to combine, in the order they run.
                ``None`` means :data:`DEFAULT_MEMBERS`.
            mode: One of :data:`MODES`. ``None`` reads :data:`MODE_ENV`, and
                falls back to :data:`DEFAULT_MODE`.
            device: Passed straight to every member's ``load()``: ``"auto"``
                (CUDA when visible), ``"cpu"``, or an explicit device string.

        Raises:
            ValueError: if ``mode`` -- or :data:`MODE_ENV`, when ``mode`` is
                ``None`` -- names a mode that does not exist.
        """
        self.members = tuple(members) if members is not None else DEFAULT_MEMBERS
        self.mode: Mode = resolve_mode(mode)
        self._configured_device = device
        self.device = "cpu"
        self._instances: list[BaseDetector] = []
        self._missing: list[str] = []
        self._attempted = False

    @property
    def is_loaded(self) -> bool:
        """Whether at least one member was built and has its weights."""
        return bool(self._instances)

    @property
    def loaded_members(self) -> list[str]:
        """The names of the members that will actually run."""
        return [instance.name for instance in self._instances]

    def load(self, device: str = "auto") -> None:
        """Build every member and load it, keeping the ones that have weights.

        Never raises for missing weights: a member whose ``is_loaded`` is
        false after its own ``load()`` is dropped here and named in the
        abstention reason if nothing is left. A member that exposes no
        ``is_loaded`` at all -- any plain :class:`BaseDetector` -- is taken to
        be ready, since only the two weight-gated localizers have anything to
        be missing.

        ``device`` defaults to ``"auto"`` rather than the base class's
        ``"cpu"``, matching the members: a caller that runs this detector at
        all calls ``load()`` with no argument, and these models belong on the
        GPU when there is one.

        Raises:
            KeyError: if a member name is not in the registry. That is a
                configuration mistake rather than a missing download, so it
                fails loudly instead of being absorbed into an abstention.
        """
        self._attempted = True
        requested = device if device != "auto" else self._configured_device

        instances: list[BaseDetector] = []
        missing: list[str] = []
        for name in self.members:
            instance = registry.get(name)()
            instance.load(requested)
            if getattr(instance, "is_loaded", True):
                instances.append(instance)
            else:
                missing.append(name)

        self._instances = instances
        self._missing = missing
        # Every member resolved the same request, so the first one that
        # reports a device reports the ensemble's.
        member_devices = [str(getattr(instance, "device", "")) for instance in instances]
        self.device = next((device for device in member_devices if device), requested)

    def _abstain(self, missing: Sequence[str]) -> DetectionResult:
        """The result returned when no member produced a heatmap."""
        named = ", ".join(missing) if missing else "no members configured"
        reason = (
            "no ensemble member produced a heatmap; every member is missing its "
            f"weights or abstained: {named}"
        )
        return DetectionResult(
            detector=self.name,
            score=_ABSTAIN_SCORE,
            label="uncertain",
            details={"reason": reason, "members": list(missing)},
        )

    def predict(self, image: ForensicImage) -> DetectionResult:
        """Combine the members' heatmaps, or abstain when none produced one.

        ``details`` distinguishes two member lists on purpose:
        ``member_scores`` and ``member_elapsed_ms`` cover every member that
        ran, while ``members`` lists only those whose heatmap was actually
        combined -- so a member that ran and abstained is visible as the
        difference between them rather than disappearing from the record.
        """
        if not self._attempted:
            self.load()

        shape = (image.height, image.width)
        heatmaps: list[np.ndarray] = []
        combined_names: list[str] = []
        member_scores: dict[str, float] = {}
        member_elapsed: dict[str, float | None] = {}
        abstained: list[str] = []

        for instance in self._instances:
            result = instance.run(image)
            member_scores[instance.name] = round(float(result.score), _DETAIL_DECIMALS)
            member_elapsed[instance.name] = (
                round(float(result.elapsed_ms), _ELAPSED_DECIMALS)
                if result.elapsed_ms is not None
                else None
            )
            if result.heatmap is None:
                abstained.append(instance.name)
                continue
            heatmaps.append(resize_heatmap(result.heatmap, shape))
            combined_names.append(instance.name)

        if not heatmaps:
            return self._abstain([*self._missing, *abstained])

        heatmap = combine(heatmaps, self.mode)
        score = top_fraction_score(heatmap)

        details: dict[str, Any] = {
            "mode": self.mode,
            "members": combined_names,
            "member_scores": member_scores,
            "member_elapsed_ms": member_elapsed,
            "max_prob": round(float(heatmap.max()), _DETAIL_DECIMALS),
            "mean_prob": round(float(heatmap.mean()), _DETAIL_DECIMALS),
            f"area_fraction_above_{_MASK_THRESHOLD}": round(
                float((heatmap > _MASK_THRESHOLD).mean()), _DETAIL_DECIMALS
            ),
            "device": self.device,
        }
        return DetectionResult(
            detector=self.name,
            score=score,
            label=label_from_score(score),
            heatmap=heatmap,
            details=details,
        )
