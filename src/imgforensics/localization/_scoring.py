"""The image-level score rule every localizer in this package shares.

:mod:`imgforensics.localization.iml_vit` and
:mod:`imgforensics.localization.catnet` both wrap models whose papers report
pixel-level metrics only, so neither has an upstream rule for turning a
heatmap into one number. Both therefore report the **mean of the top
:data:`TOP_FRACTION` of heatmap values**, and the ensemble in
:mod:`imgforensics.localization.ensemble` reports the same statistic over the
combined map. Keeping the rule in one place is the point: the three are meant
to be read off the same benchmark table, and a score rule that drifted apart
between them would make their image-level AUCs incomparable.

Why this statistic rather than a plain mean or a plain max: a mean scales with
the manipulated region's size and so calls every small edit authentic, while a
max is one pixel of noise away from calling everything fake. The top 1% is the
smallest summary that still needs a confident *region* rather than a spike.

This module imports nothing beyond numpy, so it stays usable from the
torch-free import path every localizer wrapper is built on.
"""

from __future__ import annotations

import numpy as np

#: Fraction of the heatmap the image-level score averages over; always at
#: least one pixel, however small the map is.
TOP_FRACTION = 0.01

#: What an empty heatmap scores. No :class:`~imgforensics.core.image.ForensicImage`
#: has zero pixels, so this is a guard rather than a reachable value, and it
#: matches the score the localizers abstain with.
_EMPTY_SCORE = 0.5


def top_fraction_score(heatmap: np.ndarray, fraction: float = TOP_FRACTION) -> float:
    """Mean of the highest ``fraction`` of ``heatmap``'s values, clipped to [0, 1].

    The array's own dtype is kept rather than promoted, so a float32 heatmap
    averages in float32 exactly as it did before this helper was factored out
    of the two localizers.

    Args:
        heatmap: Any-shaped array of per-pixel probabilities; it is flattened
            first, so a region's shape never affects the result.
        fraction: Share of the pixels to average over. Rounded to a whole
            number of pixels, floored at one.

    Returns:
        The mean of that many largest values, or 0.5 for an empty array.
    """
    flat = np.asarray(heatmap).reshape(-1)
    if flat.size == 0:  # pragma: no cover - ForensicImage always has pixels
        return _EMPTY_SCORE
    keep = max(1, int(round(flat.size * fraction)))
    top = np.partition(flat, flat.size - keep)[flat.size - keep :]
    return float(np.clip(top.mean(), 0.0, 1.0))
