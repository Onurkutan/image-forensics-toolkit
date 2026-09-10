"""Plain-language explanations for one fused verdict.

:func:`explain` turns a fitted :class:`~imgforensics.fusion.stacking.Fuser`
and a live ``{detector: score}`` dict into a ranked list of
:class:`Contribution` objects, so the CLI (and any other caller) can show
*why* the fused probability came out the way it did, not just the number.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from imgforensics.fusion.stacking import Fuser, logit_from_score

#: One-line, plain-language description of what each detector actually
#: measures, shown next to its contribution in the fused-verdict report.
#: Keyed by the detector's registry name (see ``imgforensics.core.registry``).
DETECTOR_NOTES: dict[str, str] = {
    "metadata": "editor or generator markers in file metadata",
    "ela": "recompression error map",
    "c2pa": "C2PA content-provenance manifest verification",
    "sd_watermark": "Stable Diffusion invisible watermark decode",
    "copy_move": "duplicated (copy-moved) region detection",
    "jpeg_ghost": "JPEG recompression-quality mismatch (splice indicator)",
    "double_jpeg": "double-JPEG compression / blocking-grid offset",
    "dinov2_head": (
        "frozen DINOv2 features scored by a trained AI-generation head; its heatmap is each "
        "crop's probability, its attribution map is where inside those crops the head looked"
    ),
    "iml_vit": "IML-ViT pixel-level manipulation localization, reduced to an image score",
    "catnet_v2": "CAT-Net v2 compression-aware localization, reduced to an image score",
}

#: Shown for a detector name not in :data:`DETECTOR_NOTES` (a custom, future,
#: or renamed detector).
GENERIC_NOTE = "no plain-language note available for this detector"


@dataclass(frozen=True)
class Contribution:
    """One detector's share of a fused verdict.

    ``contribution`` is ``weight * logit``: the fuser's logistic weight for
    this detector times the logit of ``score`` (0 when the detector
    abstained at exactly 0.5 -- see
    :func:`imgforensics.fusion.stacking.logit_from_score`). It is signed:
    positive pushes the fused verdict toward "fake", negative toward "real".
    Only the logit weight is attributed here, not the presence-indicator
    weight (see ``Fuser.feature_vector``), since the indicator has no
    natural "direction" to show a reader -- it only matters when the
    detector is missing, reflected instead in ``present``.
    """

    detector: str
    score: float
    logit: float
    weight: float
    contribution: float
    present: bool
    note: str


def explain(fuser: Fuser, scores: Mapping[str, float]) -> list[Contribution]:
    """Rank ``fuser``'s detectors by their contribution to a prediction on ``scores``.

    ``scores`` need not include every detector ``fuser`` was fitted on -- a
    missing detector is imputed as abstaining (score 0.5, logit 0,
    ``present=False``), the same convention :meth:`Fuser.predict` uses.
    """
    contributions = []
    for index, detector in enumerate(fuser.detectors):
        present = detector in scores
        score = float(scores[detector]) if present else 0.5
        logit = logit_from_score(score)
        weight = float(fuser.logit_weights[index])
        contributions.append(
            Contribution(
                detector=detector,
                score=score,
                logit=logit,
                weight=weight,
                contribution=weight * logit,
                present=present,
                note=DETECTOR_NOTES.get(detector, GENERIC_NOTE),
            )
        )
    contributions.sort(key=lambda contribution: abs(contribution.contribution), reverse=True)
    return contributions
