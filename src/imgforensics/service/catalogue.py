"""The tool catalogue: what a client can run, before it runs anything.

A workbench draws a tool tree before the user has chosen a tool, and it has to
draw the tools it cannot run as well as the ones it can -- greyed out and
labelled, never missing (``docs/design/01_toolbox_architecture.md``, sections
2.4 and 3.1). :func:`catalogue` is that listing: one :class:`ToolSpec` per
registered tool, grouped, named, described, with its parameters and with
``installed`` saying whether its weights are on this machine.

Two properties are load-bearing here:

- **Nothing is built.** The catalogue reads class attributes and file paths
  only. Asking what is available must not load a 900 MB checkpoint, and on a
  torch-free install it must not import torch to find out that torch is
  missing -- so ``installed`` is answered from
  :mod:`imgforensics.localization.weights` and the head's checkpoint
  directory, both of which are plain path checks.
- **The registry decides membership.** A tool that registered itself is in the
  catalogue, and one that did not is absent, which is exactly how the ``ml``
  extra already gates the learned detector and the two localizers. The tables
  below only *decorate* what the registry holds, so a name missing from them
  yields a usable entry rather than an exception.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from imgforensics.core import registry
from imgforensics.core.parameters import ParameterSpec
from imgforensics.core.types import ToolKind
from imgforensics.fusion.report import DETECTOR_NOTES, GENERIC_NOTE
from imgforensics.localization.weights import WEIGHTS, weights_file

#: Which group of the design note's tool catalogue (section 6) each tool
#: belongs to. The grouping is the user's mental model, not the package
#: layout: ``sd_watermark`` is a noise-domain decode and sits with the noise
#: tools, while ``c2pa`` is provenance rather than metadata parsing.
_CATEGORIES: dict[str, str] = {
    "metadata": "Metadata",
    "c2pa": "Provenance",
    "ela": "JPEG",
    "jpeg_ghost": "JPEG",
    "double_jpeg": "JPEG",
    "copy_move": "Tampering",
    "iml_vit": "Tampering",
    "catnet_v2": "Tampering",
    "localizer_ensemble": "Tampering",
    "dinov2_head": "AI generation",
    "sd_watermark": "Noise",
    "noise_residual": "Noise",
    "bit_planes": "Noise",
    "luminance_gradient": "Detail",
}

#: Category for a tool not named in :data:`_CATEGORIES` -- the design note's
#: own bucket for tools that fit nowhere else.
FALLBACK_CATEGORY = "Various"

#: Names as a person would write them, where lowercasing the registry name
#: would be wrong (an acronym, a model's own capitalisation).
_DISPLAY_NAMES: dict[str, str] = {
    "c2pa": "C2PA provenance",
    "ela": "Error Level Analysis",
    "jpeg_ghost": "JPEG ghost",
    "double_jpeg": "Double JPEG",
    "sd_watermark": "SD watermark",
    "copy_move": "Copy-move",
    "dinov2_head": "DINOv2 head",
    "iml_vit": "IML-ViT",
    "catnet_v2": "CAT-Net v2",
    "localizer_ensemble": "Localizer ensemble",
}

#: Tools that only exist when the optional ``ml`` extra is installed. Listed
#: explicitly rather than inferred from :data:`~imgforensics.core.types.ToolKind`,
#: because a classical detector or localizer needing no torch is a perfectly
#: possible future entry.
_NEEDS_ML = frozenset({"dinov2_head", "iml_vit", "catnet_v2", "localizer_ensemble"})

#: Files a trained head is made of; both must exist for ``dinov2_head`` to
#: count as installed (:mod:`imgforensics.detectors.learned`).
_HEAD_FILES = ("head.json", "head.safetensors")


class ToolSpec(BaseModel):
    """One registered tool, described well enough to draw and to run.

    Attributes:
        name: The registry name, and what a caller passes to run it.
        display_name: The name to show a person.
        category: Which group of the tool tree it belongs to.
        kind: Signal, detector, localizer or view -- see
            :data:`~imgforensics.core.types.ToolKind`.
        needs_ml: Whether the tool comes from the optional ``ml`` extra. It is
            always false on an install without it, where such a tool is simply
            not in the catalogue at all.
        installed: Whether the tool can actually produce a result here.
            Signals and views are always installed; a weight-gated tool is
            installed once its weights have been fetched. A tool that is
            registered but not installed still runs -- it abstains with a
            reason -- so this drives a "fetch the weights" hint, not a refusal.
        parameters: The tool's declared settings, in display order.
        note: One plain-language line on what the tool reads, shared with the
            fused-verdict report (:data:`~imgforensics.fusion.report.DETECTOR_NOTES`).
    """

    name: str
    display_name: str
    category: str
    kind: ToolKind
    needs_ml: bool
    installed: bool
    parameters: list[ParameterSpec] = Field(default_factory=list)
    note: str


def display_name(name: str) -> str:
    """The label for a registry name: the curated one, else the name made readable."""
    if name in _DISPLAY_NAMES:
        return _DISPLAY_NAMES[name]
    words = name.replace("_", " ")
    return words[:1].upper() + words[1:]


def is_installed(name: str) -> bool:
    """Whether ``name`` has everything it needs to run on this machine.

    Answered from the filesystem alone, so it costs nothing and works without
    the ``ml`` extra: the two localizers and the ensemble look for the weight
    files :mod:`imgforensics.localization.weights` would download (the
    ensemble is installed as soon as *one* member is, since it runs whichever
    members it has), and the learned detector looks for the head checkpoint
    under ``$IMGFORENSICS_HEAD_DIR``. Anything else -- every signal, every
    view -- is installed by virtue of being importable.
    """
    if name in WEIGHTS:
        return weights_file(name).is_file()
    if name == "localizer_ensemble":
        # Imported here, not at module scope: the ensemble only exists where
        # the ``ml`` extra does, and this branch is only reached when it has
        # registered itself, which means the module is already imported.
        from imgforensics.localization.ensemble import DEFAULT_MEMBERS

        return any(weights_file(member).is_file() for member in DEFAULT_MEMBERS)
    if name == "dinov2_head":
        from imgforensics.detectors.learned import resolve_checkpoint_dir

        checkpoint_dir = resolve_checkpoint_dir()
        return all((checkpoint_dir / filename).is_file() for filename in _HEAD_FILES)
    return True


def tool_spec(name: str) -> ToolSpec:
    """Describe one registered tool.

    Raises:
        KeyError: if ``name`` is not registered (the registry's message).
    """
    tool = registry.get(name)
    return ToolSpec(
        name=name,
        display_name=display_name(name),
        category=_CATEGORIES.get(name, FALLBACK_CATEGORY),
        kind=tool.kind,
        needs_ml=name in _NEEDS_ML,
        installed=is_installed(name),
        parameters=tool.parameters(),
        note=DETECTOR_NOTES.get(name, GENERIC_NOTE),
    )


def catalogue() -> list[ToolSpec]:
    """Every registered tool, ordered by category and then by name.

    The order is the one a tool tree is drawn in, and it is stable: two calls
    on the same install return the same list, so a client can compare
    catalogues across runs.
    """
    specs = [tool_spec(name) for name in registry.available()]
    return sorted(specs, key=lambda spec: (spec.category, spec.name))
