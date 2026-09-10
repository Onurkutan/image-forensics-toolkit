"""Tests for imgforensics.service.catalogue: what a client is offered before it runs anything."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from imgforensics.core import registry
from imgforensics.core.base import BaseDetector
from imgforensics.core.image import ForensicImage
from imgforensics.core.types import DetectionResult
from imgforensics.detectors import is_ml_available
from imgforensics.fusion.report import GENERIC_NOTE
from imgforensics.localization.weights import WEIGHTS
from imgforensics.service import catalogue
from imgforensics.service.catalogue import FALLBACK_CATEGORY, ToolSpec, tool_spec
from imgforensics.signals import SIGNAL_NAMES
from imgforensics.views import VIEW_NAMES

requires_ml = pytest.mark.skipif(not is_ml_available(), reason="needs the optional ml extra")

#: A tool none of the catalogue's tables knows about, to check the fallbacks.
_UNLISTED_NAME = "unlisted_probe"


class _UnlistedDetector(BaseDetector):
    name = _UNLISTED_NAME

    def predict(self, image: ForensicImage) -> DetectionResult:
        return DetectionResult(detector=self.name, score=0.5, label="uncertain")


def _by_name() -> dict[str, ToolSpec]:
    return {spec.name: spec for spec in catalogue()}


def test_every_registered_tool_is_listed_exactly_once() -> None:
    specs = catalogue()

    names = [spec.name for spec in specs]
    assert names == sorted(set(names), key=names.index), "no tool is listed twice"
    assert set(names) == set(registry.available())


def test_the_signals_and_the_views_are_always_there() -> None:
    by_name = _by_name()

    for name in SIGNAL_NAMES:
        assert by_name[name].kind == "signal"
    for name in VIEW_NAMES:
        assert by_name[name].kind == "view"


def test_every_tool_has_a_category_a_display_name_and_a_note() -> None:
    for spec in catalogue():
        assert spec.category
        assert spec.display_name
        assert spec.note


def test_the_listing_is_ordered_by_category_then_name() -> None:
    specs = catalogue()

    assert specs == sorted(specs, key=lambda spec: (spec.category, spec.name))


def test_the_listing_is_stable_across_calls() -> None:
    assert catalogue() == catalogue()


def test_a_tool_nobody_categorised_still_gets_a_usable_entry() -> None:
    registry.register(_UNLISTED_NAME)(_UnlistedDetector)

    spec = tool_spec(_UNLISTED_NAME)

    assert spec.category == FALLBACK_CATEGORY
    assert spec.display_name == "Unlisted probe"
    assert spec.note == GENERIC_NOTE
    assert spec.kind == "signal"
    assert spec.needs_ml is False
    assert spec.installed is True
    assert spec.parameters == []


def test_the_declared_parameters_are_offered_with_their_ranges() -> None:
    quality = _by_name()["ela"].parameters[0]

    assert quality.name == "quality"
    assert quality.kind == "int"
    assert quality.default == 95
    assert (quality.minimum, quality.maximum) == (50, 100)
    assert quality.description


def test_a_signal_without_parameters_declares_none() -> None:
    assert _by_name()["metadata"].parameters == []


def test_a_signal_needs_nothing_installed() -> None:
    spec = _by_name()["ela"]

    assert spec.needs_ml is False
    assert spec.installed is True


@requires_ml
def test_the_ml_tools_declare_their_parameters() -> None:
    by_name = _by_name()

    assert by_name["dinov2_head"].kind == "detector"
    attribution = by_name["dinov2_head"].parameters[0]
    assert (attribution.name, attribution.kind, attribution.default) == (
        "attribution",
        "bool",
        True,
    )

    assert by_name["localizer_ensemble"].kind == "localizer"
    mode = by_name["localizer_ensemble"].parameters[0]
    assert (mode.name, mode.kind, mode.default) == ("mode", "choice", "mean")
    assert set(mode.choices) == {"mean", "max", "rank_mean"}


@requires_ml
def test_a_weight_gated_tool_is_not_installed_until_its_weights_are_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMGFORENSICS_WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("IMGFORENSICS_HEAD_DIR", str(tmp_path / "head"))

    by_name = _by_name()
    for name in ("iml_vit", "catnet_v2", "localizer_ensemble", "dinov2_head"):
        assert by_name[name].needs_ml is True
        assert by_name[name].installed is False, name

    weights_path = tmp_path / "weights" / "iml_vit" / WEIGHTS["iml_vit"].filename
    weights_path.parent.mkdir(parents=True)
    weights_path.write_bytes(b"stand-in for a checkpoint")

    by_name = _by_name()
    assert by_name["iml_vit"].installed is True
    assert by_name["catnet_v2"].installed is False
    # The ensemble runs whichever members it has, so one member is enough.
    assert by_name["localizer_ensemble"].installed is True


@requires_ml
def test_a_head_needs_both_of_its_files_to_count_as_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head_dir = tmp_path / "head"
    head_dir.mkdir()
    monkeypatch.setenv("IMGFORENSICS_HEAD_DIR", str(head_dir))
    (head_dir / "head.json").write_text("{}", encoding="utf-8")

    assert _by_name()["dinov2_head"].installed is False

    (head_dir / "head.safetensors").write_bytes(b"stand-in for weights")

    assert _by_name()["dinov2_head"].installed is True


def test_the_catalogue_works_without_torch() -> None:
    """The service layer imports, and lists tools, on a base install.

    The ``ml`` tools are then simply absent -- the same contract
    ``imgforensics analyze`` already has -- and asking whether their weights
    are installed must not be what drags torch in.
    """
    snippet = textwrap.dedent(
        """
        import sys

        blocked = {"torch", "timm", "torchvision"}

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in blocked:
                    raise ImportError(f"blocked for this test: {name}")
                return None

        sys.meta_path.insert(0, Blocker())

        from imgforensics.service import catalogue

        names = [spec.name for spec in catalogue()]
        assert not blocked & set(sys.modules)
        assert "dinov2_head" not in names
        assert "iml_vit" not in names
        assert "localizer_ensemble" not in names
        assert "ela" in names
        assert "luminance_gradient" in names
        assert all(spec.installed for spec in catalogue())
        assert not any(spec.needs_ml for spec in catalogue())
        print("imported cleanly")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", snippet], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert "imported cleanly" in completed.stdout
