"""Tests for imgforensics.demo: what the public page offers, runs and says.

Two tools are registered here -- a signal reporting both maps and a numpy
value in its details, and a view -- so no real model runs, no weights are
needed and no torch is imported, the same arrangement ``test_api.py`` uses.
The weight and head directories point at an empty temporary directory and the
working directory is moved there too, so a tool that somehow got run would
abstain rather than download anything and a fuser sitting in the checkout's
``weights/`` is never picked up by accident.

The tool list is asserted against a hand-built catalogue rather than the real
one: whether this machine has CAT-Net's weights is not something a test about
checkbox defaults should depend on, and an uninstalled ``ml`` tool must be
listed even on an install where the ``ml`` extra is absent and no such tool is
registered at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import piexif
import pytest
from conftest import natural_like_image, synthetic_fusion_records
from PIL import Image, PngImagePlugin

from imgforensics.core import registry
from imgforensics.core.base import BaseDetector
from imgforensics.core.image import ForensicImage
from imgforensics.core.types import DetectionResult, label_from_score
from imgforensics.demo import DemoConfig, analyze_image, tool_choices
from imgforensics.demo.core import NOT_INSTALLED_NOTE
from imgforensics.fusion.stacking import Fuser, fit_fuser
from imgforensics.service.catalogue import ToolSpec
from imgforensics.views.base import ViewTool

#: The two tools these tests run, registered here rather than borrowed from
#: the real catalogue so that a change in what ELA scores never fails a demo
#: test, and so that both maps are always present.
_SIGNAL_NAME = "demo_probe"
_VIEW_NAME = "demo_probe_view"

#: Display names the catalogue derives from those registry names.
_SIGNAL_LABEL = "Demo probe"
_VIEW_LABEL = "Demo probe view"

#: The probe's fixed score: comfortably "fake", so a summary line and a fused
#: verdict both have something to say.
_PROBE_SCORE = 0.75


class _ProbeSignal(BaseDetector):
    """A signal with both maps and a numpy scalar in its details."""

    name = _SIGNAL_NAME

    def predict(self, image: ForensicImage) -> DetectionResult:
        ramp = np.linspace(0.0, 1.0, image.height * image.width, dtype=np.float32).reshape(
            image.height, image.width
        )
        return DetectionResult(
            detector=self.name,
            score=_PROBE_SCORE,
            label=label_from_score(_PROBE_SCORE),
            heatmap=ramp,
            attribution=np.ascontiguousarray(ramp[::-1]),
            details={"not_json_native": np.float32(0.25)},
        )


class _ProbeView(ViewTool):
    """A view: one map, no verdict, and no attribution to go with it."""

    name = _VIEW_NAME

    def predict(self, image: ForensicImage) -> DetectionResult:
        return self.view_result(np.zeros((image.height, image.width), dtype=np.float32))


registry.register(_SIGNAL_NAME)(_ProbeSignal)
registry.register(_VIEW_NAME)(_ProbeView)


@pytest.fixture(autouse=True)
def _isolated_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No weights, no head checkpoint, and no fuser found by accident."""
    monkeypatch.setenv("IMGFORENSICS_WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("IMGFORENSICS_HEAD_DIR", str(tmp_path / "head"))
    monkeypatch.delenv("IMGFORENSICS_FUSER", raising=False)
    monkeypatch.chdir(tmp_path)


def _spec(name: str, category: str, kind: str, *, installed: bool, needs_ml: bool = False):
    """One catalogue entry, hand-built."""
    return ToolSpec(
        name=name,
        display_name=name.replace("_", " ").capitalize(),
        category=category,
        kind=kind,
        needs_ml=needs_ml,
        installed=installed,
        parameters=[],
        note="what this tool reads",
    )


#: A catalogue with one of each case the tool list has to draw.
_FAKE_CATALOGUE = [
    _spec("fake_signal", "JPEG", "signal", installed=True),
    _spec("fake_localizer", "Tampering", "localizer", installed=False, needs_ml=True),
    _spec("fake_view", "Detail", "view", installed=True),
]


def _fitted_fuser() -> Fuser:
    """A fuser fitted on synthetic records naming the two probes."""
    records = synthetic_fusion_records(
        {
            _SIGNAL_NAME: lambda y, rng: float(
                np.clip(y * 0.9 + 0.05 + rng.normal(0, 0.05), 1e-3, 1 - 1e-3)
            ),
            _VIEW_NAME: lambda _y, _rng: 0.5,
        },
        n_per_class=150,
        seed=0,
    )
    return fit_fuser(records)


def _image(size: tuple[int, int] = (120, 90)) -> Image.Image:
    return natural_like_image(size=size, seed=3)


# --------------------------------------------------------------------------
# analyze_image
# --------------------------------------------------------------------------


def test_every_selected_tool_gets_a_card_and_every_map_gets_an_overlay() -> None:
    result = analyze_image(_image(), [_SIGNAL_NAME, _VIEW_NAME])

    assert [card.name for card in result.cards] == [_SIGNAL_NAME, _VIEW_NAME]
    signal_card, view_card = result.cards
    assert (signal_card.display_name, signal_card.kind) == (_SIGNAL_LABEL, "signal")
    assert (signal_card.score, signal_card.label) == (_PROBE_SCORE, "fake")
    assert signal_card.elapsed_ms is not None
    assert signal_card.caution is None
    assert view_card.kind == "view"
    assert view_card.label == "uncertain"

    assert [label for label, _ in result.overlays] == [
        f"{_SIGNAL_LABEL} heatmap",
        f"{_SIGNAL_LABEL} attribution",
        f"{_VIEW_LABEL} heatmap",
    ]
    assert all(overlay.size == _image().size for _, overlay in result.overlays)


def test_a_cards_details_survive_json() -> None:
    result = analyze_image(_image(), [_SIGNAL_NAME])

    details = result.cards[0].details
    assert json.loads(json.dumps(details))["not_json_native"] == pytest.approx(0.25)


def test_the_summary_leads_with_a_score_bar_per_tool() -> None:
    result = analyze_image(_image(), [_SIGNAL_NAME, _VIEW_NAME])

    lines = [line for line in result.summary_markdown.splitlines() if line.startswith("- `[")]
    assert len(lines) == 2
    assert f"0.75 fake -- {_SIGNAL_LABEL}" in lines[0]
    assert f"{_VIEW_LABEL} (view -- no verdict)" in lines[1]
    assert "Fused verdict" not in result.summary_markdown


def test_selecting_nothing_says_so_instead_of_showing_an_empty_page() -> None:
    result = analyze_image(_image(), [])

    assert result.cards == []
    assert result.overlays == []
    assert "No tools were selected" in result.summary_markdown


# --------------------------------------------------------------------------
# The demo's own downscale
# --------------------------------------------------------------------------


def test_an_upload_over_the_cap_is_downscaled_and_the_summary_says_so() -> None:
    result = analyze_image(_image((400, 100)), [_SIGNAL_NAME], config=DemoConfig(max_side=128))

    assert result.downscaled is True
    assert "downscaled to 128 px" in result.summary_markdown
    assert "the library itself never resizes" in result.summary_markdown
    assert max(result.overlays[0][1].size) == 128


def test_an_upload_under_the_cap_is_analyzed_untouched() -> None:
    result = analyze_image(_image((64, 48)), [_SIGNAL_NAME], config=DemoConfig(max_side=128))

    assert result.downscaled is False
    assert "downscaled" not in result.summary_markdown
    assert result.overlays[0][1].size == (64, 48)


# --------------------------------------------------------------------------
# Where the upload's bytes come from
# --------------------------------------------------------------------------
#
# The real ``metadata`` signal (imgforensics.signals.metadata), not the demo
# probe, is what these tests run: it abstains with score 0.5 unless it can
# read an encoded file's EXIF, so its score is the proof that a path or
# ``bytes`` upload reached ``analyze_image`` with its original bytes intact,
# and that a bare PIL image or an over-the-cap upload did not.


def _jpeg_with_editor_marker(path: Path, size: tuple[int, int] = (64, 48)) -> None:
    """Write a JPEG at ``path`` whose EXIF Software tag names a known editor.

    Gives ``MetadataSignal`` an editor marker to find (see its
    ``editor_markers`` rule), so it scores 0.70 "fake" instead of abstaining.
    """
    exif_bytes = piexif.dump({"0th": {piexif.ImageIFD.Software: b"Adobe Photoshop 25.0"}})
    natural_like_image(size=size, seed=11).save(path, format="JPEG", quality=90, exif=exif_bytes)


def test_a_path_upload_lets_metadata_read_the_editor_marker(tmp_path: Path) -> None:
    path = tmp_path / "edited.jpg"
    _jpeg_with_editor_marker(path)

    result = analyze_image(str(path), ["metadata"])

    card = result.cards[0]
    assert (card.score, card.label) == (0.70, "fake")
    assert card.details["editor_markers"]
    assert card.caution is None
    assert "original encoded bytes were not available" not in result.summary_markdown


def test_a_bytes_upload_lets_metadata_read_the_editor_marker(tmp_path: Path) -> None:
    path = tmp_path / "edited.jpg"
    _jpeg_with_editor_marker(path)

    result = analyze_image(path.read_bytes(), ["metadata"])

    card = result.cards[0]
    assert (card.score, card.label) == (0.70, "fake")
    assert card.details["editor_markers"]
    assert "original encoded bytes were not available" not in result.summary_markdown


def test_a_png_with_an_automatic1111_chunk_is_read_from_a_path(tmp_path: Path) -> None:
    path = tmp_path / "generated.png"
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "Steps: 20, Sampler: Euler a, Stable Diffusion")
    natural_like_image(size=(64, 48), seed=12).save(path, format="PNG", pnginfo=info)

    result = analyze_image(str(path), ["metadata"])

    card = result.cards[0]
    assert (card.score, card.label) == (0.95, "fake")
    assert card.details["ai_markers"]


def test_a_pil_upload_abstains_metadata_and_says_why_in_the_summary() -> None:
    result = analyze_image(_image(), ["metadata"])

    card = result.cards[0]
    assert (card.score, card.label) == (0.5, "uncertain")
    assert card.details["reason"] == "no encoded file available"
    assert card.caution == "abstained -- no encoded file available"
    assert (
        "This image's original encoded bytes were not available, so `metadata` and `c2pa` "
        "could not run." in result.summary_markdown
    )


def test_a_downscaled_path_upload_names_the_signals_it_cost(tmp_path: Path) -> None:
    path = tmp_path / "edited.jpg"
    _jpeg_with_editor_marker(path, size=(400, 100))

    result = analyze_image(str(path), ["metadata"], config=DemoConfig(max_side=128))

    assert result.downscaled is True
    card = result.cards[0]
    # The pixels analyzed came from a downscale, not the original file, so
    # even a JPEG carrying an editor marker abstains here.
    assert (card.score, card.label) == (0.5, "uncertain")
    assert "downscaled to 128 px" in result.summary_markdown
    assert (
        "`metadata` and `c2pa` abstain and `double_jpeg` loses the JPEG-history half"
        in result.summary_markdown
    )
    assert "original encoded bytes were not available" not in result.summary_markdown


# --------------------------------------------------------------------------
# Failures and abstentions
# --------------------------------------------------------------------------


def test_a_tool_that_raises_becomes_a_card_and_the_others_still_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(self: _ProbeSignal, image: ForensicImage) -> DetectionResult:
        raise RuntimeError("the probe fell over")

    monkeypatch.setattr(_ProbeSignal, "predict", _boom)

    result = analyze_image(_image(), [_SIGNAL_NAME, _VIEW_NAME])

    broken, view_card = result.cards
    assert (broken.label, broken.score) == ("error", 0.5)
    assert broken.elapsed_ms is None
    assert broken.details["error"] == "RuntimeError: the probe fell over"
    assert broken.caution is not None
    assert "RuntimeError: the probe fell over" in result.summary_markdown
    assert view_card.label == "uncertain"
    assert [label for label, _ in result.overlays] == [f"{_VIEW_LABEL} heatmap"]


def test_an_unregistered_tool_is_reported_rather_than_raised() -> None:
    result = analyze_image(_image(), ["no_such_tool"])

    assert result.cards[0].label == "error"
    assert "No detector registered" in result.cards[0].details["error"]


def test_an_abstaining_tool_is_named_in_the_caveats(monkeypatch: pytest.MonkeyPatch) -> None:
    def _abstain(self: _ProbeSignal, image: ForensicImage) -> DetectionResult:
        return DetectionResult(
            detector=self.name,
            score=0.5,
            label="uncertain",
            details={"reason": "nothing to read here"},
        )

    monkeypatch.setattr(_ProbeSignal, "predict", _abstain)

    result = analyze_image(_image(), [_SIGNAL_NAME])

    assert result.cards[0].caution == "abstained -- nothing to read here"
    assert f"**{_SIGNAL_LABEL}** abstained -- nothing to read here" in result.summary_markdown


# --------------------------------------------------------------------------
# The fused verdict
# --------------------------------------------------------------------------


def test_the_fused_verdict_leads_the_summary_when_a_fuser_is_configured() -> None:
    result = analyze_image(_image(), [_SIGNAL_NAME], _fitted_fuser())

    assert result.fusion is not None
    assert set(result.fusion) == {"probability", "label", "band", "contributions"}
    first_line = result.summary_markdown.splitlines()[0]
    assert first_line.startswith("**Fused verdict:")
    assert "abstain band [" in first_line


def test_there_is_no_fused_line_without_a_fuser() -> None:
    result = analyze_image(_image(), [_SIGNAL_NAME])

    assert result.fusion is None
    assert "Fused verdict" not in result.summary_markdown


# --------------------------------------------------------------------------
# The tool list
# --------------------------------------------------------------------------


def test_installed_verdict_tools_start_checked_and_views_do_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("imgforensics.demo.core.catalogue", lambda: list(_FAKE_CATALOGUE))

    by_name = {choice.name: choice for choice in tool_choices()}

    assert by_name["fake_signal"].checked is True
    assert by_name["fake_signal"].label == "JPEG \N{MIDDLE DOT} Fake signal"
    assert by_name["fake_view"].checked is False
    assert by_name["fake_view"].available is True


def test_a_tool_without_its_weights_is_listed_and_labelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("imgforensics.demo.core.catalogue", lambda: list(_FAKE_CATALOGUE))

    withheld = next(choice for choice in tool_choices() if not choice.available)

    assert withheld.name == "fake_localizer"
    assert withheld.checked is False
    assert NOT_INSTALLED_NOTE in withheld.label


def test_naming_tools_offers_only_those(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("imgforensics.demo.core.catalogue", lambda: list(_FAKE_CATALOGUE))

    choices = tool_choices(DemoConfig(tools=["fake_view"]))

    assert [choice.name for choice in choices] == ["fake_view"]


# --------------------------------------------------------------------------
# The page itself, and the CLI command that serves it
# --------------------------------------------------------------------------

gr = pytest.importorskip("gradio")

from typer.testing import CliRunner  # noqa: E402

from imgforensics.cli import app as cli_app  # noqa: E402
from imgforensics.demo.app import build_demo  # noqa: E402

runner = CliRunner()


def _components(page: Any) -> list[Any]:
    return list(page.blocks.values())


def _of_type(page: Any, component: type) -> list[Any]:
    return [block for block in _components(page) if isinstance(block, component)]


def _callback(page: Any) -> Any:
    """The one function the Analyze button is wired to."""
    functions = list(page.fns.values())
    assert len(functions) == 1
    return functions[0].fn


def test_build_demo_lays_out_one_upload_one_tool_list_and_three_outputs() -> None:
    page = build_demo(DemoConfig(tools=[_SIGNAL_NAME, _VIEW_NAME]))

    assert isinstance(page, gr.Blocks)
    (image,) = _of_type(page, gr.Image)
    # "filepath" is what hands analyze_image a path instead of decoded
    # pixels, which is what lets metadata/c2pa read the uploaded file.
    assert image.type == "filepath"
    assert len(_of_type(page, gr.Button)) == 1
    assert len(_of_type(page, gr.Gallery)) == 1
    assert len(_of_type(page, gr.JSON)) == 1
    # The title paragraph, and the summary the callback writes into.
    assert len(_of_type(page, gr.Markdown)) == 2


def test_the_checkboxes_follow_the_catalogue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("imgforensics.demo.core.catalogue", lambda: list(_FAKE_CATALOGUE))

    page = build_demo()

    offered, withheld = _of_type(page, gr.CheckboxGroup)
    assert [value for _, value in offered.choices] == ["fake_signal", "fake_view"]
    assert offered.value == ["fake_signal"]
    assert [value for _, value in withheld.choices] == ["fake_localizer"]
    assert withheld.value == []
    assert withheld.interactive is False


def test_the_button_callback_turns_one_run_into_the_pages_three_outputs(tmp_path: Path) -> None:
    page = build_demo(DemoConfig(tools=[_SIGNAL_NAME, _VIEW_NAME]))
    # The upload component is type="filepath", so the callback receives a
    # path string, never decoded pixels -- exercise it with one here.
    upload_path = tmp_path / "upload.png"
    _image().save(upload_path)

    summary, gallery, payload = _callback(page)(str(upload_path), [_SIGNAL_NAME])

    assert f"0.75 fake -- {_SIGNAL_LABEL}" in summary
    assert [caption for _, caption in gallery] == [
        f"{_SIGNAL_LABEL} heatmap",
        f"{_SIGNAL_LABEL} attribution",
    ]
    assert all(isinstance(overlay, Image.Image) for overlay, _ in gallery)
    assert [card["name"] for card in payload["cards"]] == [_SIGNAL_NAME]
    assert payload["fusion"] is None


def test_the_callback_asks_for_an_image_instead_of_reporting_an_empty_run() -> None:
    page = build_demo(DemoConfig(tools=[_SIGNAL_NAME]))

    summary, gallery, payload = _callback(page)(None, [_SIGNAL_NAME])

    assert "Upload an image" in summary
    assert (gallery, payload) == ([], {})


def test_demo_without_the_demo_extra_prints_an_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "gradio", None)

    result = runner.invoke(cli_app, ["demo"])

    assert result.exit_code == 1
    assert 'pip install "imgforensics[demo]"' in result.stdout


def test_demo_hands_the_configured_page_to_gradio_without_launching_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {}

    def _record(self: Any, **kwargs: Any) -> None:
        recorded["page"] = self
        recorded.update(kwargs)

    monkeypatch.setattr(gr.Blocks, "launch", _record)

    result = runner.invoke(
        cli_app,
        [
            "demo",
            "--host",
            "127.0.0.1",
            "--port",
            "7999",
            "--tool",
            _SIGNAL_NAME,
            "--max-side",
            "512",
            "--no-share",
        ],
    )

    assert result.exit_code == 0, result.output
    assert recorded["server_name"] == "127.0.0.1"
    assert recorded["server_port"] == 7999
    assert recorded["share"] is False
    offered = _of_type(recorded["page"], gr.CheckboxGroup)[0]
    assert [value for _, value in offered.choices] == [_SIGNAL_NAME]


def test_demo_refuses_a_tool_nobody_registered() -> None:
    result = runner.invoke(cli_app, ["demo", "--tool", "no_such_tool"])

    assert result.exit_code != 0
    assert "Unknown detector(s): no_such_tool" in result.output
