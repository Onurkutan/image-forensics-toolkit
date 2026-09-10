"""What the demo page shows, computed without knowing that a page exists.

Everything a person reads on the Gradio page is decided here: which tools are
offered and which of them start checked, what one run produces, and how that
run is written down. None of it imports gradio, for two reasons. The first is
testability -- a checkbox default and a summary line are worth asserting on,
and asserting on them through a web framework's component tree would test the
framework. The second is that the demo is a stopgap
(``docs/design/01_toolbox_architecture.md``, section 4): the workbench that
replaces it talks to the API instead, and the rules in this module -- what a
card says, how an overlay is drawn -- are the part that should outlive the
page they are currently drawn on.

Two decisions in here are the demo's alone and are not the library's:

- **Large uploads are downscaled.** A public CPU Space cannot afford to run
  CAT-Net over a 12-megapixel photograph, so :func:`analyze_image` shrinks
  anything past :attr:`DemoConfig.max_side` and says so in the summary. The
  library never resizes an image behind a caller's back, and a local
  ``imgforensics analyze`` still sees every pixel.
- **A failing tool is a card, not a stack trace.** The CLI lets an exception
  out, which is right for a terminal where the traceback is the useful part.
  A public page has no such reader, so a tool that raises becomes a card
  labelled ``"error"`` and the other tools still run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from imgforensics.core.image import ForensicImage
from imgforensics.core.types import DetectionResult, ToolKind
from imgforensics.fusion.explain_report import _build_overlay_image, _score_bar
from imgforensics.fusion.report import DETECTOR_NOTES, GENERIC_NOTE, fusion_payload
from imgforensics.fusion.stacking import Fuser
from imgforensics.service import AnalysisSession, catalogue
from imgforensics.service.catalogue import ToolSpec, display_name
from imgforensics.service.session import MAP_NAMES
from imgforensics.utils.jsonsafe import to_jsonable

#: Longest side an upload is analyzed at by default. Chosen for the Space's
#: CPU rather than for forensics: it is far enough above a typical web image
#: that most uploads are untouched, and far enough below a modern camera's
#: output that the slowest tool stays inside a browser's patience.
DEFAULT_MAX_SIDE = 2048

#: The score a tool reports when it has no opinion, and the value a caution
#: line looks for (see :class:`~imgforensics.views.base.ViewTool` and every
#: detector's abstain path).
ABSTAIN_SCORE = 0.5

#: Session name for an upload with no file name of its own -- raw ``bytes``,
#: or a decoded :class:`PIL.Image.Image` that never had a file behind it. A
#: path-based upload uses the file's own name instead (see
#: :func:`_forensic_image_from_input`).
UPLOAD_NAME = "upload"

#: Suffix appended to a tool that is registered but cannot run here.
NOT_INSTALLED_NOTE = "weights not installed"

DEFAULT_TITLE = "imgforensics -- image forensics demo"

DEFAULT_DESCRIPTION = (
    "Upload one image and run the [imgforensics]"
    "(https://github.com/Onurkutan/image-forensics-toolkit) tools on it: classical JPEG and "
    "noise signals, learned manipulation localizers, and a calibrated fused verdict with an "
    "abstain band.\n\n"
    "The toolkit is MIT-licensed. The pretrained weights it downloads are not -- each carries "
    "its own license, printed before anything is fetched -- and the DINOv2 AI-generation head "
    "is trained on research-only data and published separately under a research-only card. "
    "This is a research demo: no score here, fused or otherwise, is evidence about an image."
)


@dataclass
class DemoConfig:
    """How one demo page is set up.

    Attributes:
        tools: Registry names to offer, or ``None`` for the catalogue's own
            answer -- every registered tool, with the installed non-views
            checked and the views offered unchecked (see
            :func:`tool_choices`).
        fuser_path: A fitted fuser for the fused verdict. ``None`` follows the
            CLI's resolution (``$IMGFORENSICS_FUSER``, then
            ``weights/fuser.json`` when it exists), and finding none simply
            leaves the fused line out.
        max_side: Longest side an upload is analyzed at; anything larger is
            downscaled first, and the summary says so.
        title: Page title, also the browser tab's.
        description: The paragraph under the title.
    """

    tools: list[str] | None = None
    fuser_path: Path | None = None
    max_side: int = DEFAULT_MAX_SIDE
    title: str = DEFAULT_TITLE
    description: str = DEFAULT_DESCRIPTION


@dataclass(frozen=True)
class ToolChoice:
    """One entry of the tool list, ready to draw.

    Attributes:
        name: The registry name, which is what a run is asked for.
        label: What the checkbox says -- the category and the display name,
            plus a note when the tool cannot run here.
        checked: Whether it starts selected.
        available: Whether it can produce a result on this machine. A tool
            that cannot is still listed, because a missing tool looks like a
            tool this project does not have.
    """

    name: str
    label: str
    checked: bool
    available: bool


@dataclass(frozen=True)
class ToolCard:
    """One tool's result, as the page states it.

    Attributes:
        name: The registry name.
        display_name: The name shown to a person.
        kind: Signal, detector, localizer or view.
        score: Probability that the image is generated or manipulated; 0.5
            for a view, and for a tool that abstained or failed.
        label: ``real`` / ``fake`` / ``uncertain``, or ``error`` for a tool
            that raised.
        elapsed_ms: Wall-clock time of the run, or ``None`` when there was no
            result to time.
        note: The plain-language line shared with the report
            (:data:`~imgforensics.fusion.report.DETECTOR_NOTES`).
        caution: Why this card should be read carefully -- an abstention with
            its reason, or the exception a failing tool raised. ``None`` when
            the tool simply answered.
        details: The tool's own numbers, JSON-safe.
    """

    name: str
    display_name: str
    kind: ToolKind
    score: float
    label: str
    elapsed_ms: float | None
    note: str
    caution: str | None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DemoResult:
    """Everything one press of the Analyze button produced.

    Attributes:
        cards: One per selected tool, in the order they were asked for.
        overlays: ``(label, image)`` per map produced, drawn by the report
            builder's own overlay function so the page shows exactly what
            ``analyze --report-dir`` would write.
        fusion: The fused-verdict payload, or ``None`` when no fuser is
            configured.
        summary_markdown: The verdict, the per-tool lines and the caveats.
        downscaled: Whether the upload was shrunk before analysis.
    """

    cards: list[ToolCard]
    overlays: list[tuple[str, Image.Image]]
    fusion: dict[str, Any] | None
    summary_markdown: str
    downscaled: bool


def resolve_fuser(fuser_path: Path | None) -> Fuser | None:
    """The fuser the page fuses with, resolved the way every other entry point does.

    The CLI's resolver is imported inside the function so that the rule --
    an explicit path, then ``$IMGFORENSICS_FUSER``, then ``weights/fuser.json``
    when it exists -- has one implementation rather than a copy per front end.

    Raises:
        FileNotFoundError: if a path was given, or found in the environment,
            but nothing is there. A page that was asked for a fuser and
            cannot load one should say so while it is being built, not once
            per upload.
    """
    from imgforensics.cli import _resolve_fuser_path

    path = _resolve_fuser_path(fuser_path)
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Fuser file not found: {path}")
    return Fuser.load(path)


def tool_choices(config: DemoConfig | None = None) -> list[ToolChoice]:
    """The tool list the page draws, in the catalogue's order.

    Three rules, all of them about what a first-time visitor should see
    happen when they press Analyze without touching anything:

    - An installed tool that claims a verdict starts checked. That is the
      run the page is for.
    - A view starts unchecked. A view renders a map and claims nothing
      (:mod:`imgforensics.views`), so it is worth offering and not worth
      spending a CPU Space's seconds on by default -- the same reason
      ``analyze`` skips the views.
    - A tool whose weights are missing is listed, labelled, and cannot be
      selected. Hiding it would misrepresent what the toolkit has.
    """
    config = config if config is not None else DemoConfig()
    wanted = None if config.tools is None else set(config.tools)
    choices: list[ToolChoice] = []
    for spec in catalogue():
        if wanted is not None and spec.name not in wanted:
            continue
        label = f"{spec.category} \N{MIDDLE DOT} {spec.display_name}"
        if not spec.installed:
            label = f"{label} ({NOT_INSTALLED_NOTE})"
        choices.append(
            ToolChoice(
                name=spec.name,
                label=label,
                checked=spec.installed and spec.kind != "view",
                available=spec.installed,
            )
        )
    return choices


def downscale_for_demo(image: Image.Image, max_side: int) -> tuple[Image.Image, bool]:
    """Shrink ``image`` so its longer side is at most ``max_side``, if it is not already.

    Returns:
        ``(image, was_downscaled)`` -- the original object and ``False`` when
        nothing needed doing, so a small upload is analyzed byte for byte.

    Raises:
        ValueError: if ``max_side`` is below 1.
    """
    if max_side < 1:
        raise ValueError(f"max_side must be at least 1, got {max_side}")
    long_side = max(image.size)
    if long_side <= max_side:
        return image, False
    scale = max_side / long_side
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS), True


def _forensic_image_from_input(
    image: str | Path | bytes | Image.Image,
) -> tuple[ForensicImage, str]:
    """The :class:`ForensicImage` to analyze, and the session name it gets.

    A path or ``bytes`` keeps the original encoded file, so ``metadata``,
    ``c2pa`` and the JPEG-history half of ``double_jpeg`` -- every signal
    that reads :attr:`ForensicImage.raw` -- can run against it. An
    already-decoded :class:`PIL.Image.Image` carries no such file, so those
    signals abstain (see :func:`_pil_only_caveat`).
    """
    if isinstance(image, bytes):
        return ForensicImage.from_bytes(image, path=None), UPLOAD_NAME
    if isinstance(image, str | Path):
        path = Path(image)
        return ForensicImage.from_path(path), path.name
    return ForensicImage.from_pil(image), UPLOAD_NAME


def analyze_image(
    image: str | Path | bytes | Image.Image,
    selected_tools: Sequence[str],
    fuser: Fuser | None = None,
    config: DemoConfig | None = None,
) -> DemoResult:
    """Run the selected tools on one uploaded image and write down what they said.

    Args:
        image: The upload to analyze. A path or ``bytes`` is read as an
            encoded file, giving the raw-bytes signals something to read; a
            decoded :class:`PIL.Image.Image` has no such file, and the
            summary says so. Downscaled first when the longer side is larger
            than ``config.max_side`` -- past that point the analysis runs on
            shrunk pixels with no encoded original behind them either,
            whatever ``image`` was, and the summary names what that costs.
        selected_tools: Registry names, run in the order given. A name
            nothing is registered under yields an error card, like any other
            failure.
        fuser: A fitted fuser, or ``None`` to leave the fused verdict out.
        config: The page's settings; defaults to :class:`DemoConfig`.

    Returns:
        A :class:`DemoResult`. It always has one card per selected tool: a
        tool that raises is reported, never re-raised, so one broken tool
        cannot take the page down with it.
    """
    config = config if config is not None else DemoConfig()
    forensic_image, session_name = _forensic_image_from_input(image)
    pil_only = forensic_image.raw is None
    prepared, downscaled = downscale_for_demo(forensic_image.rgb, config.max_side)
    if downscaled:
        forensic_image = ForensicImage.from_pil(prepared)
    session = AnalysisSession(forensic_image, name=session_name)
    specs = {spec.name: spec for spec in catalogue()}

    cards: list[ToolCard] = []
    overlays: list[tuple[str, Image.Image]] = []
    for name in selected_tools:
        spec = specs.get(name)
        try:
            result = session.run(name)
        except Exception as exc:
            # Deliberately broad: what a tool can raise is the tool's business
            # (a missing file, a torch error, an unregistered name), and none
            # of it is a reason to hand the visitor a blank page.
            cards.append(_error_card(name, spec, exc))
            continue
        cards.append(_result_card(name, spec, result))
        overlays += _overlays(session.image.rgb, result, _display_name(name, spec))

    fusion = to_jsonable(fusion_payload(fuser, session.results())) if fuser is not None else None
    return DemoResult(
        cards=cards,
        overlays=overlays,
        fusion=fusion,
        summary_markdown=_summary_markdown(cards, fusion, downscaled, pil_only, config.max_side),
        downscaled=downscaled,
    )


def _display_name(name: str, spec: ToolSpec | None) -> str:
    """The label for a tool, whether or not the catalogue knows it."""
    return spec.display_name if spec is not None else display_name(name)


def _kind(spec: ToolSpec | None) -> ToolKind:
    """A tool's kind; a name the catalogue does not carry is described as a signal.

    That is the kind :class:`~imgforensics.core.base.BaseDetector` gives a
    tool which declares none, so it is the least surprising thing to say
    about a tool nobody registered either.
    """
    return spec.kind if spec is not None else "signal"


def _caution(result: DetectionResult) -> str | None:
    """Why a card should be read carefully, or ``None`` when it should not.

    A tool abstains by scoring exactly 0.5 and saying why, which every
    front end shows verbatim. A view also scores 0.5 but gives no reason,
    and gets no caution: claiming nothing is what a view is for.
    """
    reason = result.details.get("reason")
    if result.score == ABSTAIN_SCORE and reason:
        return f"abstained -- {reason}"
    return None


def _result_card(name: str, spec: ToolSpec | None, result: DetectionResult) -> ToolCard:
    """One tool's answer as a card."""
    return ToolCard(
        name=name,
        display_name=_display_name(name, spec),
        kind=_kind(spec),
        score=result.score,
        label=result.label,
        elapsed_ms=result.elapsed_ms,
        note=spec.note if spec is not None else DETECTOR_NOTES.get(name, GENERIC_NOTE),
        caution=_caution(result),
        details=to_jsonable(result.details),
    )


def _error_card(name: str, spec: ToolSpec | None, exc: Exception) -> ToolCard:
    """One tool's failure as a card: what broke, in the words the exception used.

    The score is 0.5 because that is this project's spelling of "no opinion",
    and a failure is the strongest possible version of one.
    """
    message = f"{type(exc).__name__}: {exc}"
    return ToolCard(
        name=name,
        display_name=_display_name(name, spec),
        kind=_kind(spec),
        score=ABSTAIN_SCORE,
        label="error",
        elapsed_ms=None,
        note=spec.note if spec is not None else DETECTOR_NOTES.get(name, GENERIC_NOTE),
        caution=f"failed -- {message}",
        details={"error": message},
    )


def _overlays(
    rgb: Image.Image, result: DetectionResult, display: str
) -> list[tuple[str, Image.Image]]:
    """The colour-mapped overlays one result carries, labelled by map.

    Built with the report builder's own overlay function rather than a second
    renderer, so a map on this page and the same map in ``--report-dir``'s
    folder are the same picture -- including its 1,024 px cap, which is why
    the gallery stays light even when the analysis ran at 2,048.
    """
    overlays: list[tuple[str, Image.Image]] = []
    for map_name in MAP_NAMES:
        array = getattr(result, map_name)
        if array is None:
            continue
        overlay, _ = _build_overlay_image(rgb, array)
        overlays.append((f"{display} {map_name}", overlay))
    return overlays


def _tool_line(card: ToolCard) -> str:
    """One tool's line of the summary: a bar, the number, the label, the name."""
    suffix = " (view -- no verdict)" if card.kind == "view" else ""
    return (
        f"- `{_score_bar(card.score)}` {card.score:.2f} {card.label} -- {card.display_name}{suffix}"
    )


#: The signals that need :attr:`ForensicImage.raw` to run at all -- named
#: here once, for both caveats below, rather than spelled out twice.
_RAW_ONLY_SIGNALS = "`metadata` and `c2pa`"


def _pil_only_caveat() -> str:
    """The line that admits the upload never had an encoded file behind it.

    Shown for a decoded :class:`PIL.Image.Image` passed straight to
    :func:`analyze_image`, whether or not it also needed downscaling: the
    raw-bytes signals were already unreachable before that decision was
    made.
    """
    return (
        "This image's original encoded bytes were not available, so "
        f"{_RAW_ONLY_SIGNALS} could not run."
    )


def _downscale_caveat(max_side: int, *, name_lost_signals: bool) -> str:
    """The line that admits the page did not read the pixels it was given.

    ``name_lost_signals`` is ``False`` when :func:`_pil_only_caveat` already
    covers why the raw-bytes signals did not run -- the upload never had an
    encoded file to lose. Otherwise the upload arrived as a path or ``bytes``
    and downscaling is what cost it that file, which this line says.
    """
    base = (
        f"The upload was downscaled to {max_side} px on its longest side before analysis. "
        "This demo does that so a CPU machine can answer in a few seconds; the library "
        "itself never resizes, so a local run reads the original pixels."
    )
    if not name_lost_signals:
        return base
    return (
        f"{base} The pixels analyzed no longer match the original file, so "
        f"{_RAW_ONLY_SIGNALS} abstain and `double_jpeg` loses the JPEG-history half of "
        "its check."
    )


def _summary_markdown(
    cards: Sequence[ToolCard],
    fusion: dict[str, Any] | None,
    downscaled: bool,
    pil_only: bool,
    max_side: int,
) -> str:
    """The verdict, then the tools, then what to be careful about.

    The order is deliberate: whoever reads this page reads the top of it, so
    the fused verdict and its band go first, the per-tool bars next, and the
    reasons not to trust a line -- an abstention, a failure, a missing file,
    a resize -- last, where they sit under the numbers they qualify rather
    than above them.
    """
    lines: list[str] = []
    if fusion is not None:
        band = fusion["band"]
        lines += [
            f"**Fused verdict: {fusion['label']}** -- probability "
            f"{float(fusion['probability']):.2f}, abstain band "
            f"[{float(band['low']):.2f}, {float(band['high']):.2f}].",
            "",
        ]

    if cards:
        lines += [_tool_line(card) for card in cards]
    else:
        lines.append("No tools were selected, so there is nothing to report.")

    caveats = [f"**{card.display_name}** {card.caution}" for card in cards if card.caution]
    if downscaled:
        caveats.insert(0, _downscale_caveat(max_side, name_lost_signals=not pil_only))
    if pil_only:
        caveats.insert(0, _pil_only_caveat())
    if caveats:
        lines += ["", "**Caveats**", ""] + [f"- {caveat}" for caveat in caveats]

    return "\n".join(lines) + "\n"
