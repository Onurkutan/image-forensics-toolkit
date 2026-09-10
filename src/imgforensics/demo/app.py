"""The Gradio page itself: one upload, one tool list, one button, three outputs.

This is the public stopgap demo the design note schedules before the
workbench (``docs/design/01_toolbox_architecture.md``, section 4 and milestone
6c). It is deliberately thin, and thin is the whole point: Gradio cannot give
synchronized pan and zoom across several maps, per-tool parameter panels, or
tiled views of a 12-megapixel image, and pretending otherwise here would build
a second, worse workbench that the real one would then have to replace. So the
page does the one thing Gradio is good at -- a link anybody can open, showing
what the toolkit says about an image they brought -- and stops.

Everything that decides *what* is shown lives in
:mod:`imgforensics.demo.core`, which knows nothing about gradio. This module
is the translation: components in, that module's :class:`DemoResult` out,
converted into the three outputs the page renders. It is the only module in
the project that imports gradio at module scope, which is what keeps the
optional ``demo`` extra optional.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import gradio as gr
from PIL import Image

from imgforensics.demo.core import (
    DemoConfig,
    ToolChoice,
    analyze_image,
    resolve_fuser,
    tool_choices,
)

#: Shown when Analyze is pressed with no image. A Gradio input can be empty,
#: and an empty summary would read like a run that found nothing.
NO_IMAGE_MARKDOWN = "Upload an image first, then press **Analyze**."

#: The hint under the tools that cannot run here. It names the command that
#: fixes it, since the answer to a greyed-out localizer is one line long.
FETCH_HINT = (
    "Registered but not runnable on this machine. Fetch them with "
    "'imgforensics weights fetch <name> --accept-license'."
)

_TOOLS_HINT = (
    "Views draw a map and claim no verdict, so they start unchecked. "
    "Everything else installed here starts on."
)


def build_demo(config: DemoConfig | None = None) -> gr.Blocks:
    """Build the demo page.

    The fuser is resolved once, here, rather than per upload: whether this
    page can show a fused verdict is a fact about how it was started, and a
    page that finds no fuser simply leaves that line out.

    Args:
        config: The page's settings; defaults to :class:`DemoConfig`, which
            offers the whole catalogue and takes the fuser from
            ``$IMGFORENSICS_FUSER`` or ``weights/fuser.json``.

    Returns:
        An unlaunched :class:`gradio.Blocks`. Launching it is the caller's
        job -- ``imgforensics demo`` for a local run, ``spaces/app.py`` for
        the Hugging Face Space.

    Raises:
        FileNotFoundError: if a fuser was named but no file is there.
    """
    config = config if config is not None else DemoConfig()
    fuser = resolve_fuser(config.fuser_path)
    choices = tool_choices(config)
    offered = [choice for choice in choices if choice.available]
    withheld = [choice for choice in choices if not choice.available]

    def _run(
        uploaded: str | None, names: list[str]
    ) -> tuple[str, list[tuple[Image.Image, str]], dict[str, Any]]:
        """Analyze the upload and lay the result out as the page's three outputs.

        ``uploaded`` is a filesystem path, not decoded pixels: the component
        below is ``type="filepath"`` precisely so :func:`analyze_image` gets
        the encoded file and ``metadata``/``c2pa`` have something to read.
        """
        if uploaded is None:
            return NO_IMAGE_MARKDOWN, [], {}
        result = analyze_image(uploaded, list(names), fuser, config)
        return (
            result.summary_markdown,
            # The gallery wants (image, caption); the result carries the
            # caption first, because a label is what identifies an overlay.
            [(overlay, label) for label, overlay in result.overlays],
            {"cards": [asdict(card) for card in result.cards], "fusion": result.fusion},
        )

    with gr.Blocks(title=config.title) as demo:
        gr.Markdown(f"# {config.title}\n\n{config.description}")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(label="Image", type="filepath", sources=["upload", "clipboard"])
                selected = gr.CheckboxGroup(
                    choices=_labelled(offered),
                    value=[choice.name for choice in offered if choice.checked],
                    label="Tools",
                    info=_TOOLS_HINT,
                )
                if withheld:
                    gr.CheckboxGroup(
                        choices=_labelled(withheld),
                        value=[],
                        label="Not available here",
                        info=FETCH_HINT,
                        interactive=False,
                    )
                analyze = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Markdown(label="Summary")
                gallery = gr.Gallery(label="Overlays", columns=2, object_fit="contain")
                payload = gr.JSON(label="Cards and fused verdict")

        analyze.click(_run, inputs=[image, selected], outputs=[summary, gallery, payload])

    return demo


def _labelled(choices: list[ToolChoice]) -> list[tuple[str, str]]:
    """Tool choices as the ``(label, value)`` pairs a ``CheckboxGroup`` takes."""
    return [(choice.label, choice.name) for choice in choices]
