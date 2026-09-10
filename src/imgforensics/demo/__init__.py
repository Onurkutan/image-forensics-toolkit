"""The public stopgap demo: one Gradio page over the service layer.

``docs/design/01_toolbox_architecture.md``, milestone 6c. A Hugging Face Space
is the cheapest way to let somebody try this project without installing it,
and Gradio is the cheapest way to build one -- so the page exists, it is
deliberately thin, and it is retired once the workbench of milestone 6d is
deployed over the API.

The package is split in two so that the interesting half stays testable and
importable everywhere:

- :mod:`imgforensics.demo.core` decides what the page says -- which tools are
  offered, what one run produces, how it is summarised -- and imports no web
  framework at all.
- :mod:`imgforensics.demo.app` builds the :class:`gradio.Blocks` around it,
  and is the only module in this project that imports gradio at module scope.

Only the first half is re-exported here, so that ``import imgforensics.demo``
works on an install without the optional ``demo`` extra -- which is what lets
``imgforensics demo`` print an install hint instead of a traceback.
:func:`~imgforensics.demo.app.build_demo` is imported from its own module by
the two callers that need a page: the CLI command and ``spaces/app.py``.
"""

from imgforensics.demo.core import (
    DemoConfig,
    DemoResult,
    ToolCard,
    ToolChoice,
    analyze_image,
    downscale_for_demo,
    resolve_fuser,
    tool_choices,
)

__all__ = [
    "DemoConfig",
    "DemoResult",
    "ToolCard",
    "ToolChoice",
    "analyze_image",
    "downscale_for_demo",
    "resolve_fuser",
    "tool_choices",
]
