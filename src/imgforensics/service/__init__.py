"""The headless service layer: sessions, the tool catalogue, and map tiles.

This is the layer a user interface talks to and the layer that never talks
back to one (``docs/design/01_toolbox_architecture.md``, section 3.1). It adds
no forensics of its own: every number and every map it hands out came from a
tool in the registry, through the same
:class:`~imgforensics.core.types.DetectionResult` the CLI prints and the
benchmark runner scores. What it adds is the state and the shapes an
interactive client needs and a one-shot command does not -- a session that
remembers what has already been run, parameters validated before a tool is
built, and maps served as pyramid tiles instead of as one 48 MB array.

Importing this package registers every tool that this install has: the
signals and the views always, the learned detector and the localizers when the
optional ``ml`` extra is present. So :func:`~imgforensics.service.catalogue`
answers correctly on a bare ``import imgforensics.service``, with no ordering
rule for the caller to remember.

The FastAPI wrapper (milestone 6b) is a separate package that imports this
one; nothing here knows what HTTP is.
"""

# Imported for their registration side effects only, so that a catalogue built
# straight after "import imgforensics.service" is complete.
from imgforensics import detectors, localization, signals, views  # noqa: F401
from imgforensics.service.catalogue import ToolSpec, catalogue
from imgforensics.service.session import (
    AnalysisSession,
    MapPyramid,
    SessionStore,
    to_png,
)

__all__ = [
    "AnalysisSession",
    "MapPyramid",
    "SessionStore",
    "ToolSpec",
    "catalogue",
    "to_png",
]
