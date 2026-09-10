"""Views: maps that show what an image looks like, without claiming a verdict.

A signal answers a question ("was this JPEG compressed twice?") and pays for
the answer with a score, a benchmark row and a documented failure mode. A
*view* answers no question. It renders the image under one transform --
gradient, noise residual, a single bit plane -- and leaves the reading to the
person looking at it. That is what makes an interactive toolbox usable
(``docs/design/01_toolbox_architecture.md``, section 3.1): the tools that fire
only on some images tell you nothing about the rest, while a view always shows
something, and a trained eye finds things in it that no fixed threshold would.

The price of claiming nothing is that a view must never look like a claim.
Every view reports score 0.5 and label ``"uncertain"`` (see
:class:`~imgforensics.views.base.ViewTool`), so:

- ``analyze`` skips views unless one is named with ``--detector``; a card
  reading "0.5 uncertain" on every image, for every view, is noise.
- ``benchmark`` refuses them outright: there is no verdict to score.
- a fuser never sees one -- it lists its detectors explicitly, and a constant
  0.5 carries no information to fuse anyway.

Importing this package registers all three views with
:mod:`imgforensics.core.registry`, exactly as :mod:`imgforensics.signals`
does for the signals, and :data:`VIEW_NAMES` names them. Pure numpy and
Pillow: no torch, and no optional extra.
"""

from imgforensics.views import bitplane, luminance, noise
from imgforensics.views.base import ViewTool

#: Registry names of the view tools, sorted.
VIEW_NAMES: tuple[str, ...] = (
    "bit_planes",
    "luminance_gradient",
    "noise_residual",
)

__all__ = [
    "VIEW_NAMES",
    "ViewTool",
    "bitplane",
    "luminance",
    "noise",
]
