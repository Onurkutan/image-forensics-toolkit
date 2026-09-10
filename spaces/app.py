"""Hugging Face Space entry point: the demo page, configured by the environment.

Nothing is passed to :class:`~imgforensics.demo.DemoConfig` on purpose. The
two settings a Space actually varies are already environment variables the
library reads for itself -- ``IMGFORENSICS_FUSER`` for the fused verdict and
``IMGFORENSICS_HEAD_DIR`` for the AI-generation head's checkpoint -- so the
Space's configuration lives in its own settings panel rather than in a fork of
this file. See ``spaces/README.md`` for what the build fetches and why.
"""

from imgforensics.demo import DemoConfig
from imgforensics.demo.app import build_demo

demo = build_demo(DemoConfig())

if __name__ == "__main__":
    demo.launch()
