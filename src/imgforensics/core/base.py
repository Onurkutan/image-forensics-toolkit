"""Abstract base class that every detector implementation must follow."""

from __future__ import annotations

import abc
import time
from collections.abc import Sequence

from imgforensics.core.image import ForensicImage
from imgforensics.core.types import DetectionResult


class BaseDetector(abc.ABC):
    """Contract for an image forensics detector.

    Implementations receive a :class:`~imgforensics.core.image.ForensicImage`
    (decoded RGB pixels plus, when available, the original encoded bytes) and
    return a :class:`~imgforensics.core.types.DetectionResult` whose ``score``
    is the probability that the image is AI-generated, AI-inpainted, or
    otherwise manipulated (0 = confidently real, 1 = confidently fake).
    Subclasses must set the ``name`` class attribute and implement
    :meth:`predict`.
    """

    name: str

    def load(self, device: str = "cpu") -> None:
        """Load any resources (weights, models) needed for inference.

        The default implementation is a no-op; override for detectors that
        require loading model weights onto a device.
        """
        return None

    @abc.abstractmethod
    def predict(self, image: ForensicImage) -> DetectionResult:
        """Run the detector on a single image and return its result.

        Implementations set ``score``, ``label``, ``heatmap`` (if any) and
        ``details``. ``elapsed_ms`` is left unset here; :meth:`run` fills it
        in with the wall-clock time of this call.
        """

    def predict_batch(self, images: Sequence[ForensicImage]) -> list[DetectionResult]:
        """Run the detector on a sequence of images by looping over :meth:`predict`."""
        return [self.predict(image) for image in images]

    def run(self, image: ForensicImage) -> DetectionResult:
        """Run :meth:`predict` and fill in ``elapsed_ms`` with the wall-clock time.

        This is the entry point callers (such as the CLI) should use instead
        of calling :meth:`predict` directly, so timing is measured
        consistently across all detectors.
        """
        start = time.perf_counter()
        result = self.predict(image)
        result.elapsed_ms = (time.perf_counter() - start) * 1000
        return result
