"""Abstract base class that every detector implementation must follow."""

from __future__ import annotations

import abc
from collections.abc import Sequence

from PIL import Image

from imgforensics.core.types import DetectionResult


class BaseDetector(abc.ABC):
    """Contract for an image forensics detector.

    Implementations receive an RGB :class:`PIL.Image.Image` and return a
    :class:`~imgforensics.core.types.DetectionResult` whose ``score`` is the
    probability that the image is AI-generated, AI-inpainted, or otherwise
    manipulated (0 = confidently real, 1 = confidently fake). Subclasses must
    set the ``name`` class attribute and implement :meth:`predict`.
    """

    name: str

    def load(self, device: str = "cpu") -> None:
        """Load any resources (weights, models) needed for inference.

        The default implementation is a no-op; override for detectors that
        require loading model weights onto a device.
        """
        return None

    @abc.abstractmethod
    def predict(self, image: Image.Image) -> DetectionResult:
        """Run the detector on a single RGB image and return its result."""

    def predict_batch(self, images: Sequence[Image.Image]) -> list[DetectionResult]:
        """Run the detector on a sequence of images by looping over :meth:`predict`."""
        return [self.predict(image) for image in images]
