"""``ForensicImage``: the input type shared by every detector.

Decoded pixels alone throw away information a forensic signal may need (the
original encoded bytes, the container format, embedded metadata). This module
wraps a decoded RGB image together with its original encoding so detectors can
choose which one to look at.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from imgforensics.utils.image_io import load_image


@dataclass(frozen=True)
class ForensicImage:
    """A single image as seen by the forensic pipeline.

    Attributes:
        rgb: Upright RGB pixels, EXIF-transposed (see
            :func:`imgforensics.utils.image_io.load_image`). Always available.
        raw: The original encoded bytes (JPEG/PNG/... file content) when the
            image was loaded from a path or from bytes. ``None`` when built
            from an already-decoded :class:`PIL.Image.Image` via
            :meth:`from_pil`, in which case no encoded file exists to inspect.
        path: The source path, when known.
        format: The PIL format name of the original file ("JPEG", "PNG", ...),
            or ``None`` when it cannot be determined.
    """

    rgb: Image.Image
    raw: bytes | None
    path: Path | None
    format: str | None

    @classmethod
    def from_path(cls, path: str | Path) -> ForensicImage:
        """Load a :class:`ForensicImage` from a file path, keeping the raw bytes."""
        path = Path(path)
        raw = path.read_bytes()
        return cls.from_bytes(raw, path=path)

    @classmethod
    def from_bytes(cls, data: bytes, path: Path | None = None) -> ForensicImage:
        """Load a :class:`ForensicImage` from an in-memory encoded file."""
        rgb = load_image(data)
        with Image.open(io.BytesIO(data)) as original:
            fmt = original.format
        return cls(rgb=rgb, raw=data, path=path, format=fmt)

    @classmethod
    def from_pil(cls, image: Image.Image) -> ForensicImage:
        """Wrap an already-decoded :class:`PIL.Image.Image`.

        No encoded original is available in this case, so :attr:`raw` is
        ``None`` and :meth:`open_original` will raise.
        """
        fmt = image.format
        rgb = load_image(image)
        return cls(rgb=rgb, raw=None, path=None, format=fmt)

    def open_original(self) -> Image.Image:
        """Re-open the original encoded file, with its metadata intact.

        Unlike :attr:`rgb`, the returned image is not EXIF-transposed or
        converted to RGB, so EXIF/XMP/ICC and other metadata survive
        untouched. Returns a fresh :class:`PIL.Image.Image` on every call.

        Raises:
            ValueError: if :attr:`raw` is ``None`` (the image was built from
                a decoded :class:`PIL.Image.Image`, so no encoded original
                exists).
        """
        if self.raw is None:
            raise ValueError(
                "no encoded original bytes available for this ForensicImage "
                "(it was built from an already-decoded PIL.Image.Image)"
            )
        return Image.open(io.BytesIO(self.raw))

    @property
    def width(self) -> int:
        return self.rgb.width

    @property
    def height(self) -> int:
        return self.rgb.height
