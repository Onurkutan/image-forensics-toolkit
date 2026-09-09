"""Tests for imgforensics.core.image.ForensicImage."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from imgforensics.core.image import ForensicImage


def _make_jpeg_bytes_with_exif() -> bytes:
    image = Image.new("RGB", (16, 12), color=(200, 50, 10))
    exif = Image.Exif()
    exif[0x010F] = "Canon"  # Make
    exif[0x0110] = "EOS R5"  # Model
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def test_from_path(tmp_path: Path) -> None:
    data = _make_jpeg_bytes_with_exif()
    jpeg_path = tmp_path / "photo.jpg"
    jpeg_path.write_bytes(data)

    fi = ForensicImage.from_path(jpeg_path)

    assert fi.rgb.mode == "RGB"
    assert fi.rgb.size == (16, 12)
    assert fi.raw == data
    assert fi.path == jpeg_path
    assert fi.format == "JPEG"
    assert fi.width == 16
    assert fi.height == 12


def test_from_bytes() -> None:
    data = _make_jpeg_bytes_with_exif()

    fi = ForensicImage.from_bytes(data)

    assert fi.raw == data
    assert fi.path is None
    assert fi.format == "JPEG"
    assert fi.rgb.size == (16, 12)


def test_from_bytes_with_explicit_path(tmp_path: Path) -> None:
    data = _make_jpeg_bytes_with_exif()
    fake_path = tmp_path / "elsewhere.jpg"

    fi = ForensicImage.from_bytes(data, path=fake_path)

    assert fi.path == fake_path
    assert fi.raw == data


def test_from_pil() -> None:
    source = Image.new("RGB", (5, 7), color="white")

    fi = ForensicImage.from_pil(source)

    assert fi.raw is None
    assert fi.path is None
    assert fi.rgb.mode == "RGB"
    assert fi.rgb.size == (5, 7)
    assert fi.width == 5
    assert fi.height == 7


def test_open_original_preserves_exif() -> None:
    data = _make_jpeg_bytes_with_exif()
    fi = ForensicImage.from_bytes(data)

    original = fi.open_original()

    exif = original.getexif()
    assert exif.get(0x010F) == "Canon"
    assert exif.get(0x0110) == "EOS R5"
    # Unlike `rgb`, the original is not forced to RGB mode by us.
    assert original.format == "JPEG"


def test_open_original_raises_when_raw_is_none() -> None:
    fi = ForensicImage.from_pil(Image.new("RGB", (4, 4)))

    with pytest.raises(ValueError, match="no encoded original bytes"):
        fi.open_original()


def test_open_original_returns_fresh_image_each_call() -> None:
    data = _make_jpeg_bytes_with_exif()
    fi = ForensicImage.from_bytes(data)

    first = fi.open_original()
    second = fi.open_original()

    assert first is not second
