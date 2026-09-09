"""Tests for imgforensics.signals.metadata.MetadataSignal."""

from __future__ import annotations

import io

import piexif
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from imgforensics.core.image import ForensicImage
from imgforensics.signals.metadata import MetadataSignal


def _jpeg_bytes(
    *,
    make: str | None = None,
    model: str | None = None,
    software: str | None = None,
) -> bytes:
    image = Image.new("RGB", (32, 32), color=(10, 20, 30))
    exif = Image.Exif()
    if make is not None:
        exif[0x010F] = make
    if model is not None:
        exif[0x0110] = model
    if software is not None:
        exif[0x0131] = software
    buffer = io.BytesIO()
    if make is None and model is None and software is None:
        image.save(buffer, format="JPEG")
    else:
        image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def test_editor_marker_gives_fake_score() -> None:
    data = _jpeg_bytes(make="Canon", model="EOS", software="Adobe Photoshop 25.0")
    fi = ForensicImage.from_bytes(data)

    result = MetadataSignal().predict(fi)

    assert result.score == pytest.approx(0.70)
    assert result.label == "fake"
    assert any("Photoshop" in marker for marker in result.details["editor_markers"])


def test_camera_exif_without_editor_marker_gives_real_score() -> None:
    data = _jpeg_bytes(make="Canon", model="EOS")
    fi = ForensicImage.from_bytes(data)

    result = MetadataSignal().predict(fi)

    assert result.score == pytest.approx(0.30)
    assert result.label == "real"
    assert result.details["camera_make"] == "Canon"
    assert result.details["editor_markers"] == []


def test_png_ai_generator_marker_gives_fake_score() -> None:
    image = Image.new("RGB", (16, 16), color=(1, 2, 3))
    pnginfo = PngInfo()
    pnginfo.add_text("parameters", "masterpiece, 1girl, Steps: 20, Sampler: Euler")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", pnginfo=pnginfo)
    fi = ForensicImage.from_bytes(buffer.getvalue())

    result = MetadataSignal().predict(fi)

    assert result.score == pytest.approx(0.95)
    assert result.label == "fake"
    assert result.details["ai_markers"]


def test_plain_jpeg_without_exif_is_uncertain() -> None:
    data = _jpeg_bytes()
    fi = ForensicImage.from_bytes(data)

    result = MetadataSignal().predict(fi)

    assert result.score == pytest.approx(0.50)
    assert result.label == "uncertain"
    assert result.details["has_exif"] is False
    assert "note" in result.details


def test_thumbnail_mismatch_gives_fake_score() -> None:
    main_image = Image.new("RGB", (200, 200), color=(255, 0, 0))
    main_buffer = io.BytesIO()
    main_image.save(main_buffer, format="JPEG", quality=90)

    # Thumbnail from an unrelated, differently colored image.
    other_thumb = Image.new("RGB", (64, 64), color=(0, 255, 0))
    thumb_buffer = io.BytesIO()
    other_thumb.save(thumb_buffer, format="JPEG", quality=90)

    exif_dict = {
        "0th": {piexif.ImageIFD.Make: b"Canon"},
        "Exif": {},
        "1st": {piexif.ImageIFD.Make: b"Canon"},
        "thumbnail": thumb_buffer.getvalue(),
        "GPS": {},
    }
    exif_bytes = piexif.dump(exif_dict)

    out_buffer = io.BytesIO()
    main_image.save(out_buffer, format="JPEG", quality=90, exif=exif_bytes)
    fi = ForensicImage.from_bytes(out_buffer.getvalue())

    result = MetadataSignal().predict(fi)

    assert result.details["thumbnail_mismatch"] is True
    assert result.score == pytest.approx(0.85)
    assert result.label == "fake"
    assert result.details["thumbnail_ncc"] is not None
    assert result.details["thumbnail_mad"] is not None


def test_from_pil_abstains() -> None:
    fi = ForensicImage.from_pil(Image.new("RGB", (8, 8)))

    result = MetadataSignal().predict(fi)

    assert result.score == pytest.approx(0.5)
    assert result.label == "uncertain"
    assert result.details["reason"] == "no encoded file available"


def test_jpeg_quality_estimate_within_tolerance() -> None:
    image = Image.new("RGB", (64, 64), color=(120, 130, 140))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=75)
    fi = ForensicImage.from_bytes(buffer.getvalue())

    result = MetadataSignal().predict(fi)

    estimate = result.details["jpeg_quality_estimate"]
    assert estimate is not None
    assert abs(estimate - 75) <= 10
    assert result.details["jpeg_quant_tables_count"] is not None


def test_ai_marker_guard_rejects_ordinary_words() -> None:
    """A Software tag / comment containing ordinary words that happen to
    contain AI-tool substrings ("Dallas" contains "dall", "imagenes"
    contains "imagen") must not be mistaken for AI-generator markers."""
    image = Image.new("RGB", (16, 16), color=(4, 5, 6))
    exif = Image.Exif()
    exif[0x0131] = "Dallas Photo Club"
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif, comment=b"influx of imagenes")
    fi = ForensicImage.from_bytes(buffer.getvalue())

    result = MetadataSignal().predict(fi)

    assert result.details["ai_markers"] == []
    assert result.score == pytest.approx(0.30) or result.score == pytest.approx(0.50)


_XMP_TEMPLATE = (
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
    b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    b'<rdf:Description rdf:about=""'
    b' xmlns:iptcExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"'
    b' iptcExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/{}"/>'
    b"</rdf:RDF></x:xmpmeta>"
)


def test_ai_marker_xmp_trained_algorithmic_media_gives_fake_score() -> None:
    image = Image.new("RGB", (16, 16), color=(7, 8, 9))
    xmp = _XMP_TEMPLATE.replace(b"{}", b"trainedAlgorithmicMedia")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", xmp=xmp)
    fi = ForensicImage.from_bytes(buffer.getvalue())

    result = MetadataSignal().predict(fi)

    assert any(
        "IPTC digitalSourceType algorithmicMedia" in marker
        for marker in result.details["ai_markers"]
    )
    assert result.score == pytest.approx(0.95)
    assert result.label == "fake"


def test_ai_marker_xmp_digital_capture_not_flagged() -> None:
    image = Image.new("RGB", (16, 16), color=(11, 12, 13))
    xmp = _XMP_TEMPLATE.replace(b"{}", b"digitalCapture")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", xmp=xmp)
    fi = ForensicImage.from_bytes(buffer.getvalue())

    result = MetadataSignal().predict(fi)

    assert result.details["ai_markers"] == []


def test_png_has_no_jpeg_quality_estimate() -> None:
    image = Image.new("RGB", (16, 16), color=(9, 9, 9))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    fi = ForensicImage.from_bytes(buffer.getvalue())

    result = MetadataSignal().predict(fi)

    assert result.details["jpeg_quality_estimate"] is None
    assert result.details["jpeg_quant_tables_count"] is None
