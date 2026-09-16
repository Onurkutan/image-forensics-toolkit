"""Tests for imgforensics.signals.metadata.MetadataSignal."""

from __future__ import annotations

import io
import struct

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


def _irb(resource_id: int, data: bytes, name: bytes = b"") -> bytes:
    """One Photoshop image-resource block ("8BIM"), padded to an even length."""
    block = b"8BIM" + struct.pack(">H", resource_id) + bytes([len(name)]) + name
    if len(block) % 2:
        block += b"\x00"
    block += struct.pack(">I", len(data)) + data
    if len(data) % 2:
        block += b"\x00"
    return block


def _iptc(record: int, dataset: int, value: bytes) -> bytes:
    """One IPTC-IIM dataset, as it appears inside the 1028 resource block."""
    return b"\x1c" + bytes([record, dataset]) + struct.pack(">H", len(value)) + value


def _with_app13(jpeg: bytes, resources: dict[int, bytes]) -> bytes:
    """Splice an APP13 ("Photoshop 3.0") segment holding ``resources`` into ``jpeg``."""
    payload = b"Photoshop 3.0\x00" + b"".join(_irb(rid, data) for rid, data in resources.items())
    segment = b"\xff\xed" + struct.pack(">H", len(payload) + 2) + payload
    assert jpeg[:2] == b"\xff\xd8"
    return jpeg[:2] + segment + jpeg[2:]


_FBMD = b"FBMD2300096c01000034580000397700006f9f0000"


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
    assert result.details["jpeg_quant_standard"] is None
    assert result.details["jpeg_quant_quality_exact"] is None


def test_pillow_saved_q75_has_standard_quant_tables() -> None:
    image = Image.new("RGB", (64, 64), color=(120, 130, 140))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=75)
    fi = ForensicImage.from_bytes(buffer.getvalue())

    result = MetadataSignal().predict(fi)

    assert result.details["jpeg_quant_standard"] is True
    assert result.details["jpeg_quant_quality_exact"] == 75


def test_hand_modified_quant_table_is_not_standard() -> None:
    image = Image.new("RGB", (64, 64), color=(120, 130, 140))
    # A flat, non-IJG-scaled table: no combination of luma/chroma the
    # standard libjpeg quality formula can produce is this uniform.
    custom_table = [10] * 64
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", qtables=[custom_table, custom_table])
    fi = ForensicImage.from_bytes(buffer.getvalue())

    result = MetadataSignal().predict(fi)

    assert result.details["jpeg_quant_standard"] is False
    assert result.details["jpeg_quant_quality_exact"] is None


# --------------------------------------------------------------------------
# Photoshop APP13: editor evidence vs. platform re-encode
# --------------------------------------------------------------------------


def test_fbmd_app13_is_a_platform_marker_not_an_editor_marker() -> None:
    """An Instagram download carries an APP13 whose only block is 1028 and whose
    only IPTC dataset is Meta's FBMD fingerprint -- a re-encode, not an edit."""
    data = _with_app13(_jpeg_bytes(), {1028: _iptc(2, 40, _FBMD)})
    fi = ForensicImage.from_bytes(data)

    result = MetadataSignal().predict(fi)

    assert result.details["editor_markers"] == []
    assert len(result.details["platform_markers"]) == 1
    assert "FBMD" in result.details["platform_markers"][0]
    assert result.details["app13_resources"] == [1028]
    assert result.details["iptc"]["special_instructions"].startswith("FBMD")
    assert result.score == pytest.approx(0.50)
    assert result.label == "uncertain"
    assert "platform" in result.details["note"]
    assert "FBMD" in result.details["note"]


def test_fbmd_platform_marker_beside_camera_exif_stays_real() -> None:
    data = _with_app13(_jpeg_bytes(make="Canon", model="EOS"), {1028: _iptc(2, 40, _FBMD)})
    fi = ForensicImage.from_bytes(data)

    result = MetadataSignal().predict(fi)

    assert result.score == pytest.approx(0.30)
    assert result.label == "real"
    assert result.details["editor_markers"] == []
    assert len(result.details["platform_markers"]) == 1


def test_news_agency_iptc_caption_is_not_an_editor_marker() -> None:
    iptc_block = (
        _iptc(2, 80, b"Jane Doe")
        + _iptc(2, 110, b"AFP")
        + _iptc(2, 116, b"2024 Agence France-Presse")
        + _iptc(2, 120, b"Demonstrators march through the city centre.")
    )
    fi = ForensicImage.from_bytes(_with_app13(_jpeg_bytes(), {1028: iptc_block}))

    result = MetadataSignal().predict(fi)

    assert result.details["editor_markers"] == []
    assert result.details["platform_markers"] == []
    assert result.details["iptc"]["byline"] == "Jane Doe"
    assert result.details["iptc"]["caption"].startswith("Demonstrators march")
    assert result.details["iptc"]["credit"] == "AFP"
    assert result.score == pytest.approx(0.50)


def test_extra_app13_resource_blocks_give_an_editor_marker() -> None:
    """A genuine Photoshop save writes further image-resource blocks beside 1028."""
    resources = {
        1028: _iptc(2, 120, b"A caption"),
        1061: b"\x00" * 16,
        1036: b"\x01\x02\x03\x04",
    }
    fi = ForensicImage.from_bytes(_with_app13(_jpeg_bytes(), resources))

    result = MetadataSignal().predict(fi)

    markers = result.details["editor_markers"]
    assert len(markers) == 1
    assert "1036" in markers[0] and "1061" in markers[0]
    assert result.details["app13_resources"] == [1028, 1036, 1061]
    assert result.score == pytest.approx(0.70)
    assert result.label == "fake"


def test_iptc_originating_program_names_an_editor() -> None:
    iptc_block = _iptc(2, 65, b"Adobe Photoshop") + _iptc(2, 70, b"25.0")
    fi = ForensicImage.from_bytes(_with_app13(_jpeg_bytes(), {1028: iptc_block}))

    result = MetadataSignal().predict(fi)

    assert any("Photoshop" in marker for marker in result.details["editor_markers"])
    assert result.details["iptc"]["originating_program"] == "Adobe Photoshop"
    assert result.details["iptc"]["program_version"] == "25.0"
    assert result.score == pytest.approx(0.70)
    assert result.label == "fake"


def test_special_instructions_starting_with_fbmd_but_not_hex_is_ordinary_text() -> None:
    block = {1028: _iptc(2, 40, b"FBMD notes for the desk")}
    fi = ForensicImage.from_bytes(_with_app13(_jpeg_bytes(), block))

    result = MetadataSignal().predict(fi)

    assert result.details["platform_markers"] == []
    assert result.details["editor_markers"] == []
    assert result.details["iptc"]["special_instructions"] == "FBMD notes for the desk"


@pytest.mark.parametrize(
    "blob",
    [
        pytest.param(b"\x1c\x02\x28\x80\x10", id="extended-size"),
        pytest.param(b"\x1c\x02\x28\x00\x40FB", id="truncated-value"),
        pytest.param(b"not an iptc record at all", id="no-tag-marker"),
    ],
)
def test_malformed_iptc_is_ignored_without_error(blob: bytes) -> None:
    """A dataset list we cannot parse is "nothing parsed", not an exception."""
    fi = ForensicImage.from_bytes(_with_app13(_jpeg_bytes(), {1028: blob}))

    result = MetadataSignal().predict(fi)

    assert result.details["editor_markers"] == []
    assert result.details["platform_markers"] == []
    assert result.details["iptc"] == {}
    assert result.details["app13_resources"] == [1028]
    assert "error" not in result.details
    assert "app13_error" not in result.details


def test_jpeg_without_app13_reports_no_resources() -> None:
    fi = ForensicImage.from_bytes(_jpeg_bytes(make="Canon", model="EOS"))

    result = MetadataSignal().predict(fi)

    assert result.details["app13_resources"] is None
    assert result.details["iptc"] == {}
    assert result.details["platform_markers"] == []
