"""Tests for imgforensics.signals.provenance.C2PASignal.

Real signing was achieved: these tests generate a self-signed two-level
ES256 certificate chain (root CA + leaf, both throwaway, generated fresh per
test run) with `cryptography`, then drive the actual c2pa-python 0.37.10
Builder/Signer API to sign a small in-memory JPEG. c2pa-python rejects a
single self-signed leaf certificate outright ("the certificate is invalid"),
so a minimal two-cert chain (leaf issued by a throwaway root CA) is the
minimum that satisfies its structural checks; verifying that chain against a
trust list is a separate step this signal does not need, since our score
rules explicitly treat "valid except untrusted certificate" the same as
fully valid (see ``_CERT_TRUST_FAILURE_PREFIXES`` in the signal module).
"""

from __future__ import annotations

import datetime
import io

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from PIL import Image

from imgforensics.core.image import ForensicImage
from imgforensics.signals.provenance import C2PASignal

c2pa = pytest.importorskip(
    "c2pa", reason="c2pa-python is an optional extra (imgforensics[provenance])"
)


def _make_root() -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    now = datetime.datetime.now(datetime.timezone.utc)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            # c2pa-rs 0.90.19 requires an Organization attribute on the
            # signer's certificate chain: omitting it produces a spurious
            # "claimSignature.mismatch" validation failure alongside the
            # expected "signingCredential.untrusted" one (verified directly:
            # identical certs differing only in this attribute reproduce it
            # every time).
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "imgforensics test root CA"),
            x509.NameAttribute(NameOID.COMMON_NAME, "imgforensics Test Root CA"),
        ]
    )
    ski = x509.SubjectKeyIdentifier.from_public_key(key.public_key())
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=1), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(ski, critical=False)
        .sign(key, hashes.SHA256())
    )
    return key, cert


def _make_leaf(
    root_key: ec.EllipticCurvePrivateKey, root_cert: x509.Certificate
) -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    now = datetime.datetime.now(datetime.timezone.utc)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "imgforensics test signer"),
            x509.NameAttribute(NameOID.COMMON_NAME, "imgforensics Test Signer"),
        ]
    )
    root_ski = root_cert.extensions.get_extension_for_class(x509.SubjectKeyIdentifier)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(root_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=True,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(root_ski.value),
            critical=False,
        )
        .sign(root_key, hashes.SHA256())
    )
    return key, cert


def _pem(cert: x509.Certificate) -> str:
    return cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")


def _make_signer() -> c2pa.Signer:
    root_key, root_cert = _make_root()
    leaf_key, leaf_cert = _make_leaf(root_key, root_cert)
    cert_chain_pem = _pem(leaf_cert) + _pem(root_cert)
    leaf_key_pem = leaf_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    signer_info = c2pa.C2paSignerInfo(
        alg=c2pa.C2paSigningAlg.ES256,
        sign_cert=cert_chain_pem.encode("utf-8"),
        private_key=leaf_key_pem.encode("utf-8"),
        ta_url=None,
    )
    return c2pa.Signer.from_info(signer_info)


def _sign_jpeg(manifest: dict, *, size: tuple[int, int] = (64, 64)) -> bytes:
    signer = _make_signer()
    builder = c2pa.Builder(manifest)
    src_img = Image.new("RGB", size, color=(120, 130, 140))
    src_buf = io.BytesIO()
    src_img.save(src_buf, format="JPEG", quality=90)
    src_buf.seek(0)
    dest_buf = io.BytesIO()
    builder.sign(signer, "image/jpeg", src_buf, dest_buf)
    dest_buf.seek(0)
    return dest_buf.getvalue()


def _digital_source_type_url(name: str) -> str:
    return f"http://cv.iptc.org/newscodes/digitalsourcetype/{name}"


def test_ai_generated_manifest_gives_fake_score() -> None:
    manifest = {
        "claim_generator": "imgforensics-test/0.1",
        "title": "ai.jpg",
        "format": "image/jpeg",
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "digitalSourceType": _digital_source_type_url(
                                "trainedAlgorithmicMedia"
                            ),
                        }
                    ]
                },
            }
        ],
    }
    data = _sign_jpeg(manifest)
    fi = ForensicImage.from_bytes(data)

    result = C2PASignal().predict(fi)

    assert result.details["manifest_present"] is True
    assert result.details["digital_source_type"] == "trainedAlgorithmicMedia"
    assert result.details["cert_untrusted"] is True
    assert result.score == pytest.approx(0.95)
    assert result.label == "fake"


def test_capture_manifest_with_no_edits_gives_real_score() -> None:
    manifest = {
        "claim_generator": "imgforensics-test/0.1",
        "title": "capture.jpg",
        "format": "image/jpeg",
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "digitalSourceType": _digital_source_type_url("digitalCapture"),
                        }
                    ]
                },
            }
        ],
    }
    data = _sign_jpeg(manifest)
    fi = ForensicImage.from_bytes(data)

    result = C2PASignal().predict(fi)

    assert result.details["digital_source_type"] == "digitalCapture"
    assert result.details["actions"] == ["c2pa.created"]
    assert result.score == pytest.approx(0.15)
    assert result.label == "real"


def test_capture_manifest_with_edit_action_gives_uncertain_score() -> None:
    manifest = {
        "claim_generator": "imgforensics-test/0.1",
        "title": "edited.jpg",
        "format": "image/jpeg",
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "digitalSourceType": _digital_source_type_url("digitalCapture"),
                        },
                        {"action": "c2pa.edited"},
                    ]
                },
            }
        ],
    }
    data = _sign_jpeg(manifest)
    fi = ForensicImage.from_bytes(data)

    result = C2PASignal().predict(fi)

    assert "c2pa.edited" in result.details["actions"]
    assert result.score == pytest.approx(0.60)
    assert result.label == "uncertain"
    assert "edit history" in result.details["note"]


def test_tampered_asset_gives_hash_mismatch_fake_score() -> None:
    manifest = {
        "claim_generator": "imgforensics-test/0.1",
        "title": "capture.jpg",
        "format": "image/jpeg",
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "digitalSourceType": _digital_source_type_url("digitalCapture"),
                        }
                    ]
                },
            }
        ],
    }
    data = bytearray(_sign_jpeg(manifest))
    # Flip bytes well into the compressed scan data (past all JPEG markers and
    # the embedded JUMBF/C2PA manifest, which sits near the front of the
    # file) so the file stays structurally parseable but its pixel content
    # no longer matches what was hashed at signing time.
    tamper_at = len(data) - 200
    for i in range(tamper_at, tamper_at + 20):
        data[i] ^= 0xFF

    fi = ForensicImage.from_bytes(bytes(data))
    result = C2PASignal().predict(fi)

    assert any(
        code.startswith("assertion.dataHash.mismatch")
        for code in result.details["validation_status"]
    )
    assert result.score == pytest.approx(0.85)
    assert result.label == "fake"
    assert result.details["note"] == "content changed after signing"


def test_plain_jpeg_without_manifest_is_uncertain() -> None:
    image = Image.new("RGB", (32, 32), color=(4, 5, 6))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    fi = ForensicImage.from_bytes(buffer.getvalue())

    result = C2PASignal().predict(fi)

    assert result.score == pytest.approx(0.5)
    assert result.label == "uncertain"
    assert result.details["manifest_present"] is False
    assert "note" in result.details


def test_from_pil_abstains() -> None:
    fi = ForensicImage.from_pil(Image.new("RGB", (8, 8)))

    result = C2PASignal().predict(fi)

    assert result.score == pytest.approx(0.5)
    assert result.label == "uncertain"
    assert result.details["reason"] == "no encoded file available"


def test_c2pa_not_installed_abstains(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "c2pa":
            raise ImportError("simulated: c2pa-python not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    image = Image.new("RGB", (16, 16), color=(1, 2, 3))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    fi = ForensicImage.from_bytes(buffer.getvalue())

    result = C2PASignal().predict(fi)

    assert result.score == pytest.approx(0.5)
    assert result.label == "uncertain"
    assert result.details["reason"] == (
        "c2pa-python not installed; pip install imgforensics[provenance]"
    )
