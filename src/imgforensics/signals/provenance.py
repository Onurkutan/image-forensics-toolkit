"""C2PA (Content Authenticity Initiative) provenance signal.

Uses the `c2pa-python` package (PyPI: ``c2pa-python``, import name ``c2pa``;
dual-licensed MIT OR Apache-2.0, https://github.com/contentauth/c2pa-python)
to read and verify any C2PA manifest embedded in the image. Verified against
c2pa-python 0.37.10 (c2pa-rs/c2pa-c-ffi 0.90.19): the modern ``Reader`` /
``Builder`` class API, not the older free-function API from pre-0.6 releases.

``c2pa-python`` is imported lazily, only inside :meth:`C2PASignal.predict`,
so the rest of the package works without it installed; it is an optional
extra (``pip install imgforensics[provenance]``).
"""

from __future__ import annotations

import io
import json
from typing import Any

from imgforensics.core.base import BaseDetector
from imgforensics.core.image import ForensicImage
from imgforensics.core.registry import register
from imgforensics.core.types import DetectionResult, Label
from imgforensics.signals.metadata import _find_ai_terms

# Formats mapped to an explicit MIME type; anything else falls back to
# mime=None, letting c2pa-python auto-detect the container from the bytes.
_MIME_BY_FORMAT: dict[str, str] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "TIFF": "image/tiff",
    "HEIC": "image/heic",
    "HEIF": "image/heif",
}

# Validation status code prefixes (see the C2PA spec's status code list, and
# c2pa-rs's own validation_status output) that mean the signed content hash
# no longer matches the asset -- i.e. the pixels changed after signing.
_HASH_MISMATCH_PREFIXES = ("assertion.dataHash.mismatch", "assertion.hashedURI.mismatch")

# Status code prefixes that mean only the certificate's trust chain could not
# be verified (e.g. our own test signer, or any self-signed/private CA not in
# the verifier's trust list) -- everything else about the signature checked
# out. Measured directly against c2pa-python 0.37.10: signing with a fresh,
# structurally valid but untrusted test CA yields exactly
# ``signingCredential.untrusted`` and nothing else.
_CERT_TRUST_FAILURE_PREFIXES = ("signingCredential.untrusted", "signingCredential.selfSigned")

# c2pa.actions action ids (github.com/c2pa-org, "C2PA Actions" assertion)
# that mean the asset was changed after its first "c2pa.created" state.
_EDIT_ACTIONS = frozenset(
    {
        "c2pa.edited",
        "c2pa.placed",
        "c2pa.removed",
        "c2pa.cropped",
        "c2pa.filtered",
        "c2pa.color_adjustments",
        "c2pa.resized",
    }
)

# IPTC digitalSourceType values (the last path segment of the
# http://cv.iptc.org/newscodes/digitalsourcetype/... URL) that mean
# generative-AI output.
_AI_DIGITAL_SOURCE_TYPES = (
    "trainedAlgorithmicMedia",
    "compositeWithTrainedAlgorithmicMedia",
    "algorithmicMedia",
)
_CAPTURE_DIGITAL_SOURCE_TYPE = "digitalCapture"


def _mime_type_for_format(fmt: str | None) -> str | None:
    return _MIME_BY_FORMAT.get(fmt or "")


def _codes_with_prefixes(codes: list[str], prefixes: tuple[str, ...]) -> bool:
    return any(code.startswith(prefix) for code in codes for prefix in prefixes)


def _only_cert_trust_failures(codes: list[str]) -> bool:
    return bool(codes) and all(
        any(code.startswith(prefix) for prefix in _CERT_TRUST_FAILURE_PREFIXES) for code in codes
    )


def _claim_generator_str(claim_generator_info: list[dict[str, Any]]) -> str | None:
    """Join the manifest's ``claim_generator_info`` entries into one display string.

    c2pa-python 0.37.10's Reader.json() no longer exposes the legacy
    ``claim_generator`` free-text string (deprecated in the C2PA spec in
    favour of structured ``claim_generator_info``); this reconstructs an
    equivalent string for AI-marker matching and display.
    """
    parts = []
    for entry in claim_generator_info:
        name = entry.get("name")
        if not name:
            continue
        version = entry.get("version")
        parts.append(f"{name}/{version}" if version else str(name))
    return "; ".join(parts) or None


def _extract_actions(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Every action dict from every ``c2pa.actions`` assertion (v1 or v2)."""
    actions: list[dict[str, Any]] = []
    for assertion in manifest.get("assertions") or []:
        label = assertion.get("label") or ""
        if not label.startswith("c2pa.actions"):
            continue
        actions.extend((assertion.get("data") or {}).get("actions") or [])
    return actions


def _extract_digital_source_type(actions: list[dict[str, Any]]) -> str | None:
    """The ``digitalSourceType`` of the ``c2pa.created`` action, or the first one found."""
    created = next((a for a in actions if a.get("action") == "c2pa.created"), None)
    source_url = None
    if created is not None and created.get("digitalSourceType"):
        source_url = created["digitalSourceType"]
    else:
        source_url = next(
            (a["digitalSourceType"] for a in actions if a.get("digitalSourceType")), None
        )
    if not source_url:
        return None
    return str(source_url).rstrip("/").rsplit("/", 1)[-1]


def _extract_manifest_fields(parsed: dict[str, Any]) -> dict[str, Any]:
    """Pull the fields this signal cares about out of ``Reader.json()``'s parsed dict."""
    active_label = parsed.get("active_manifest")
    manifests = parsed.get("manifests") or {}
    manifest: dict[str, Any] = (manifests.get(active_label) or {}) if active_label else {}

    claim_generator = _claim_generator_str(manifest.get("claim_generator_info") or [])

    signature_info = manifest.get("signature_info") or {}

    raw_status = parsed.get("validation_status") or []
    validation_status = [
        item["code"] for item in raw_status if isinstance(item, dict) and "code" in item
    ]

    actions_raw = _extract_actions(manifest)
    actions = [a["action"] for a in actions_raw if a.get("action")]

    return {
        "manifest_present": True,
        "active_manifest_label": active_label,
        "claim_generator": claim_generator,
        "signature_issuer": signature_info.get("issuer"),
        "signature_time": signature_info.get("time"),
        "validation_status": validation_status,
        "is_valid": len(validation_status) == 0,
        "actions": actions,
        "digital_source_type": _extract_digital_source_type(actions_raw),
        "ingredient_count": len(manifest.get("ingredients") or []),
    }


def _no_manifest_details() -> dict[str, Any]:
    return {
        "manifest_present": False,
        "active_manifest_label": None,
        "claim_generator": None,
        "signature_issuer": None,
        "signature_time": None,
        "validation_status": [],
        "is_valid": None,
        "actions": [],
        "digital_source_type": None,
        "ingredient_count": None,
    }


def _score(details: dict[str, Any]) -> tuple[float, Label]:
    """Score rules, first match wins; see :class:`C2PASignal` for the numbered list."""
    if not details["manifest_present"]:
        details["note"] = "no C2PA manifest; absence proves nothing, most images have none"
        return 0.50, "uncertain"

    codes: list[str] = details["validation_status"]

    if _codes_with_prefixes(codes, _HASH_MISMATCH_PREFIXES):
        details["note"] = "content changed after signing"
        return 0.85, "fake"

    cert_untrusted = _codes_with_prefixes(codes, _CERT_TRUST_FAILURE_PREFIXES)
    if cert_untrusted:
        details["cert_untrusted"] = True
    manifest_ok = bool(details["is_valid"]) or _only_cert_trust_failures(codes)

    digital_source_type = details["digital_source_type"] or ""
    ai_source_type = any(t in digital_source_type for t in _AI_DIGITAL_SOURCE_TYPES)
    ai_claim_generator = bool(details["claim_generator"]) and bool(
        _find_ai_terms(details["claim_generator"])
    )
    edit_actions_present = any(a in _EDIT_ACTIONS for a in details["actions"])

    if manifest_ok and (ai_source_type or ai_claim_generator):
        return 0.95, "fake"
    if manifest_ok and edit_actions_present:
        details["note"] = "signed edit history present"
        return 0.60, "uncertain"
    if (
        manifest_ok
        and digital_source_type == _CAPTURE_DIGITAL_SOURCE_TYPE
        and not edit_actions_present
    ):
        return 0.15, "real"
    if not manifest_ok:
        details["note"] = f"C2PA validation failed: {', '.join(codes) or 'unknown reason'}"
        return 0.70, "fake"

    # Manifest present and valid (rules 1-4 above all had a specific
    # condition to match), but the digital source type is neither a
    # recognized AI marker nor a plain capture, and there is no edit
    # history -- e.g. no c2pa.actions assertion at all, or a source type
    # like "humanEdits"/"digitalCreation" this signal does not specifically
    # score. Fallback for a valid manifest that matches none of the six
    # rules above; added so the signal never falls through without a
    # documented answer.
    details["note"] = (
        "C2PA manifest valid but inconclusive: no recognized source-type or action evidence"
    )
    return 0.50, "uncertain"


@register("c2pa")
class C2PASignal(BaseDetector):
    """Reads and verifies a C2PA (Content Authenticity Initiative) manifest.

    Requires the original encoded bytes (``image.raw``); when unavailable,
    or when ``c2pa-python`` is not installed, this signal abstains (score
    0.5, label "uncertain") rather than raising -- see ``details["reason"]``.

    Score rules, evaluated in order (first match wins):

    1. Manifest present and any validation failure indicates the content
       hash no longer matches (a status code starting with
       ``"assertion.dataHash.mismatch"`` or ``"assertion.hashedURI.mismatch"``)
       -> score 0.85, label "fake" ("content changed after signing").
    2. Manifest valid (or valid except an untrusted/self-signed certificate
       status, e.g. ``"signingCredential.untrusted"`` -- recorded as
       ``details["cert_untrusted"] = True``) and either
       ``digital_source_type`` names a trained-algorithmic-media source
       (``trainedAlgorithmicMedia``, ``compositeWithTrainedAlgorithmicMedia``,
       ``algorithmicMedia``) or ``claim_generator`` names a known
       generative-AI tool (reusing
       :func:`imgforensics.signals.metadata._find_ai_terms`) -> score 0.95,
       label "fake".
    3. Manifest valid (as in rule 2) with any edit action present
       (``c2pa.edited``, ``c2pa.placed``, ``c2pa.removed``, ``c2pa.cropped``,
       ``c2pa.filtered``, ``c2pa.color_adjustments``, ``c2pa.resized``) ->
       score 0.60, label "uncertain" ("signed edit history present").
    4. Manifest valid (as in rule 2) with a capture-type source
       (``digitalCapture``) and no edit actions -> score 0.15, label "real".
    5. Manifest present but validation failed for other reasons -> score
       0.70, label "fake", with the failing codes listed in ``details["note"]``.
    6. No manifest at all -> score 0.50, label "uncertain".

    A manifest present, valid, but inconclusive about source type or edits
    (e.g. no ``c2pa.actions`` assertion) falls through all six rules above to
    an added score 0.50 "uncertain" default -- not part of the six numbered
    rules, but needed since they do not cover every valid manifest shape.

    Every computed field is placed in ``details``; this signal never raises,
    catching unexpected errors into ``details["error"]`` with score 0.5.
    """

    name = "c2pa"

    def predict(self, image: ForensicImage) -> DetectionResult:
        if image.raw is None:
            return DetectionResult(
                detector=self.name,
                score=0.5,
                label="uncertain",
                details={"reason": "no encoded file available"},
            )
        try:
            import c2pa  # noqa: F401  (import-only probe; see _predict_from_raw)
        except ImportError:
            return DetectionResult(
                detector=self.name,
                score=0.5,
                label="uncertain",
                details={
                    "reason": "c2pa-python not installed; pip install imgforensics[provenance]"
                },
            )
        try:
            return self._predict_from_raw(image.raw, image.format)
        except Exception as exc:  # never raise: record and abstain
            return DetectionResult(
                detector=self.name,
                score=0.5,
                label="uncertain",
                details={"error": f"{type(exc).__name__}: {exc}"},
            )

    def _predict_from_raw(self, raw: bytes, fmt: str | None) -> DetectionResult:
        import c2pa

        mime_type = _mime_type_for_format(fmt)
        reader = c2pa.Reader.try_create(mime_type, io.BytesIO(raw))

        if reader is None:
            details = _no_manifest_details()
        else:
            with reader:
                parsed = json.loads(reader.json())
            details = _extract_manifest_fields(parsed)

        score, label = _score(details)
        return DetectionResult(detector=self.name, score=score, label=label, details=details)
