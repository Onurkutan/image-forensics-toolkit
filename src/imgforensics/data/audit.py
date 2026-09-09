"""Bias audit: compare the real/fake halves of a manifest for shortcut-learning risk.

The classic self-deception in AI-generated-image detection is a dataset
where the real and fake classes differ in some trivial, non-semantic way
(all reals are JPEG and all fakes are PNG, reals are higher resolution,
fakes were all compressed at one JPEG quality, ...): a model trained on it
learns to detect the confound, not the generator (see ``docs/ROADMAP.md``,
section 2, and the "format bias" pitfall in
``docs/research/01_ai_generated_image_detection.md``, section 4). This
module compares per-label distributions of format, resolution, JPEG
quality, and aspect ratio, using the Jensen-Shannon distance to flag
divergence, and separately flags duplicate images (within a label, and the
same image labelled both real and fake).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from imgforensics.data.manifest import Manifest, ManifestEntry

#: Jensen-Shannon distance above which a distribution divergence is flagged.
_JS_THRESHOLD = 0.3
#: A format/bucket covering at least this share of one label and at most
#: (1 - this share) of the other is flagged even if the overall JS distance
#: is below the threshold (a single dominant category can hide inside an
#: otherwise-moderate JS distance when there are many small categories).
_DOMINANCE_HIGH = 0.9
_DOMINANCE_LOW = 0.1
#: Class imbalance worse than this ratio (majority:minority) is flagged.
_IMBALANCE_RATIO = 5.0


class BiasError(Exception):
    """Raised by :func:`audit_manifest` in strict mode when problems are found."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass
class AuditReport:
    """Result of :func:`audit_manifest`.

    ``distributions`` is nested as ``{axis: {label: {bucket: count}}}`` for
    ``axis`` in ``"format"``, ``"resolution"``, ``"jpeg_quality"``,
    ``"aspect_ratio"``.
    """

    distributions: dict[str, dict[str, dict[str, int]]]
    problems: list[str] = field(default_factory=list)
    ok: bool = True

    def to_markdown(self) -> str:
        """Render this report as a Markdown document."""
        lines = ["# Bias audit report", ""]
        if self.ok:
            lines.append("No problems found.")
        else:
            lines.append(f"**{len(self.problems)} problem(s) found:**")
            lines.append("")
            lines.extend(f"- {problem}" for problem in self.problems)
        lines.append("")

        for axis, per_label in self.distributions.items():
            labels = sorted(per_label)
            buckets = sorted({bucket for counts in per_label.values() for bucket in counts})
            lines.append(f"## {axis}")
            lines.append("")
            lines.append("| bucket | " + " | ".join(labels) + " |")
            lines.append("|---|" + "---|" * len(labels))
            for bucket in buckets:
                row = " | ".join(str(per_label[label].get(bucket, 0)) for label in labels)
                lines.append(f"| {bucket} | {row} |")
            lines.append("")
        return "\n".join(lines)


def _js_distance(counts_a: dict[str, int], counts_b: dict[str, int]) -> float:
    """Base-2 Jensen-Shannon distance between two count distributions, in [0, 1].

    Computed as ``sqrt(0.5 * KL(P||M) + 0.5 * KL(Q||M))`` where ``M`` is the
    mixture ``(P + Q) / 2``; this is the square root of the Jensen-Shannon
    *divergence*, which is bounded in ``[0, 1]`` for base-2 logarithms.
    Returns ``0.0`` when either distribution has zero total count (nothing
    to compare).
    """
    keys = sorted(set(counts_a) | set(counts_b))
    if not keys:
        return 0.0
    a = np.array([counts_a.get(key, 0) for key in keys], dtype=np.float64)
    b = np.array([counts_b.get(key, 0) for key in keys], dtype=np.float64)
    total_a, total_b = a.sum(), b.sum()
    if total_a <= 0 or total_b <= 0:
        return 0.0
    p, q = a / total_a, b / total_b
    m = 0.5 * (p + q)

    def _kl(x: np.ndarray, y: np.ndarray) -> float:
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    divergence = max(0.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m))
    return float(min(1.0, np.sqrt(divergence)))


def _dominance_note(
    counts_a: dict[str, int], counts_b: dict[str, int], total_a: int, total_b: int
) -> str | None:
    """Describe a bucket that dominates one label and is nearly absent from the other."""
    for key in sorted(set(counts_a) | set(counts_b)):
        frac_a = counts_a.get(key, 0) / total_a if total_a else 0.0
        frac_b = counts_b.get(key, 0) / total_b if total_b else 0.0
        if frac_a >= _DOMINANCE_HIGH and frac_b <= _DOMINANCE_LOW:
            return f"'{key}' covers {frac_a:.0%} of real but only {frac_b:.0%} of fake."
        if frac_b >= _DOMINANCE_HIGH and frac_a <= _DOMINANCE_LOW:
            return f"'{key}' covers {frac_b:.0%} of fake but only {frac_a:.0%} of real."
    return None


def _resolution_bucket(min_side: int) -> str:
    if min_side < 512:
        return "<512"
    if min_side < 1024:
        return "512-1023"
    if min_side < 2048:
        return "1024-2047"
    return ">=2048"


def _quality_bucket(quality: int | None) -> str:
    if quality is None:
        return "none"
    clamped = max(1, min(100, quality))
    lower = ((clamped - 1) // 10) * 10 + 1
    return f"{lower}-{lower + 9}"


def _aspect_bucket(width: int, height: int) -> str:
    ratio = width / height if height else 0.0
    if ratio < 0.75:
        return "<0.75"
    if ratio <= 1.33:
        return "0.75-1.33"
    return ">1.33"


def _duplicate_extra_count(hashes: list[str]) -> int:
    """Number of entries beyond the first in every repeated-hash group."""
    counts = Counter(hashes)
    return sum(count - 1 for count in counts.values() if count > 1)


def _bucket_counts(entries: list[ManifestEntry]) -> dict[str, dict[str, int]]:
    format_counts: dict[str, int] = {}
    resolution_counts: dict[str, int] = {}
    quality_counts: dict[str, int] = {}
    aspect_counts: dict[str, int] = {}
    for entry in entries:
        format_counts[entry.format] = format_counts.get(entry.format, 0) + 1
        resolution_key = _resolution_bucket(min(entry.width, entry.height))
        resolution_counts[resolution_key] = resolution_counts.get(resolution_key, 0) + 1
        quality_key = _quality_bucket(entry.jpeg_quality)
        quality_counts[quality_key] = quality_counts.get(quality_key, 0) + 1
        aspect_key = _aspect_bucket(entry.width, entry.height)
        aspect_counts[aspect_key] = aspect_counts.get(aspect_key, 0) + 1
    return {
        "format": format_counts,
        "resolution": resolution_counts,
        "jpeg_quality": quality_counts,
        "aspect_ratio": aspect_counts,
    }


def audit_manifest(manifest: Manifest, *, strict: bool = False) -> AuditReport:
    """Compare the real and fake halves of ``manifest`` for shortcut-learning bias.

    See the module docstring for the checks performed. When ``strict`` is
    true and any problems are found, raises :class:`BiasError` instead of
    returning normally (the report is still fully computed first, so the
    exception's ``problems`` attribute is the same list a non-strict call
    would return in ``report.problems``).
    """
    by_label: dict[str, list[ManifestEntry]] = {"real": [], "fake": []}
    for entry in manifest.entries:
        by_label.setdefault(entry.label, []).append(entry)

    per_label_buckets: dict[str, dict[str, dict[str, int]]] = {
        label: _bucket_counts(entries) for label, entries in by_label.items()
    }
    distributions: dict[str, dict[str, dict[str, int]]] = {
        "format": {},
        "resolution": {},
        "jpeg_quality": {},
        "aspect_ratio": {},
    }
    for label, axes in per_label_buckets.items():
        for axis, counts in axes.items():
            distributions[axis][label] = counts

    hashes_by_label = {
        label: [entry.sha256 for entry in entries] for label, entries in by_label.items()
    }

    problems: list[str] = []
    real_count, fake_count = len(by_label["real"]), len(by_label["fake"])

    if real_count and fake_count:
        js_format = _js_distance(distributions["format"]["real"], distributions["format"]["fake"])
        dominance = _dominance_note(
            distributions["format"]["real"], distributions["format"]["fake"], real_count, fake_count
        )
        if js_format > _JS_THRESHOLD or dominance is not None:
            message = (
                "Format distribution differs between real and fake "
                f"(Jensen-Shannon distance {js_format:.2f})."
            )
            if dominance:
                message += f" {dominance}"
            problems.append(message)

        js_resolution = _js_distance(
            distributions["resolution"]["real"], distributions["resolution"]["fake"]
        )
        if js_resolution > _JS_THRESHOLD:
            problems.append(
                "Resolution-bucket distribution differs between real and fake "
                f"(Jensen-Shannon distance {js_resolution:.2f})."
            )

        real_quality = {
            k: v for k, v in distributions["jpeg_quality"]["real"].items() if k != "none"
        }
        fake_quality = {
            k: v for k, v in distributions["jpeg_quality"]["fake"].items() if k != "none"
        }
        if real_quality and fake_quality:
            js_quality = _js_distance(real_quality, fake_quality)
            if js_quality > _JS_THRESHOLD:
                problems.append(
                    "JPEG-quality distribution differs between real and fake "
                    f"(Jensen-Shannon distance {js_quality:.2f}, "
                    "computed on entries with a known quality)."
                )

        ratio = max(real_count, fake_count) / min(real_count, fake_count)
        if ratio > _IMBALANCE_RATIO:
            problems.append(
                f"Class imbalance worse than 1:{_IMBALANCE_RATIO:g} "
                f"(real={real_count}, fake={fake_count}, ratio {ratio:.1f}:1)."
            )
    elif real_count or fake_count:
        problems.append(
            f"Only one label is present in the manifest (real={real_count}, fake={fake_count})."
        )

    for label, hashes in hashes_by_label.items():
        extra = _duplicate_extra_count(hashes)
        if extra > 0:
            problems.append(
                f"{extra} duplicate image(s) found within label '{label}' (same sha256 repeats)."
            )

    cross_label_duplicates = set(hashes_by_label["real"]) & set(hashes_by_label["fake"])
    if cross_label_duplicates:
        problems.append(
            f"{len(cross_label_duplicates)} image(s) have the same sha256 under both the 'real' "
            "and 'fake' labels (same image labelled both ways)."
        )

    report = AuditReport(distributions=distributions, problems=problems, ok=not problems)
    if strict and problems:
        raise BiasError(problems)
    return report
