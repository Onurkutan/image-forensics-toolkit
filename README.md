# image-forensics-toolkit

[![CI](https://github.com/Onurkutan/image-forensics-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/Onurkutan/image-forensics-toolkit/actions/workflows/ci.yml)

A toolkit to detect AI-generated images, AI-inpainted regions, and classic manipulations
(splicing/copy-move), producing an image-level score plus an optional heatmap.

Status: early development (v0.1.0). Seven classical signals are implemented so
far: `metadata` (EXIF/XMP/editor/AI-generator markers, thumbnail consistency,
JPEG quality estimate and quantization-table classification), `ela` (Error
Level Analysis with heatmap), `c2pa` (C2PA manifest verification),
`sd_watermark` (Stable Diffusion invisible-watermark decode), `copy_move`
(block-matching duplicated-region detection with heatmap), `jpeg_ghost`
(JPEG-ghost recompression-quality mismatch with heatmap) and `double_jpeg`
(blocking-grid offset and aligned double-quantization periodicity). See
[Signals](#signals) below for what each one reads and its blind spots.

## Planned architecture

- **Detectors** (`imgforensics.detectors`): image-level classifiers that score a whole image as
  real or AI-generated.
- **Localization** (`imgforensics.localization`): pixel-level localizers that highlight
  manipulated or AI-inpainted regions with a heatmap.
- **Signals** (`imgforensics.signals`): classical low-level signal analyzers such as noise
  residuals, error level analysis, and JPEG artifact statistics.
- **Fusion** (`imgforensics.fusion`): combines detector and localizer outputs into a single
  image-level score with an explanation.

## Quickstart

```bash
git clone https://github.com/Onurkutan/image-forensics-toolkit.git
cd image-forensics-toolkit
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
# Optional: pip install -e ".[dev,provenance]" to also enable the c2pa signal
imgforensics analyze path/to/image.jpg
imgforensics analyze image.jpg --json --save-heatmaps out/
pytest
```

## Signals

Each signal is a self-contained, explainable check. Score is the probability
the image is generated/manipulated (0 = confidently real, 1 = confidently
fake); label is `real`, `fake` or `uncertain`. No signal alone should be read
as a verdict -- see `details` in its output for the evidence.

| Signal | Reads | Score means | Known blind spots |
|---|---|---|---|
| `metadata` | EXIF/XMP, PNG text chunks, Photoshop APP13, embedded EXIF thumbnail, JPEG quantization tables | High = a known AI-generator/editor marker or a thumbnail/image mismatch was found; low = camera EXIF with no edit trace; 0.5 = no usable metadata | Stripped by almost every sharing platform (upload to social media and this signal goes blind); markers are only as good as the list of known tool names |
| `ela` | Re-encodes the image at a fixed JPEG quality and diffs against the original | Bounded to [0.3, 0.65] -- an explanation aid (see the heatmap), never a standalone verdict | Useless on an already-uniformly-recompressed image; a second JPEG save by any platform equalises the error level everywhere |
| `c2pa` | Any C2PA manifest embedded in the file (via `c2pa-python`, optional `provenance` extra) | High = signed as AI-generated or the signed content hash no longer matches; low = signed camera capture with no edits; 0.5 = no manifest or an untrusted/self-signed signer with no other evidence | A missing manifest proves nothing -- most images, including AI-generated ones, carry no C2PA data at all; manifests are stripped by many platforms just like EXIF |
| `sd_watermark` | Decodes the DWT-DCT ("dwtDct") invisible watermark Stable Diffusion reference pipelines embed, checking bit-agreement against known payloads | High = a known payload's decoded bits matched; 0.45 = no known payload matched | This specific scheme embeds only in chroma and does not survive JPEG re-encoding (even at quality 100, due to chroma subsampling) or a resize; only covers the two reference-pipeline payloads, not every SD fork or later generators |
| `copy_move` | Block-matching search (downscaled, quantized zig-zag DCT features) for a duplicated region copied and pasted elsewhere in the same image, with a matched-region heatmap | 0.85 "fake" = a dominant shift with enough votes and matched area found; 0.60 "uncertain" = accepted but small matched area; 0.45 "uncertain" = no duplicate found (not evidence of authenticity) | Blind to a clone that was rotated or rescaled before pasting; heavy recompression can in principle merge distinct blocks; naturally repetitive textures (tiles, fences) are guarded against via a minimum-distance rule and a shift-consistency check, but are not impossible to fool |
| `jpeg_ghost` | Re-saves the image at a range of JPEG qualities and finds, per block, the quality whose re-save error is anomalously low compared to the rest of the image (Farid's JPEG ghosts), with a heatmap | Bounded to [0.3, 0.7] like `ela` -- an explanation aid, never a standalone verdict | Requires the image to be JPEG-derived; useless once a platform has uniformly re-encoded the whole image after the fact |
| `double_jpeg` | Two independent pixel-domain checks: whether the JPEG 8x8 blocking grid still starts at the image origin, with a secondary-phase check for a masked older grid underneath it (crop/composite detection); and, for JPEG inputs only, whether the DCT coefficient histogram -- normalized by the file's own quantization step, so the check measures a genuine second, coarser compression rather than just how lossy the current one is -- shows aligned double-quantization periodicity | 0.75 "fake" = blocking grid misaligned; 0.60 "uncertain" = a second, offset grid or double compression suspected (weak evidence alone); 0.45/0.40 "uncertain" = no JPEG history detectable / no evidence either way | Both checks are quantization-history fingerprints, not proof of malicious editing; the double-quantization check only detects a *coarser-then-finer* double compression (the reverse order, and same-or-finer-then-coarser, leave no detectable comb) and only on JPEG inputs; a suspected double compression is common for any re-shared image, and a misaligned grid only proves a crop-then-resave happened |

## Project layout

```
image-forensics-toolkit/
├── src/imgforensics/
│   ├── core/            # types, base detector class, registry
│   ├── detectors/       # image-level detectors
│   ├── localization/    # pixel-level localizers
│   ├── signals/         # classical low-level signals
│   ├── fusion/          # score fusion and explanation
│   ├── data/            # dataset loaders and download helpers
│   ├── utils/           # image I/O helpers
│   └── cli.py           # command-line interface
├── tests/
└── docs/
```

## Data and evaluation

A dataset manifest is a JSON Lines file of labeled images (path, label,
source, generator, split, mask path, sha256, resolution, format, JPEG
quality) plus a `*.meta.json` sidecar (dataset name, license,
`commercial_ok`). Build one from a folder tree with `real`/`fake`
subfolders via `imgforensics manifest build ROOT --dataset NAME --out
manifest.jsonl`, browse the external dataset registry with `imgforensics
datasets list` / `datasets show NAME`, and check a manifest's real/fake
halves for format, resolution, JPEG-quality, and duplicate-image bias with
`imgforensics audit manifest.jsonl [--strict]`. See
[`imgforensics.eval.metrics`](src/imgforensics/eval/metrics.py) for the
image-level (AUC, AP, accuracy, ECE, ...) and pixel-level (F1, best-F1, AP,
IoU) metrics used to score detectors and localizers.

## Benchmarking

`imgforensics benchmark manifest.jsonl [--detector NAME ...] [--baselines]
[--robustness default|PATH|none] [--out results.json] [--report report.md]`
runs registered detectors, plus optional trivial baselines, over a manifest
at every level of a deterministic robustness suite (clean; JPEG/WEBP
re-encoding; resize; a resize round-trip; center crop; Gaussian noise; a
"social" resize+JPEG+metadata-strip pipeline; see
[`robustness_default.yaml`](src/imgforensics/eval/robustness_default.yaml)),
and prints Markdown tables: image metrics, per-level robustness, per-group
AUC, pixel metrics, and timing. Each detector's threshold is tuned on the
manifest's val split when one exists; otherwise the report is marked
"threshold tuned in-sample" as a caveat against reading it as held-out. At
any operating threshold, a score must be strictly above it to count as a
"fake" call, so a detector abstaining at the classical-signal midpoint of
0.5 is not scored as calling every image fake.

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md).

## Research notes

See [docs/research/](docs/research/).

## License

The code in this repository is released under the MIT license, see [LICENSE](LICENSE).

This is a personal, non-commercial research project. Third-party models, weights and
datasets are not included in the repository; they are downloaded from their original sources
and keep their own licenses, some of which permit research use only. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the current library list; models and
datasets are added there as they are integrated.
