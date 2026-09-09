# image-forensics-toolkit

[![CI](https://github.com/Onurkutan/image-forensics-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/Onurkutan/image-forensics-toolkit/actions/workflows/ci.yml)

A toolkit to detect AI-generated images, AI-inpainted regions, and classic manipulations
(splicing/copy-move), producing an image-level score plus an optional heatmap.

Status: early development (v0.1.0). Four classical signals are implemented so
far: `metadata` (EXIF/XMP/editor/AI-generator markers, thumbnail consistency,
JPEG quality estimate and quantization-table classification), `ela` (Error
Level Analysis with heatmap), `c2pa` (C2PA manifest verification) and
`sd_watermark` (Stable Diffusion invisible-watermark decode). See
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
