# image-forensics-toolkit

[![CI](https://github.com/Onurkutan/image-forensics-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/Onurkutan/image-forensics-toolkit/actions/workflows/ci.yml)

A toolkit to detect AI-generated images, AI-inpainted regions, and classic manipulations
(splicing/copy-move), producing an image-level score plus an optional heatmap.

Status: early development (v0.1.0). Two classical signals are implemented so
far: `metadata` (EXIF/XMP/editor/AI-generator markers, thumbnail consistency,
JPEG quality estimate) and `ela` (Error Level Analysis with heatmap).

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
imgforensics analyze path/to/image.jpg
imgforensics analyze image.jpg --json --save-heatmaps out/
pytest
```

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
