# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project skeleton, core types, CLI stub, CI
- Literature surveys on AI-generated image detection, manipulation localization, and classical forensic signals (docs/research/)
- Development roadmap (docs/ROADMAP.md)
- `ForensicImage` input type carrying original bytes
- metadata signal (EXIF, editor and AI-generator markers, thumbnail consistency, JPEG quality estimate)
- ELA signal with heatmap
- CLI: per-signal cards, `--json`, `--save-heatmaps`, `--detector`
- metadata signal: `jpeg_quant_standard` / `jpeg_quant_quality_exact`, an exact IJG-standard-table classification of the JPEG's luma+chroma quantization tables (complementing the existing quality estimate)
- `c2pa` signal: C2PA manifest verification via the optional `c2pa-python` extra (`pip install imgforensics[provenance]`), lazily imported so the base install is unaffected
- `sd_watermark` signal: decodes the Stable Diffusion `dwtDct` invisible watermark via a vendored pure numpy/opencv/PyWavelets codec (`src/imgforensics/signals/_vendor/dwtdct.py`, from `ShieldMnt/invisible-watermark`), checked against the SDXL 48-bit and CompVis Stable Diffusion v1 136-bit reference payloads

### Changed

- `BaseDetector.predict` now takes `ForensicImage`
