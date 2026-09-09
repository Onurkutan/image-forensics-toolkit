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
- `copy_move` signal: block-matching copy-move (duplicated-region) detector with a matched-region heatmap, using overlapping quantized zig-zag-DCT block features sorted for near-neighbour matching and a shift-vote with a periodic-texture guard
- `jpeg_ghost` signal: Farid's JPEG-ghost recompression-quality mismatch, with a splice-localization heatmap
- `double_jpeg` signal: JPEG 8x8 blocking-grid offset (crop/composite detection) and aligned double-quantization periodicity, both pixel-domain checks with no DCT-coefficient library dependency
- shared test fixture `tests/conftest.py::natural_like_image`, reused by the copy-move, JPEG-ghost and double-JPEG test suites
- `imgforensics.data.manifest`: `Manifest`/`ManifestEntry`/`ManifestMeta` (pydantic), JSON-Lines save/load, `summary()`, and `build_manifest()` to walk a labeled image tree (sha256, resolution, format, JPEG quality via the metadata signal's quality estimator) with a `label_from_parent_folder` helper for `real`/`fake`-style folder layouts
- `imgforensics.data.registry`: `DatasetInfo` and a packaged `datasets.yaml` cataloguing 24 datasets from the AI-generated-image-detection and manipulation-localization literature surveys, each with license, access, and a `commercial_ok` flag (`None` when unverified); `load_registry()`, `get_dataset()`, `registry_table_markdown()`
- `imgforensics.data.audit`: `audit_manifest()` bias audit comparing a manifest's real/fake halves by format, resolution, JPEG-quality, and aspect-ratio distribution (Jensen-Shannon distance) plus within- and across-label duplicate detection; `AuditReport.to_markdown()`; strict mode raises `BiasError`
- `imgforensics.eval.metrics`: pure-numpy image-level metrics (`roc_auc`, `average_precision`, accuracy/balanced-accuracy/FPR/TPR at a threshold, `best_threshold`, `expected_calibration_error`, `brier_score`) and pixel-level metrics (`pixel_f1`, `pixel_best_f1`, `pixel_ap`, `pixel_iou`), bundled as `ImageMetrics`/`PixelMetrics`
- CLI: `imgforensics datasets list` / `datasets show NAME`, `imgforensics audit MANIFEST [--strict]`, `imgforensics manifest build ROOT --dataset NAME --out PATH [--license] [--commercial-ok/--no-commercial-ok]`
- `PyYAML` runtime dependency, used to load the packaged `datasets.yaml` registry

### Changed

- `BaseDetector.predict` now takes `ForensicImage`
