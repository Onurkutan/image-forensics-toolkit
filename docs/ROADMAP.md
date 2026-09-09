# Roadmap

Last updated: 2026-09-09. This document turns the three literature surveys in
[`docs/research/`](research/) into an ordered build plan. It is a living document: each
phase ends with a short results note appended to its section.

## 1. Goal and non-goals

**Goal.** A toolkit that produces *calibrated evidence* about whether an image is
(a) fully AI-generated, (b) locally AI-edited (inpainting / generative fill), or
(c) classically manipulated (splicing, copy-move, retouching), together with a heatmap and a
human-readable explanation wherever a method can provide one.

**Non-goals for v1.** Video, face-specific deepfake models, multimodal-LLM explainers
(7B–22B parameters do not fit the target hardware), active watermark embedding, and any
claim of a universal "AI or not" verdict.

## 2. What the research says

**Whole-image AI detection** ([survey 01](research/01_ai_generated_image_detection.md)).
There is no universal detector. The best public detector reaches roughly 75% mean accuracy
across 291 generators and drops to 18–30% on 2025 commercial generators
(arXiv:2602.07814). Training-data diversity explains more variance than architecture.
Frozen DINOv2 features beat CLIP features as a detection space (arXiv:2507.10236).
Two preprocessing rules carry measured gains: crop, never resize; always augment with
JPEG/WEBP recompression and noise. The main self-deception is dataset format bias, where
real images are JPEG and fakes are PNG, so the model learns a JPEG detector
(arXiv:2403.17608).

**Manipulation localization** ([survey 02](research/02_manipulation_localization.md)).
Classic splicing/copy-move models transfer poorly to diffusion inpainting because there is
no second camera and no second compression history. License-clean pretrained localizers
with public weights exist: CAT-Net v2 (Apache-2.0 code, CC-BY-4.0 weights), IML-ViT (MIT)
and SAFIRE (Apache-2.0 / CC-BY-4.0). TruFor is non-commercial and must not become a default
dependency. The one trainable recipe that fits a 6 GB GPU is a DINOv2 + LoRA patch-level
localizer (DinoLizer, arXiv:2511.20722). TGIF/TGIF2 is the main generative-edit dataset
(CC BY-SA 4.0). Pixel-level numbers are only comparable when F1 at a fixed 0.5 threshold,
best-F1 and AP are reported side by side.

**Classical signals and provenance** ([survey 03](research/03_signals_provenance_evaluation.md)).
Every JPEG-domain and metadata signal (ELA, double-JPEG, quantization tables, EXIF, editor
markers) collapses once an image has been re-encoded by a social platform, but they remain
cheap, explainable evidence on un-laundered files. C2PA verification via `c2pa-python`
and decoding of the `invisible-watermark` scheme used by Stable Diffusion are the
provenance checks worth having. A missing C2PA manifest proves nothing.

## 3. Design principles

1. **Evidence, not verdicts.** Every component returns a `DetectionResult` with a score,
   a label that can be `uncertain`, an optional heatmap and a `details` dictionary that
   explains *why*. The fusion layer may abstain.
2. **Honest evaluation.** Test generators never appear in training. Report AUC *and*
   accuracy at a threshold fixed on validation *and* the false-positive rate on real
   images. Publish a robustness table (JPEG, resize, crop, WEBP) and a per-generator-year
   decay chart. Include trivial baselines in every table.
3. **License-clean by default.** Default dependencies and shipped weights are MIT,
   Apache-2.0 or CC-BY only. Non-commercial models are either excluded or isolated in a
   clearly marked optional extra used for benchmarking only.
4. **Crop, never resize; augment always.** Enforced in the shared preprocessing code, not
   left to individual detectors.
5. **Reproducibility.** Dataset manifests with file hashes, versioned YAML configs, a
   deterministic robustness suite whose parameters live in the repo.
6. **Small, tested increments.** Each step is a reviewable change with tests; the
   changelog is updated with every user-visible addition.

## 4. Target architecture

| Module | Responsibility | Primary approach |
|---|---|---|
| `imgforensics.core` | `DetectionResult`, `BaseDetector`, registry | done in v0.1.0 |
| `imgforensics.signals` | Classical, CPU-only, explainable checks | EXIF/thumbnail/quant-table consistency, C2PA verify, ELA, block-based copy-move, double-JPEG / JPEG ghost, invisible-watermark decode |
| `imgforensics.detectors` | Whole-image AI-generated classifier | Frozen DINOv2 ViT-B/14 features + light multi-layer head (RINE-style); optional SPAI weights as ensemble member |
| `imgforensics.localization` | Pixel-level manipulation heatmaps | Inference wrappers for CAT-Net v2 and IML-ViT (SAFIRE optional); own DINOv2 + LoRA patch localizer for inpainting |
| `imgforensics.data` | Dataset registry, subset downloads, manifests, bias audit | Community Forensics Small slice, COCO, Synthbuster, ITW-SM/WildRF, TGIF subset, CASIA v2 corrected, CocoGlide, AutoSplice |
| `imgforensics.fusion` | Combine heterogeneous scores, calibrate, explain | Calibrated logistic stacking fitted on validation data; abstain band; report builder |
| `imgforensics.eval` | Metrics, protocols, robustness suite, benchmark runner | AUC, accuracy at fixed threshold, FPR, ECE, pixel F1@0.5 / best-F1 / AP, IoU |
| CLI / API / demo | `imgforensics analyze`, FastAPI service, Gradio demo | Added in the final phase |

## 5. Phases

Order rationale: the classical signals need no GPU or dataset and give a working end-to-end
CLI quickly, while dataset downloads and the evaluation harness are built in parallel.
The learned detectors come only after the evaluation harness exists, so every model is
measured the same way from its first training run.

### Phase 0 — Foundation (current)

- Package skeleton, core types, registry, CLI stub, CI, tests.
- Literature surveys and this roadmap.
- **Done when:** CI is green on GitHub and `imgforensics analyze` runs on any image.

### Phase 1 — Classical signals and explanation cards

- Metadata module: EXIF/XMP parsing, embedded-thumbnail vs image comparison, editor
  markers (Photoshop APP13, XMP history), quantization-table extraction and
  camera/software table matching.
- C2PA manifest verification (`c2pa-python`).
- Error Level Analysis with heatmap output.
- Block-based copy-move detection with matched-region heatmap.
- Double-JPEG / JPEG-ghost analysis on top of DCT coefficients (`jpegio`).
- Invisible-watermark decode pass (Stable Diffusion scheme).
- Every signal registered as a detector; CLI prints per-signal cards; JSON output flag.
- **Done when:** each signal has unit tests on synthetic fixtures (self-made spliced and
  re-saved images) and documented failure modes, including behaviour on laundered images.
- **Result (2026-09-09):** seven signals shipped (`metadata`, `ela`, `c2pa`, `sd_watermark`,
  `copy_move`, `jpeg_ghost`, `double_jpeg`), 93 tests, all seven run in about 5.5 s on a
  12-megapixel JPEG. Notable findings: the Stable Diffusion watermark lives only in chroma
  and does not survive any JPEG save; the blocking-grid check decays into noise above
  quality 85; only coarse-then-fine double compression is detectable; ELA and JPEG ghost
  are capped so they can never assert "fake" alone.

### Phase 2 — Data and evaluation infrastructure

- Dataset registry with download-subset scripts and manifests (paths, labels, masks,
  source hashes, license, `commercial_ok` flag so a future retrain can be restricted to
  commercially usable data with one filter). Disk budget about 60–90 GB, see section 6.
- Bias audit tool: format, resolution and JPEG-quality distribution per class; refuses
  to build a split whose classes differ in format.
- Shared preprocessing: native-resolution crops, deterministic augmentation config.
- Deterministic robustness suite: JPEG QF {95, 85, 75, 60, 50}, resize {1.0, 0.75,
  0.5, 0.25}, crop, WEBP, Gaussian noise; parameters in one YAML file.
- Metrics and benchmark runner producing Markdown tables.
- **Done when:** a trivial baseline and one public pretrained model (SIDBench member or
  SPAI) run through the full protocol and the tables are committed.
- **Result (2026-09-09):** manifests with a `.meta.json` sidecar, a registry of 26 datasets
  verified against primary sources with `commercial_ok` and `verified_on` fields, the bias
  audit (Jensen-Shannon distance on format, resolution, JPEG quality and aspect ratio plus
  duplicate detection), pure-numpy metrics with a strict `score > threshold` convention so
  an abstaining 0.5 is not a fake call, a 15-level deterministic robustness suite,
  training-time augmentation helpers, trivial baselines, the benchmark runner with Markdown
  reports, and license-gated dataset fetching with layout adapters and manifest sampling.
  360 tests. The pretrained-model part of the exit criterion moves to Phase 3, where the
  first learned detector is benchmarked through this harness.

### Phase 3 — Whole-image AI-generated detector

- Feature caching for frozen DINOv2 ViT-B/14 (and CLIP ViT-L/14 for comparison).
- Multi-layer head trained on a stratified slice of Community Forensics Small plus COCO
  reals, with crop-not-resize and JPEG/WEBP augmentation. Local GPU for the head;
  Colab/Kaggle for larger sweeps.
- Optional ensemble member: SPAI released weights (Apache-2.0).
- Post-hoc calibration; weights published to Hugging Face Hub.
- **Done when:** cross-generator results on Synthbuster, ITW-SM/WildRF (and Chameleon if
  access is granted), the robustness table and the per-year decay chart are in
  `docs/benchmarks/`.

### Phase 4 — Manipulation localization

- 4a. Inference wrappers for CAT-Net v2 and IML-ViT (SAFIRE optional), tiled inference,
  heatmap resampling and a simple ensemble.
- 4b. Own DINOv2 ViT-B/14 + LoRA patch localizer for inpainting, trained on a TGIF
  subset, CASIA v2 (corrected masks) and a small in-house inpainting-on-COCO set with
  released masks, prompts and scripts. The in-house set is generated with Stable
  Diffusion or FLUX.1-schnell (Apache-2.0), not FLUX.1-dev (non-commercial). Fallback if
  the reference code stays unpublished: SegFormer-B2 fine-tuning.
- **Done when:** MVSS-protocol results plus CocoGlide / AutoSplice / TGIF-test results,
  reported as F1@0.5, best-F1 and AP, with a per-mask-size breakdown.

### Phase 5 — Fusion and explanation

- Calibrated logistic stacking over signal, detector and localizer scores, fitted on a
  held-out validation split, with an explicit abstain band.
- Report builder: JSON report, heatmap overlay images, Grad-CAM for the classifier,
  per-signal cards with plain-language notes.
- **Done when:** the fused score beats the best single component on the held-out test
  sets without hurting the false-positive rate, and the report renders for any input.

### Phase 6 — Product and release

- FastAPI service, Gradio demo (Hugging Face Space), ONNX export for CPU inference.
- Documentation site or extended README with benchmark report and limitations section.
- v1.0 tag.

## 6. Compute and disk plan

| Resource | Use |
|---|---|
| Local RTX 2060 (6 GB) | All classical signals, feature caching, head training, LoRA training at batch 2 with AMP, inference for every shipped model |
| Colab / Kaggle | Larger sweeps, SDXL/FLUX inpainting for the in-house set, anything above 6 GB |
| Disk (target 60–90 GB) | Community Forensics Small slice (20–30 GB), COCO train2017 (19 GB), Synthbuster + RAISE-1k, ITW-SM/WildRF (under 5 GB), TGIF subset (about 15 GB), CASIA v2 corrected (1.5 GB), CocoGlide + AutoSplice (about 5 GB), in-house inpainting set (about 3 GB) |

Never download raw GenImage (about 500 GB), full tampCOCO, GIM or DEAL-300K.

## 7. Licensing policy

This is a personal, non-commercial, open-source research project. That widens what the
project may *use*, but not what it may *redistribute*: the repository is MIT-licensed, so
anything committed to it must be MIT-compatible. Third-party weights and datasets are
therefore never committed; users download them from the original source through a script
that prints the license and requires explicit acceptance. `THIRD_PARTY_NOTICES.md` lists
every external model, dataset and library with its license and how it is used.

| Component | License | Decision |
|---|---|---|
| Own code, configs, scripts, docs | MIT | Committed |
| CAT-Net v2, SAFIRE, SPAI, IML-ViT | Apache-2.0 / MIT / CC-BY-4.0 | Default optional extras; weights downloaded from source, never committed |
| TruFor, Noiseprint++ | Non-commercial (research use) | Allowed for this non-commercial project as a clearly labeled optional extra and benchmark baseline; wrapper code is MIT, weights are user-downloaded, README states the restriction |
| B-Free | Non-commercial code | Idea reimplemented in MIT code if used; original code is not vendored |
| MVSS-Net weights | Not stated (all rights reserved by default) | Benchmark comparison only, downloaded from the authors' link; not shipped or redistributed; authors asked for terms |
| Community Forensics, CASIA, CocoGlide, AutoSplice, Synthbuster, Chameleon | Research / academic use | Used for training and evaluation only; images never redistributed; cite |
| TGIF / TGIF2 | CC BY-SA 4.0 | Used for training and evaluation; the ShareAlike clause is why no derived image set is redistributed |
| In-house inpainting set built on COCO | COCO images carry mixed Flickr licenses | Release masks, prompts, image IDs and generation scripts only, not images |
| jpegio, c2pa-python, prnu-python, invisible-watermark, exiftool | Apache-2.0 / MIT / GPL (exiftool, called as an external binary) | Allowed; exiftool is invoked as a subprocess, not linked, so the GPL does not extend to this code |

**Keeping a commercial path open.** The project is non-commercial today, but nothing in
the design should close the door:

1. Every dataset and model entry carries a `commercial_ok` flag; training configs can
   filter on it, so a commercially clean model is a retrain on a filtered subset, not a
   rewrite. Weights trained on research-only data are treated as research-only.
2. Research-only models live in a separate optional extra (`pip install
   "imgforensics[research]"`); the default install contains only permissively licensed
   components.
3. The author remains the sole copyright holder unless outside contributions are accepted
   under a contributor agreement (DCO sign-off at minimum), which preserves the freedom to
   relicense future versions.

## 8. Risks and fallbacks

- **Detector aging.** Accuracy on new generators will fall; the temporal decay chart is
  published rather than hidden, and the training slice is refreshable.
- **Reference code availability.** If the DINOv2 + LoRA localizer code is not public,
  reimplement the recipe from the paper or fall back to SegFormer-B2.
- **Dataset access.** Chameleon requires an academic email request; the plan does not
  depend on it.
- **Windows tooling.** Some research repos assume Linux; wrappers are written against
  exported weights, not the original training code, and CI runs on Linux.
- **Disk.** Manifest-driven subsets keep the footprint under 90 GB.

## 9. Open questions

1. Availability of DinoLizer code and weights (re-check before Phase 4b).
2. MVSS-Net license (email the authors before Phase 4a).
3. SAFIRE behaviour on diffusion inpainting (no published CocoGlide number; measure it).
