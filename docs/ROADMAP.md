# Roadmap

Last updated: 2026-09-10. This document turns the three literature surveys in
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
- **Result of experiment 01 (2026-09-10):** frozen DINOv2 ViT-B/14 plus a 1.06 M-parameter
  multi-layer head, trained on 8 shards of Community Forensics Small with a generator-disjoint
  split, reaches AUC 1.000 on 42 unseen generators of the same family and then fails outside
  it: AUC 0.459 on WildRF and a 99.9% false-positive rate on COCO photographs. The head learned
  the training set's real-image sources, not generation artifacts; see
  [`docs/benchmarks/01_experiment_summary.md`](benchmarks/01_experiment_summary.md). The
  harness caught it because foreign test sets are part of the protocol. Experiment 02 changes
  the data (diverse reals, augmented-only training views, mixed-source calibration), not the
  model. Also found: the crop-never-resize policy breaks below the crop size (AUC 0.57 at
  quarter scale), and the classical signals are the benchmark bottleneck (about 270 ms per
  512 px image on one core) and need parallel workers.
- **Extraction throughput (2026-09-10):** augmenting a 2x crop window instead of the whole
  image makes extraction 2.8x faster on 2,048 px images (1.25 to 3.45 images/s) and about 10%
  slower on 512 px ones; the `--workers` pool changes nothing on either, so the remaining
  bottleneck is the main-process loop around the backbone, not decoding. See
  [`docs/benchmarks/05_feature_extraction_timing.md`](benchmarks/05_feature_extraction_timing.md).
- **Result of experiment 02 (2026-09-10):** same model, data changed (COCO and WildRF-train
  reals added, augmented-only training views, mixed-source calibration). COCO false-positive
  rate 99.9% to 2.7%, WildRF test AUC 0.459 to 0.980 (balanced accuracy 0.916 at 0.5),
  same-family val AUC 1.000, CocoGlide local edits AUC 0.644 (a whole-image head does not
  see small inpainted regions; that is Phase 4's job). WildRF is held-out images of a
  distribution seen in training; the genuine cross-dataset tests (Synthbuster, ITW-SM) are
  pending. See [`docs/benchmarks/02_experiment_summary.md`](benchmarks/02_experiment_summary.md).
- **Result of experiment 03 (2026-09-10):** experiment 02 with the WildRF train split
  removed and nothing else changed, so the WildRF test sample becomes a genuine
  cross-dataset test: AUC 0.804 (0.980 in-distribution), false-positive rate on laundered
  social-media photographs 54.7% at 0.5 (13.7%), true-positive rate 89.3% (97.0%), COCO
  false-positive rate unchanged at 2.6%. The head reads platform laundering as evidence of
  generation; the real-image distribution decides generalization, as in experiments 01 and
  02. The shipped default stays the experiment 02 head. See
  [`docs/benchmarks/06_experiment_03_summary.md`](benchmarks/06_experiment_03_summary.md).
- **Result of the Synthbuster test (2026-09-10):** nine generator families, five of them
  never seen in training (DALL-E 2 and 3, Firefly, GLIDE, Midjourney v5), 200 images each
  against the 1,000 held-out COCO photographs, at every robustness level. With both classes
  re-encoded to JPEG (`jpeg_q75`): shipped head AUC 0.969, 80.4% of fakes caught at 0.5 with
  a 3.3% false-positive rate; the experiment 03 head 0.980 / 89.3% / 3.6%. Per family at
  that level the shipped head catches 65-98% (DALL-E 3 easiest, Midjourney v5 and DALL-E 2
  hardest). The clean level is inflated by Synthbuster's PNG-versus-JPEG split (0.986); the
  JPEG rows are the ones to quote. The head trained without WildRF is the better curated-output
  detector and the worse social-media one, two operating points on one trade-off. See
  [`docs/benchmarks/07_synthbuster_summary.md`](benchmarks/07_synthbuster_summary.md).

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
- **Result of 4a's first slice (2026-09-10):** the IML-ViT model definition is vendored for
  inference (MIT, `SunnyHaze/IML-ViT` @ `07dd2be`, `fvcore`/`albumentations` and all
  training code removed), the released 350 MB checkpoint is fetched through a license-gated
  `imgforensics weights fetch`, and `iml_vit` is registered as a localizer that pads (never
  resizes) to 1024 and tiles anything larger at stride 768 instead of truncating it the way
  the upstream transform does. On all 1,024 CocoGlide images
  ([`docs/benchmarks/03_cocoglide_iml_vit.md`](benchmarks/03_cocoglide_iml_vit.md)):
  pixel F1@0.5 **0.059**, best-F1 **0.486**, AP **0.423**, IoU **0.037**, image-level AUC
  **0.535**. A predict-everything baseline scores F1 0.355 / AP 0.252 / IoU 0.252 on the same
  masks, so the model beats it on ranking (AP, best-F1) and loses to it badly at a fixed 0.5
  threshold: it ranks inpainted pixels better than chance but almost never calls one, which is
  a calibration failure on top of a domain-transfer failure. This is the CocoGlide gap section 2
  predicted, measured rather than assumed, and it is why the protocol insists on reporting
  F1@0.5, best-F1 and AP side by side — any one of the three alone tells a different story.
  Per mask area, best-F1 runs 0.316 (small, <10%) / 0.457 (medium) / 0.711 (large); against the
  trivial baseline's 0.104 / 0.312 / 0.684, the model adds most where the edit is smallest and
  almost nothing where it is largest, the opposite of the usual "the small bin is where
  everything fails". Cost on the 6 GB RTX 2060: 1.6 GB peak allocated (2.6 GB reserved) and
  0.3–0.5 s per 1024 px tile at batch 1 under fp16 autocast, independent of image size.
  Next in 4a: CAT-Net v2 (whose DCT stream may transfer differently) and a localizer ensemble;
  a classic splicing localizer alone is not a usable inpainting detector.
- **Result of 4a's second slice (2026-09-10):** CAT-Net v2 is vendored the same way
  (Apache-2.0 code + CC-BY-4.0 weights, `mjkwon2021/CAT-Net` @ `331b805`, the `yacs` config
  machinery and all training code removed) and registered as `catnet_v2`, and the DCT stream it
  needs is fed by a pure-numpy baseline-JPEG entropy decoder written for this project
  (`_jpegcoef.py`) because `jpegio` publishes no Windows wheel and none past CPython 3.10.
  **The DCT stream transfers, and it transfers well.** On the same 1,024 CocoGlide images
  ([`docs/benchmarks/03_cocoglide_catnet.md`](benchmarks/03_cocoglide_catnet.md)): pixel F1@0.5
  **0.364** (IML-ViT 0.059), best-F1 **0.605** (0.486), AP **0.566** (0.423), IoU **0.288**
  (0.037), image-level AUC **0.666** (0.535). It beats IML-ViT on every metric measured and, unlike
  IML-ViT, it also beats the predict-everything baseline (F1 0.355 / AP 0.252 / IoU 0.252) at the
  fixed 0.5 threshold rather than only on ranking -- so the calibration failure the first slice
  found is IML-ViT's, not the domain's. Per mask area its best-F1 runs 0.387 / 0.640 / 0.824
  (small < 10% / medium / large) against IML-ViT's 0.316 / 0.457 / 0.711, and the gap is widest in
  the medium bin. Two caveats keep this from being a verdict. First, CocoGlide's images are PNGs, so
  every one of them is re-encoded to a quality-100 JPEG before inference exactly as upstream's own
  demo does; the DCT stream is reading a compression history this project created, not one the
  editor left, which is the opposite of the setting CAT-Net was trained for and makes the result a
  lower bound rather than a like-for-like measurement. Second, the image-level AUC of 0.666 comes
  from a heatmap statistic, not a detection head: the model fires on authentic and inpainted images
  alike (mean top-1% score 0.83 vs 0.83 on a 12-image probe) and only the *shape* of the map
  separates them, which is why the pixel numbers move so much further than the image-level one.
  Cost on the 6 GB RTX 2060: 1.0 GB peak allocated (1.2 GB reserved) at 1024x1024 under fp16
  autocast, flat above one tile; 549 ms per 256 px CocoGlide image, of which roughly 340 ms is the
  Python JPEG decoder rather than the network. That decoder is now the bottleneck for this
  localizer and is the first thing to profile if 4a's ensemble becomes routine.
  Next in 4a: the localizer ensemble over `iml_vit` + `catnet_v2`, and re-running both under the
  JPEG re-compression robustness axis, where a DCT-stream model is expected to move most.
- **Result of 4a's third slice (2026-09-10):** `localizer_ensemble` with three parameter-free
  rules, and pixel metrics at every geometry-preserving robustness level. No combination beats
  CAT-Net alone: `max` matches it (F1@0.5 0.385 vs 0.364, best-F1 0.608 vs 0.605), `mean`
  keeps the ranking but loses two thirds of the F1@0.5 to IML-ViT's near-zero probabilities,
  `rank_mean` wins the fixed threshold by construction and loses the ranking; `max` is the
  default. Under JPEG re-compression CAT-Net's map *improves* (best-F1 0.598 clean to 0.631
  at q85, image AUC 0.633 to 0.738) because the DCT stream finally reads a real JPEG history,
  holds to q50, and comes apart under WEBP and sigma-5 noise; IML-ViT is flat at every level.
  See [`docs/benchmarks/05_cocoglide_ensemble_summary.md`](benchmarks/05_cocoglide_ensemble_summary.md).
  Phase 4a is complete; 4b (the trained inpainting localizer) is the open half.

### Phase 5 — Fusion and explanation

- Calibrated logistic stacking over signal, detector and localizer scores, fitted on a
  held-out validation split, with an explicit abstain band.
- Report builder: JSON report, heatmap overlay images, Grad-CAM for the classifier,
  per-signal cards with plain-language notes.
- **Done when:** the fused score beats the best single component on the held-out test
  sets without hurting the false-positive rate, and the report renders for any input.
- **Result of fusion 01 (2026-09-10):** a pure-numpy calibrated stacking fuser
  (`imgforensics fusion fit`, `analyze --fuser`) fitted on the WildRF val split over the
  experiment 02 head and the seven signals. On the WildRF test sample: AUC 0.981 vs 0.980 for
  the head alone, false-positive rate 13.7% to 3.6% at 0.5, and outside the abstain band
  (53% of images) balanced accuracy 0.996. `jpeg_ghost` and `copy_move` received negative
  weights on this data, a reminder that a fuser is a dataset-specific calibration layer. See
  [`docs/benchmarks/04_fusion_wildrf.md`](benchmarks/04_fusion_wildrf.md).
- **Result of fusion 02 (2026-09-10):** the same fuser refitted on the experiment 03 head
  (WildRF never seen in training). On the WildRF test sample: AUC 0.804 to 0.831,
  false-positive rate 54.7% to 23.8%, ECE 0.216 to 0.044, recall 89.3% to 73.8%; the head's
  weight falls from 0.86 to 0.29 and the JPEG-domain signals take over. The band as first
  fitted abstained on 96.9% of the images on the strength of three held-out images, which
  exposed a gap in the band search; with the minimum-support rule now in `fit_fuser` (at
  least 20 held-out images outside the band) the refitted band calls 20.2% of the test
  images at balanced accuracy 0.887 and abstains on the rest. See
  [`docs/benchmarks/06_experiment_03_summary.md`](benchmarks/06_experiment_03_summary.md).
  `imgforensics fusion eval` now reproduces every fusion table from saved records. Shipped
  since fusion 01: the explanation report with heatmap overlays and Grad-CAM attribution.
- **Result of fusion 03 (2026-09-10):** the head and the seven signals scored at all 15
  robustness levels on WildRF val and test. The head's false-positive rate on laundered reals
  rises from 13.7% clean to 31% at half scale and 23% under the social re-share level; either
  fuser brings every level back to 4-10% at an unchanged AUC. A fuser fitted on clean records
  only already generalizes across the levels (within 0.03 of the levels-fitted one's
  false-positive rate everywhere); fitting on the levels buys a little false-positive rate for
  a much wider abstain band (calls 20-31% of images instead of 42-56%). The shipped default
  stays the fusion 01 fuser; quarter-scale resize remains the wall for head and signals
  alike. See [`docs/benchmarks/05_fusion_levels_summary.md`](benchmarks/05_fusion_levels_summary.md).
  Phase 5's exit criterion is met: the fused score beats the best single component on every
  held-out set without hurting the false-positive rate, and the report renders for any input.

### Phase 6 — Product and release

The end product is an **interactive forensic workbench**: one image, a tree of tools grouped
by what they look at, every tool's map sharing one pan/zoom with the original, calibrated
scores with the abstain band always visible, and a report export. The reference for the
experience is Sherloq (GPL-3.0; ideas only, nothing ported). The architecture, the tool
catalogue and the client decision (web, taken 2026-09-10) are in
[`docs/design/01_toolbox_architecture.md`](design/01_toolbox_architecture.md).

- 6a. Service layer inside the library: tool catalogue with parameter specs, analysis
  sessions with cached results and map pyramids, a `view` tool kind for maps without a
  verdict. Headless and tested without any GUI.
- 6b. FastAPI wrapper (`api` extra) over the service; OpenAPI as the client contract.
- 6c. Gradio stopgap demo on a Hugging Face Space; head weights published to the Hub under
  a research-only card.
- 6d. The web workbench client, then the demo is retired.
- 6e. CPU inference: ONNX export of the head, a faster JPEG coefficient decoder.
- Documentation site or extended README with benchmark report and limitations section.
- v1.0 tag.
- **Result of 6a-6d (2026-09-10):** shipped in one day on top of the headless library.
  6a: `ParameterSpec`/`parameters()` and a `kind` on every tool, the `view` kind with
  `luminance_gradient`, `noise_residual` and `bit_planes`, and `imgforensics.service`
  (`catalogue()`, `AnalysisSession` with a parameter-keyed result cache and map pyramids,
  `SessionStore` with a TTL and an LRU cap). 6b: `imgforensics.api` (`api` extra) with the
  eight routes of the design note and `imgforensics serve`. 6c: `imgforensics.demo`
  (`demo` extra), `imgforensics demo` and the `spaces/` entry point for a Hugging Face
  Space; the Space itself and the head's model card are not published yet (they need the
  owner's Hub credentials). 6d: the workbench client, plain HTML/CSS/ES modules served by
  the API at `/` with no build step -- synced pan/zoom across map panels, tiles from the
  pyramid level matching the zoom, parameter controls from the specs, the fused verdict
  with its band, and the report download -- checked in a browser against the real
  models. Test count 551 to 836. The Gradio demo stays the public link until the
  workbench is deployed behind a proxy; 6e (CPU inference) is the next milestone.

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

**Licensing provenance.** The project is non-commercial and is expected to stay so. The
bookkeeping below exists so that the licensing of anything the project produces can always
be traced, not as preparation for a change of purpose:

1. Every dataset and model entry carries a `commercial_ok` flag, and a trained checkpoint
   inherits that flag ANDed across its training manifests, so a weight file always says
   which terms its data carried. Weights trained on research-only data are research-only.
2. Research-only models live in a separate optional extra (`pip install
   "imgforensics[research]"`); the default install contains only permissively licensed
   components.
3. Outside contributions, if any are accepted, come in under a DCO sign-off so the
   provenance of the code stays as clear as that of the data.

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
