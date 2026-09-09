# 02 — Image Manipulation Localization (splicing / copy-move / removal / AI inpainting)

Literature survey for `imgforensics`, component (2) locally AI-edited images and (3) classic manual manipulations.
Scope: pixel-level mask/heatmap + the image-level decision derived from it. Compiled 2026-09-09.

**Evidence markers used below**
- `[verified]` — I opened the page (GitHub repo, arXiv abstract, docs page) during this survey.
- `[search]` — the claim appeared in a search-engine snippet with a resolvable URL, but I did not open the page.
- `[memory]` — from prior knowledge, not re-checked. Treat as a hypothesis to confirm before you act on it.
- VRAM columns are **engineering estimates** (marked *est.*) derived from backbone size and input resolution, not measured numbers. Nobody publishes these; measure them yourself before committing.

---

## 1. TL;DR

1. The field split in two: **classic IMDL** (splicing/copy-move/removal, CASIA-style benchmarks) and **generative-edit localization** (diffusion inpainting). Models tuned for the first generalize poorly to the second — CocoGlide exists precisely to expose that gap `[verified: TruFor repo/paper]`.
2. **CAT-Net relicensed to Apache-2.0 (code) + CC-BY-4.0 (weights & datasets) as of Aug 2026** — it is now the strongest *commercially usable* pretrained localizer with public weights `[verified]`. This is the single most decision-relevant licensing fact in this document.
3. **TruFor is non-commercial** ("informational and nonprofit purposes") `[verified]`, confirmed independently by PhotoHolmes, which warns that using TruFor limits the whole library to non-profit use `[verified]`. Do not ship it in a portfolio repo with a permissive LICENSE.
4. **IMDL-BenCo** (NeurIPS'24 Spotlight, `scu-zjz/IMDLBenCo`, CC-BY-4.0, `pip install imdlbenco`) is the right training/eval harness: standardized protocols, GPU-accelerated metrics, robustness suite, released checkpoints `[verified]`. Its 2025 successor **ForensicHub** (NeurIPS 2025, CC-BY-4.0) widens scope to deepfake + AIGC + document domains `[verified]`.
5. For **AI inpainting specifically**, the most transferable recent recipe is **DinoLizer**: DINOv2 + LoRA on q/v + a linear head predicting at 14×14 patch resolution, sliding-window over large images; reports ~12–20% higher IoU than the next best model on inpainting sets and robustness to JPEG `[verified, arXiv 2511.20722]`. This is also the only SOTA-ish recipe that plausibly *trains* on 6 GB.
6. A 2025 study found synthetic-image detectors transfer **partially** to inpainting edits — fine on medium/large regions, weak on small ones `[verified, arXiv 2512.16688]`. Practical consequence: your full-AI detector (component 1) is a useful *prior* for component 2 but not a substitute.
7. The **fixed vs. optimal threshold** issue is real and makes cross-paper numbers incomparable: some works pick a per-image best threshold, some a per-dataset best, some fix 0.5 `[search]`. Report fixed-0.5 F1 **and** a threshold-free metric (AP / best-F1) side by side, always.
8. Dataset hygiene matters more than architecture at your scale: CASIA v2 ground truth is noisy (there is a community-corrected mask set, `SunnyHaze/CASIA2.0-Corrected-Groundtruth`) `[search]`, and CASIA v2 has a **format shortcut** — tampered images skew TIFF, authentic are JPEG/BMP `[search]`.
9. VLM/MLLM localizers (FakeShield 22B, SIDA-7B/13B, ForgeryGPT, LEGION) give explanations but are far outside 6 GB `[verified for FakeShield: `zhipeixu/fakeshield-v1-22b`, Apache-2.0]`. A 2026 paper argues VLM priors barely help localization at all `[verified, arXiv 2603.12930]`.
10. Recommended path: **inference-first with CAT-Net + MVSS-Net + (optionally) SAFIRE as an ensemble of pretrained localizers**, then **one trainable contribution**: a DINOv2-ViT-B/14 + LoRA patch-level inpainting localizer trained on a TGIF/CocoGlide subset plus a small in-house SD-inpainting-on-COCO set.

---

## 2. Problem framing

### 2.1 Manipulation taxonomy and what each leaves behind

| Type | Definition | Primary forensic trace | Difficulty |
|---|---|---|---|
| **Splicing** | Region pasted from a *different* image | Camera/PRNU mismatch, double-JPEG grid misalignment, noise-residual discontinuity, boundary artifacts | Easiest — two different imaging pipelines meet |
| **Copy-move** | Region pasted from the *same* image | No noise/camera mismatch; needs self-similarity matching (keypoints, self-correlation) | Hard for noise-based detectors, easy for correlation-based ones |
| **Removal / classic inpainting** | Object erased, hole filled by patch-match or classic inpainting | Texture over-smoothing, missing high-frequency detail, cloned texture | Medium |
| **Generative fill / diffusion inpainting** | Hole filled by SD/SDXL/FLUX/Firefly conditioned on a prompt | *No* second camera, *no* second compression if re-encoded once; only VAE/decoder statistics and frequency-domain fingerprints | **Hardest** |
| **Instruction-based full-image editing** | Mask-free editors (InstructPix2Pix-style, mask-free diffusion editors) regenerate large parts | The "edited region" is fuzzy and may cover most of the image; mask supervision itself becomes ill-defined | Hard, and the *label* is the problem — see DEAL-300K's active-learning change-detection annotation pipeline `[verified, arXiv 2511.23377]` |

### 2.2 Why AI inpainting is harder than classic splicing

Classic IMDL leans on **pipeline mismatch**: two images with different camera fingerprints, JPEG histories and noise-residual statistics get stitched together. Noiseprint++, DCT/JPEG streams (CAT-Net) and SRM residuals (ManTraNet, MVSS-Net) all exploit exactly this. Diffusion inpainting destroys the premise:

- Content is generated **inside the same image**, saved once → one camera history, one compression history.
- Modern pipelines re-encode the *whole* image through the VAE (TGIF ships this "fully-regenerated" variant alongside the composited one `[verified]`), so even untouched pixels are no longer camera-native — a noise-mismatch detector sees a *uniform* anomaly and fires everywhere or nowhere.
- The blend is semantically and photometrically consistent by construction, so boundary cues are weak.

Useful signal therefore shifts to **generator fingerprints** (VAE reconstruction statistics, frequency artifacts) and **self-consistency**. Hence the 2024–2026 wave: DIRE/FIRE reconstruction-error cues, End4 `[search]`, DiffusionPrint (arXiv 2604.12443) `[search]`, and DinoLizer's insight that a *synthetic-image-detection* backbone beats an IMDL backbone as a starting point `[verified]`. DinoLizer's subtler point: reconstructed-but-outside-mask pixels are a **separate class** `[verified]` — training "inpainted vs. not" on regenerated images with the composite mask means training on partially wrong labels.

### 2.3 The fixed-threshold vs. optimal-threshold controversy

Three incompatible conventions coexist in published F1 numbers `[search]`:

| Convention | What is tuned | Effect |
|---|---|---|
| **Fixed 0.5** | nothing | Realistic; MVSS-Net's repo defaults to 0.5 `[verified]`. Penalizes miscalibrated models. |
| **Per-dataset optimal** | one threshold per test set | Mildly optimistic; assumes you know the test distribution. |
| **Per-image optimal / best-F1** | one threshold per image | Strongly optimistic — a model with a *good ranking but no calibration* scores like a solved problem. Justified by authors as "measuring localization ability decoupled from threshold selection" `[search]`. |

Threshold-free alternatives (**AP**, **AUC**) are also inflated here because background dominates — pixel AUC on a 2%-tampered image is near-saturated by predicting all-negative. **Report IoU and F1@0.5 as primary, AP as secondary, and never compare your F1 to a paper's without checking its convention.**

### 2.4 Image-level vs. pixel-level evaluation

**Pixel-level** (F1/IoU/AUC/AP over the mask) is only defined on tampered images — including authentic images makes F1 undefined (empty GT). **Image-level** ("is this manipulated at all?") comes either from a dedicated detection head (MVSS-Net, TruFor, HiFi-Net all have one) or from pooling the mask (max, top-k mean, area-above-threshold). The trap: a localizer with high pixel F1 on tampered-only sets can be useless as a detector because it hallucinates masks on authentic images. **Always evaluate on a mixed authentic+tampered set** and report image-level AUC/F1 separately — reducing false alarms on authentic images is MVSS-Net's entire design motivation `[verified: repo reports both]`.

---

## 3. Method families

Legend: **W?** = pretrained weights publicly downloadable. VRAM = *est.*, single 3-channel image, fp32/AMP, at the model's native resolution.

| Method | Year / venue | Core idea | Input cues | Architecture | Training data | Code | W? | License (commercial?) | VRAM inf. *est.* | VRAM train *est.* |
|---|---|---|---|---|---|---|---|---|---|---|
| **ManTraNet** | 2019 CVPR | Anomaly detection vs. local feature "manipulation-trace" embedding, 385 manipulation types pretext | RGB + SRM + Bayar conv | VGG-ish featex + Z-score anomaly + ConvLSTM | Synthetic 385-class | `ISICV/ManTraNet` `[memory]` | Yes (Keras) `[memory]` | Unclear/none stated `[memory]` | ~2 GB | ~8 GB |
| **SPAN** | 2020 ECCV | Spatial pyramid attention over local self-attention blocks | RGB + SRM | Pyramid self-attention | SP-COCO synthetic | Partial `[memory]` | Limited `[memory]` | Unclear `[memory]` | ~2 GB | ~10 GB |
| **MVSS-Net / MVSS-Net++** | 2021 ICCV / 2022 TPAMI | Multi-**view** (noise + edge) multi-**scale** supervision, explicit detection branch to cut false alarms | RGB + SRM noise + edge supervision | Dual-branch ResNet-50 FCN + DAM/GSR | CASIAv2 or DEFACTO-84k | `dong03/MVSS-Net` `[verified]` | **Yes** (Google Drive + Baidu) `[verified]` | Not clearly stated in repo — check before shipping `[verified: no explicit license surfaced]` | ~1.5–2 GB @512 | ~8–12 GB |
| **CAT-Net / CAT-Netv2** | 2021 WACV / 2022 IJCV | Two-stream: RGB stream + **DCT coefficient / JPEG quantization** stream; learns compression artifacts | RGB + DCT coeffs + Q-tables | HRNetV2 ×2 streams | CASIAv2, FantasticReality, IMD2020, **tampCOCO**, **compRAISE** | `mjkwon2021/CAT-Net` `[verified]` | **Yes** (GDrive/Baidu) `[verified]` | **Apache-2.0 code, CC-BY-4.0 weights+data (Aug 2026) → commercial OK with attribution** `[verified]` | ~3–5 GB @1024 | ~16–24 GB |
| **PSCC-Net** | 2022 TCSVT | Progressive spatio-channel correlation, coarse→fine masks at 4 scales | RGB | HRNet-ish + dense cross-connections | Synthetic (splice/copy-move/removal) | `proteus1991/PSCC-Net` `[memory]` | Yes `[memory]` | Unclear `[memory]` | ~2 GB | ~10 GB |
| **TruFor (+ Noiseprint++)** | 2023 CVPR | RGB + learned **Noiseprint++** camera fingerprint, cross-modal transformer, outputs mask + **confidence map** + integrity score | RGB + Noiseprint++ | SegFormer-B2 encoder, anomaly + confidence decoders | Mixed (incl. synthetic splices) | `grip-unina/TruFor` `[verified]` | Yes; training code since Mar 2025, Noiseprint++ training still unreleased `[verified]` | **Non-commercial only** `[verified]` | ~2–4 GB @1024 | ~16 GB |
| **HiFi-Net** | 2023 CVPR | Hierarchical *fine-grained* forgery attribute classification (4 levels) + localization | RGB + frequency | Multi-branch extractor + hierarchical heads | HiFi-IFDL (13 forgery methods) | `CHELSEA234/HiFi_IFDL` `[search]` | Yes `[search]` | Unclear `[search]` | ~3 GB | ~16 GB |
| **IML-ViT** | 2023 arXiv (widely used baseline) | Plain ViT + **high resolution (1024 zero-padded)** + multi-scale + **edge supervision**; "ViT is enough" | RGB only | ViT-B MAE + simple FPN | CASIAv2 (also CAT protocol) | `SunnyHaze/IML-ViT` `[verified]` | **Yes** (GDrive/Baidu), Colab demo `[verified]` | **MIT** `[verified]` | ~4–6 GB @1024 | **~12 GB/sample** (A40 48 GB fits bs=4) `[verified]` |
| **FOCAL** | 2023→2025 TCSVT | Soft **pixel-level contrastive** training + **on-the-fly unsupervised clustering** at test time; no fixed threshold; models concatenable without retraining | RGB (backbone-agnostic: ViT/HRNet) | Encoder + contrastive head + HDBSCAN-style clustering | Multiple public sets | `HighwayWu/FOCAL` `[search]` | Likely `[search]` | Unclear `[search]` | ~3 GB | ~16 GB |
| **Mesorch** | 2025 AAAI | **Mesoscopic** view: parallel CNN + Transformer, macro semantics ∥ micro artifacts, multi-scale orchestration | RGB (+DCT variant) | ConvNeXt ∥ SegFormer, adaptive scale weighting | IMDL-BenCo protocols | `scu-zjz/Mesorch` `[search]` | **Yes** (`mesorch-98.pth`, `mesorch_p-118.pth`) `[search]` | Repo under scu-zjz org, CC-BY-4.0 family `[search]` | ~2–3 GB @512 | ~10–14 GB |
| **SparseViT** | 2025 AAAI | Drops handcrafted extractors: **sparse self-attention** learns non-semantic features self-supervisedly; parameter-efficient | RGB only | Sparse-coding ViT | IMDL-BenCo protocols | `scu-zjz/SparseViT` `[search]` | **Yes** (Google Drive) `[search]` | scu-zjz org `[search]` | ~2–3 GB | ~10 GB |
| **SAFIRE** | 2025 AAAI | **Point prompting** (SAM-style): each point segments the source region containing it → binary localization *and* multi-source partitioning | RGB | SAM-derived encoder + prompt decoder | FantasticReality, CASIA, IMD2020, tampCOCO (CAT settings, minus compRAISE) | `mjkwon2021/SAFIRE` `[verified]` | **Yes** (`safire.pth`) `[verified]` | **Apache-2.0 code, CC-BY-4.0 weights → commercial OK** `[verified]` | ~4–6 GB | trained on 6 GPUs, bs 2–6/device `[verified]` |
| **Omni-IML** | ICLR 2026 (v1 arXiv 2411.14823) | One generalist model over natural + document + face IML; modal-gate encoder, dynamic-weight decoder, anomaly enhancement; **Omni-273k** with NL artifact descriptions | RGB + selectable modality (DCT etc.) | Gated multi-modal encoder–decoder | Omni-273k + MIML etc. | `qcf-568/OmniIML` `[search]` | Check repo | Check repo | ~4 GB | ≥24 GB |
| **UnionFormer** | 2024 CVPR | Joint image-level detection + pixel localization with unified transformer over object/region consistency | RGB + noise | Transformer, tri-task | Public sets | Not confirmed `[memory]` | Unknown | Unknown | — | — |
| **WSCL** | 2023 ICCV | **Weakly supervised** (image-level labels only) via multi-source consistency learning | RGB | CNN + consistency heads | Image-label-only data | `[memory]` | Unknown | Unknown | ~2 GB | ~8 GB |
| **DiffForensics** | 2024 CVPR | Diffusion model as a **self-supervised denoising pretext** for forensic features, then localization head | RGB | Diffusion encoder–decoder | Public sets | Not confirmed `[memory]` | Unknown | Unknown | ~4 GB | ≥16 GB |
| **IID-Net** | 2021 TCSVT | NAS-designed inpainting-detection net with enhancement/extraction/decision blocks — the classic **inpainting-specific** baseline | RGB + high-pass | NAS CNN | Classic inpainting (GAN/patch-match) | `[memory]` | `[memory]` | Unknown | ~1 GB | ~6 GB |
| **InpDiffusion** | 2025 AAAI | Inpainting localization framed as **conditional diffusion** mask generation; edge-aware | RGB + edge | Conditional DDPM | Inpaint32K (32k images) `[search]` | `[search]` | Unknown | Unknown | ~4 GB | ≥16 GB |
| **End4** | 2025 arXiv 2509.13214 | End-to-end denoising diffusion for **diffusion-based inpainting** detection | RGB, reconstruction error | Diffusion | Diffusion inpainting sets | `[search]` | Unknown | Unknown | — | — |
| **DinoLizer** | 2025-11 arXiv 2511.20722 (rev. 2026-07) | **DINOv2 (B-Free-pretrained) + LoRA on q,v + linear head** at 14×14 patch resolution, sliding window; regenerated-outside-mask = separate class | RGB only | ViT + LoRA + linear probe | Own set + inpainting benchmarks | "upon acceptance" — **verify availability** `[verified: not yet public as of the v1 abstract]` | Not yet | Not yet stated | **~2–3 GB** | **~4–6 GB (LoRA)** |
| **EfficientIML** | 2025-09 arXiv 2509.08583 | Lightweight 3-stage **EfficientRWKV** (state-space ∥ attention) + multi-scale supervision for **high-res** IML; introduces SIF dataset (1.2k diffusion manipulations) | RGB | RWKV hybrid | SIF + public | Not in abstract `[verified]` | Unknown | Unknown | **~1–2 GB** | ~8 GB |
| **ForMa** | 2025 arXiv 2502.09941 | Vision-**Mamba** backbone, linear complexity; reported 37M params / 42G FLOPs `[search]` | RGB | VMamba encoder + light decoder | Public sets | `[search]` | Unknown | Unknown | ~1–2 GB | ~8 GB |
| **RITA** | CVPR 2026 Findings (arXiv 2509.20006) | Argues one-shot mask prediction causes **dimension collapse**; reformulates as **autoregressive layer-by-layer** mask sequence prediction; adds HSIM dataset + HSS metric | RGB | Autoregressive decoder | HSIM | `[search]` | Unknown | Unknown | ≥6 GB | ≥24 GB |
| **FakeShield** | ICLR 2025 (arXiv 2410.02761) | MLLM-based **explainable** IFDL: domain-tag-guided detection module + multimodal localization module; MMTD-Set-34k built with GPT-4o | RGB + text | 22B MLLM + segmentation head | MMTD-Set-34k (+SD_Inpaint) | `zhipeixu/FakeShield` `[verified]` | **Yes**, HF `zhipeixu/fakeshield-v1-22b` `[verified]` | **Apache-2.0** `[verified]` | ≥40 GB fp16 | ≫ |
| **SIDA** | CVPR 2025 (arXiv 2412.04292) | LLaVA+LISA-based detect / localize / **explain** on social-media images; **SID-Set** 300K | RGB + text | 7B & 13B LMM | SID-Set | `hzlsaber/SIDA` `[search]` | **Yes**, HF `saberzl/SIDA-7B` / `-13B` `[search]` | Inherits **LLaMA-2 Community License** `[search]` | ~16 GB (7B fp16) | ≫ |
| **ForgeryGPT** | 2024-10 arXiv 2410.10238 | Mask-aware LLM with forensic feature extractor for interactive explanation | RGB + text | LLM + mask encoder | IFDL sets | `[search]` | Unknown | Unknown | ≥24 GB | ≫ |
| **LEGION** | ICCV 2025 | Ground **and explain** synthetic-image artifacts (artifact localization for AI-generated content, not just tampering) | RGB + text | MLLM + grounding | Own artifact set | `[search]` | Unknown | Unknown | ≥24 GB | ≫ |
| **IFDL-VLM** | 2026-03 arXiv 2603.12930 | Finds VLM priors "hardly benefit" IFDL (bias toward semantic plausibility); uses **location masks as guidance** instead; SOTA on 9 benchmarks | RGB + mask guidance | VLM + mask conditioning | 9 benchmarks | `sha0fengGuo/IFDL-VLM` `[verified]` | Unknown | Unknown | ≥16 GB | ≫ |

Other 2025–2026 items worth tracking (all `[search]`, seen in the `greatzh/Papers` index `[verified]`): **ForensicsSAM** (adversarially robust unified IFDL), **MUN** (AAAI'25), **CLUE** (LoRA to capture latent uncovered evidence), **Detective SAM** (adaptive AI-image forgery localization), **PromptForge-350k** (prompt-based AI forgery localization dataset+framework), **SARIF** (Segment Anything for robust forensics), **DEAL-300K** (mask-free diffusion editing localization, frozen VFM + multi-frequency prompt tuning) `[verified]`, **TGIF2** benchmark `[search]`, **Webly-supervised IML via category-aware auto-annotation** (arXiv 2508.20987) `[search]`.

**Pattern across 2025–2026:** (a) frozen foundation backbone (DINOv2/SAM/VFM) + small adapter or prompt-tuning beats bespoke forensic CNNs on generative edits; (b) handcrafted noise streams (SRM/DCT) are being dropped or learned instead (SparseViT); (c) datasets, not architectures, are the bottleneck.

---

## 4. IMDL-BenCo (and its successor ForensicHub)

`[verified]` — https://github.com/scu-zjz/IMDLBenCo, paper arXiv 2406.10580, **NeurIPS 2024 Datasets & Benchmarks Spotlight**, license **CC-BY-4.0**, install `pip install imdlbenco`, docs at scu-zjz.github.io/IMDLBenCo-doc (EN+ZH).

**Provides** `[verified from repo + paper abstract]`: a decomposed modular pipeline (dataset → transform → model → trainer → evaluator) so a new model is one registered class; training code for **8 SOTA IMDL models** (zoo also covers Mesorch, SparseViT, IML-ViT, a DINOv3-IML entry); **2 standard protocols**, **15 GPU-accelerated metrics**, **3 robustness evaluations**; checkpoints released Mar 2025 (Baidu NetDisk — expect slow/blocked downloads); and a dataset index with counts, tamper types and *working mirrors* for datasets whose original links are dead.

**Caveat** `[verified]`: BenCo's "CAT-Net protocol" is a *balanced* variant, **not** the original CAT-Net setting (some real-image datasets omitted), so its reproduced numbers are not directly comparable to the CAT-Net paper.

**ForensicHub** `[verified]` — `scu-zjz/ForensicHub`, arXiv 2505.11003, **NeurIPS 2025**, CC-BY-4.0. Same group, config-driven, four domains (Deepfake, IMDL, AIGC, Document IML), cross-domain "FIDL leaderboard", checkpoints via OneDrive/Baidu.

**Right for this project?** Yes as an *evaluation harness* — protocol parity, metrics and robustness suite for free, which is exactly the credibility a portfolio project needs. **No** as a runtime dependency: it is a research trainer (multi-GPU `torchrun`, Linux-shaped scripts, Baidu weights). Keep `imgforensics` inference-only; put BenCo behind a `benchmarks/` extra. ForensicHub is tempting because it also covers your AIGC component (1), but it is newer — start with BenCo, watch ForensicHub. Complement with **PhotoHolmes** (arXiv 2412.14969, Apache-2.0 base, Python ≥3.10) `[verified]`: ten methods behind one interface plus a single-image CLI — ideal for a "run 5 detectors on one image" demo. **License gotcha:** it warns that invoking a restrictive method (e.g. TruFor) subjects your whole use to that method's license `[verified]`.

---

## 5. Datasets

Sizes marked *est.* are order-of-magnitude guesses; **verify before downloading on a limited disk**. `n/v` = not verified.

| Dataset | # images (auth / tampered) | Manipulations | Masks? | Format | Size | Access | License | Notes |
|---|---|---|---|---|---|---|---|---|
| **CASIA v1.0** | 800 / 921 `[search]` | splice, copy-move | Yes (derived) | JPEG | ~0.3 GB *est.* | Original `forensics.idealtest.org` flaky; mirrors in BenCo index `[verified]` | Research | Standard *test* set. Use "CASIAv1plus" (MVSS) or corrected GT. |
| **CASIA v2.0** | 7,491 / 5,123 `[search]` | splice, copy-move, removal, post-proc | Yes (noisy) | TIFF/JPEG/BMP | ~1.5 GB *est.* | Mirrors | Research | ⚠ **Noisy GT** → use `SunnyHaze/CASIA2.0-Corrected-Groundtruth` `[search]`. ⚠ **Format shortcut**: ~60% of tampered are TIFF while authentic are JPEG/BMP; metadata-only features reportedly reach ~0.92 AUC `[search, arXiv 2607.06615]`. |
| **Columbia (color)** | 183 / 180 `[search]` | splicing only, uncompressed | Yes | TIFF | ~1 GB *est.* | Columbia page; often mirrored (original link intermittently dead) | Research | Trivially easy; near-saturated. Keep only as a sanity check. |
| **Coverage** | 100 pairs `[search]` | copy-move (with similar-but-genuine distractors) | Yes | TIFF | ~0.1 GB *est.* | `github.com/wenbihan/coverage` `[search]` | Research | Tiny; designed to defeat naive similarity detectors. |
| **NIST16 / MFC** | ~564–611 tampered (source-dependent) `[search]` | splice, copy-move, removal, enhancement | Yes | JPEG | ~2–5 GB *est.* | **Registration/form required** at NIST `[search]` | NIST terms | MFC18/19/20 are much larger and also gated. Count discrepancies across papers — state your subset. |
| **IMD2020** | 35k synthetic + **2,010 real-world** `[search]` | splice, copy-move, removal | Yes | mixed | ~10 GB *est.* | `staff.utia.cas.cz/novozada/db/` `[search]` | Research | The 2,010 *real-world* forgeries are the valuable part. |
| **DEFACTO** | ~190k–200k `[search]` | splice, copy-move, removal, face morph | Yes | JPEG | tens of GB | Kaggle `[search]` | Research | MVSS uses a DEFACTO-84k train / DEFACTO-12k test split `[verified from MVSS repo]`. Auto-generated → unrealistic composites. |
| **DSO-1 / Carvalho** | 100 / 100 `[memory]` | person splicing, illumination-inconsistent | Yes | PNG | ~1 GB *est.* | Author page `[memory]` | Research | Good illumination-cue test set. |
| **Korus (realistic tampering)** | 220 spliced `[search]` | insertion, removal, hand-crafted in GIMP/Affinity | Yes | TIFF (RAW-derived) | ~10 GB *est.* | Google Drive, access request `[search]` | Research | Highest-realism classic set; hard. |
| **tampCOCO** | ~800,000 `[verified BenCo index]` | copy-move + splice variants (cm/sp/bcm/bcmc) | Yes | JPEG | tens of GB | Kaggle + Baidu (CAT-Net repo) `[verified]` | Follows MS-COCO licensing `[verified]` | CAT-Net training set. Subsample heavily. |
| **compRAISE (JPEG RAISE)** | n/v | irregular-shape forgeries on RAISE | Yes | JPEG | large | Kaggle `[verified]` | RAISE terms `[verified]` | CAT-Net training set; SAFIRE deliberately **excluded** it `[verified]`. |
| **FantasticReality** | 12,000 / 1,000 `[search]` | splicing | Yes | — | n/v | Author contact `[verified BenCo index]` | Research | Part of CAT/SAFIRE protocols. |
| **HiFi-IFDL** | 13 forgery methods, hierarchical labels `[search]` | full-synth + partial manip, hierarchical taxonomy | Yes (high-res) | — | large | `CHELSEA234/HiFi_IFDL` `[search]` | Research | Bridges components (1) and (2) of your system. |
| **CocoGlide** | 512 tampered `[search]` | **GLIDE diffusion inpainting** on COCO-val 256×256 crops | Yes | PNG | ~0.1 GB *est.* | TruFor repo `[verified]`; also HF `nebula/CocoGlide` `[search]` | Derived from COCO | ⭐ Small, free, the standard "does your classic model survive diffusion?" probe. |
| **AutoSplice** | 3,621 `[search]` | **DALL·E-2** text-prompt local edits | Yes | JPEG (multi-QF) | ~2 GB *est.* | Public repo `[search]` | Research | Ships several JPEG qualities — useful for robustness curves. |
| **TGIF** | 3,124 authentic → ~75k fake `[verified]` | SD2, SDXL, Adobe Firefly inpainting; composited **and** fully-regenerated variants | Yes (segmentation + bbox) | PNG | large (multi-GB) | `IDLabMedia/tgif-dataset` + Zenodo `[verified]` | **CC BY-SA 4.0** (COCO-derived) `[verified]` | ⭐ Splits: 2,440 / 341 / 343 authentic `[verified]`. |
| **TGIF2** | +196k fakes (271,788 total) `[verified]` | adds FLUX.1 schnell/dev/fill-dev + **random non-semantic masks** | Yes | PNG | very large | Same repo `[verified]` | CC BY-SA 4.0 | ⭐ 2026 extension; the random-mask split kills semantic shortcut learning. |
| **GIM** | 1,140k pairs `[search]` | generative manipulation, SAM-derived masks, +JPEG/blur/downsample degradations | Yes | JPG img / PNG mask | very large | `chenyirui/GIM` `[search]`, AAAI 2025 | Research | Only if you have disk. |
| **MIML / MIMLv2** | 123,150 / 246,212 `[search]` | **manually** forged (web-collected), auto pixel-annotated (CAAA + QES) | Yes | — | large | `qcf-568/MIML`, **email application required** `[search]` | Research, gated | ⭐ Closest to "real Photoshop work" at scale. Gated → plan ahead. |
| **DEAL-300K** | 300k+ `[verified]` | **mask-free instruction editing** (MLLM instructions + diffusion editor), active-learning change-detection masks | Yes | — | very large | `ymhzyj/DEAL-300K` `[verified]` | n/v | ⭐ The only large set targeting instruction-based full-image editors. |
| **SID-Set** | 300k `[search]` | full-synth + tampered social-media images, with text annotations | Yes | — | large | HF (SIDA repo) `[search]` | n/v | For the VLM branch. |
| **MMTD-Set-34k** | 34k triplets | image–mask–description, GPT-4o enriched | Yes | — | medium | HF `[verified]` | Apache-2.0 repo `[verified]` | FakeShield training data. |
| **Inpaint32K** | 32,000 `[search]` | inpainting | Yes | — | n/v | InpDiffusion `[search]` | n/v | — |
| **SIF** | 1,200+ `[verified]` | high-res diffusion splicing + inpainting, semantic masks | Yes | — | small | EfficientIML `[verified]` | n/v | Small but high-resolution. |
| **OpenForensics** | 115,325 `[verified BenCo index]` | face manipulation (multi-face, in-the-wild) | Yes | — | large | Public | Research | **Faces only** — out of scope here; mentioned for completeness. |

**Cross-cutting dataset problems to state in your README:** (1) **label noise** — CASIA v2 masks are wrong often enough to matter, corrected GT exists `[search]`; (2) **format/compression shortcuts** — CASIA v2's TIFF-vs-JPEG confound `[search]`, plus synthetic sets that save the two classes through different pipelines, so re-encode *both* classes identically before training; (3) **content leakage** — tampCOCO, CocoGlide, TGIF, DEAL-300K and MIML all derive from COCO or the web, so split **by source image** as TGIF does `[verified]`; (4) **saturation** — Columbia and Coverage are effectively solved and inflate averages; (5) **test-set-as-training-set** — CASIA v1 shares source content with v2, so training on v2 and calling v1 "cross-dataset" is wrong.

---

## 6. Evaluation protocols and robustness

| Protocol | Train on | Test on | Metrics | Notes |
|---|---|---|---|---|
| **MVSS protocol** | CASIAv2 (or DEFACTO-84k) | CASIAv1(+), Columbia, COVERAGE, NIST16, DEFACTO-12k | pixel F1 @ **fixed 0.5**, pixel AUC, **image-level AUC + F1**, and a combined score | `[verified from repo]`. The most honest of the three because it forces mixed authentic/tampered evaluation and a fixed threshold. |
| **CAT-Net protocol** | 5 sets: CASIAv2, FantasticReality, IMD2020, tampCOCO, compRAISE | held-out splits + cross-dataset | pixel-level F1/AUC | `[verified from repo]`. Heavy (≈1M training images). SAFIRE follows it minus compRAISE `[verified]`. |
| **IMDL-BenCo unified** | 2 built-in protocols, incl. a **balanced** CAT-Net-like variant | standard suite | 15 GPU-accelerated pixel+image metrics | `[verified]`. ⚠ Not numerically comparable to original CAT-Net results. |

**Robustness axes** (IMDL-BenCo ships 3 categories `[verified]`; GIM bakes 3 degradations into the data `[search]`): **JPEG re-compression** (QF 100→50 — the most important axis; DCT-stream models like CAT-Net shift behaviour sharply and platforms always re-encode); **resize** 0.25×–1.0× (patch-level ViTs degrade as artifact scale shifts relative to the patch grid — sliding-window at native resolution helps, per DinoLizer `[verified]`); **noise/blur**; and **social-media round-trip** (upload to WhatsApp/Telegram/X, re-download). The last is the realistic test almost nobody reports — 200 images × 3 platforms is a cheap, genuinely novel portfolio contribution.

Report every method under the *same* preprocessing: the most common silent bug in this literature is one model seeing native resolution and another a 512² resize.

---

## 7. Recommendations for this project

### (a) Inference-first path — runnable today on 6 GB, license-clean

| Rank | Model | Why | License | Expected behaviour on **AI inpainting** |
|---|---|---|---|---|
| 1 | **CAT-Net v2** | Apache-2.0 + CC-BY-4.0 weights `[verified]`, strong on JPEG-history manipulations, HRNet fits 6 GB at ~1024 px if you tile | ✅ commercial | Weak-to-moderate. If the edit was pasted into an already-JPEG image it fires; on a fully re-encoded diffusion output the DCT stream loses its cue. |
| 2 | **MVSS-Net(++)** | Cheap (ResNet-50 @512, ~2 GB), has a real **detection head** → few false alarms on authentic images `[verified]` | ⚠ license not stated in repo — **email the authors before shipping** | Poor on diffusion inpainting (SRM noise mismatch is absent), decent on splicing/copy-move. |
| 3 | **SAFIRE** | Apache-2.0 + CC-BY-4.0 `[verified]`, SAM-quality boundaries, and **multi-source partitioning** is a great demo feature | ✅ commercial | Unknown-to-moderate; the point-prompt formulation is source-based, so a generated region *is* a distinct "source". Worth measuring on CocoGlide. |
| 4 | **IML-ViT** | MIT `[verified]`, weights + Colab demo, RGB-only so no preprocessing plumbing | ✅ commercial | Moderate; edge supervision picks up mask boundaries even without noise cues. |
| — | **TruFor** | Best-known classic baseline, gives a confidence map | ❌ **non-commercial** `[verified]` | Use **only** for internal benchmark comparison in a clearly-marked optional extra; never as a default dependency. |
| — | **FakeShield / SIDA / LEGION** | explanations | ❌ 22B / 13B — impossible on 6 GB | Out of scope for local inference. |

Ship a `LocalizerEnsemble` that runs 2–4 permissive models, resamples heatmaps to a common grid and fuses by mean with per-model calibrated thresholds. FOCAL's finding that feature-level concatenation boosts performance without retraining `[search]` is the justification; heatmap averaging is the practical version.

### (b) Trainable option that fits Colab free tier or 6 GB

**Recommended: DinoLizer-style DINOv2 + LoRA patch localizer** `[verified as a published recipe, arXiv 2511.20722]`. Frozen **DINOv2 ViT-B/14** (~86M, ~0.3 GB fp16); LoRA rank 8–16 on **q,v** only (~1–2M trainable); one linear head → per-patch logit at 14×14 stride; 518² windows with **sliding-window** inference at native resolution and averaged overlapping logits. Memory *est.* **~4–6 GB at batch 2 with AMP + gradient checkpointing** → fits the 2060; batch 8 on a Colab T4. Why not a U-Net from scratch: at your data scale a frozen self-supervised backbone is the only route to generalization, and DinoLizer's large IoU gain over specialist models is the evidence. **Label detail:** for fully-regenerated images treat regenerated-outside-mask pixels as a third class or mask them out of the loss `[verified]`, or you teach the model to call clean VAE output "authentic".

**Fallback if DinoLizer code never appears:** fine-tune **SegFormer-B0/B2** (~3.7M/25M params) on 512² crops — ~3–4 GB at batch 4. Essentially the Mesorch transformer branch alone; defensible and reproducible.

**Dataset subset plan (target ≤ 60 GB on disk):**

| Purpose | Source | Take | Size *est.* |
|---|---|---|---|
| Classic manipulation train | CASIA v2 (**corrected GT**) | all | ~1.5 GB |
| Classic manipulation train | tampCOCO | 30k random (source-disjoint) | ~5 GB |
| AI-inpainting train | TGIF, one generator per epoch group (SD2 + SDXL) | 25k fakes + all 2,440 authentic | ~15 GB |
| AI-inpainting train | TGIF2 FLUX + **random-mask** split | 10k | ~8 GB |
| In-house | SD/FLUX inpainting on COCO (see (c)) | 5k | ~3 GB |
| Test — classic | CASIA v1+, Columbia, COVERAGE, IMD2020-realworld | all | ~4 GB |
| Test — generative | **CocoGlide** (512), AutoSplice, TGIF test split, DEAL-300K sample | all + 2k | ~5 GB |
| Robustness | derived on the fly (JPEG QF, resize, blur) | — | 0 |

Skip GIM, MIML, full DEAL-300K, HiFi-IFDL and NIST/MFC for v1 — gated access and/or disk cost outweigh the marginal value.

### (c) Building a small in-house AI-inpainting set — **worth it, with a caveat**

Recipe: COCO-val images → COCO instance masks (free, exact, zero annotation work) → dilate 5–15 px → SD1.5 / SDXL / FLUX-fill inpainting with the instance class name as prompt (plus an empty-prompt variant = object removal) → save the composited *and* fully-regenerated versions plus the mask.

- **Cost:** ~5k images at 512² on a 2060 is roughly 4–8 s/image for SD1.5 → about 8 h unattended; SDXL/FLUX need Colab. Disk ~3 GB.
- **Value: high** — (i) a *generator you control* for held-out-generator experiments, (ii) a controllable **mask-size sweep**, which is exactly where synthetic-image detectors fail `[verified, arXiv 2512.16688]`, (iii) your own contribution rather than a redistributed dataset.
- **Caveat:** keep it *supplementary*. TGIF/TGIF2 already cover SD2/SDXL/Firefly/FLUX at 271k images with source-disjoint splits and a permissive licence `[verified]` — do not redo that. Use yours for ablations and demos, never to claim generalization (train+test on your own generator is the classic self-deception here). Release masks + scripts + prompts, not an image dump.

### (d) Evaluation protocol for `imgforensics`

1. **Primary**: MVSS protocol (train CASIAv2-corrected, test CASIAv1+/Columbia/COVERAGE/IMD2020) — pixel F1@0.5, pixel IoU, pixel AUC, **image-level AUC and F1 on a mixed authentic+tampered set**.
2. **Generative track**: CocoGlide + AutoSplice + TGIF test split + a held-out-generator split (train on SD2/SDXL, test on FLUX). Report per-generator and **per-mask-area-bin** (small / medium / large) — the small bin is where everything fails.
3. **Threshold discipline**: report F1@0.5 *and* best-F1 *and* AP, in three separate columns, for every method including baselines. Say explicitly which convention each cited paper used.
4. **Robustness**: JPEG QF ∈ {95, 85, 75, 60, 50}, resize ∈ {1.0, 0.75, 0.5, 0.25}, Gaussian noise σ ∈ {0, 2, 5}. Plot F1 vs. degradation.
5. **Harness**: IMDL-BenCo for protocol parity and metrics `[verified]`; keep `imgforensics` itself inference-only.
6. Always include a **trivial baseline** (all-ones mask; random mask of the same area) in tables — it is shocking how often published pixel-AUC is close to it.

### (e) What to avoid

- ❌ Shipping **TruFor** (or PhotoHolmes-with-TruFor) in a repo you label permissively — non-commercial `[verified]`.
- ❌ Using **MVSS-Net weights** commercially without confirming terms — no explicit license surfaced in the repo `[verified]`.
- ❌ Training **IML-ViT** locally: ~12 GB per sample at 1024² (A40 48 GB fits bs=4) `[verified]`. Inference only.
- ❌ Any **7B–22B MLLM** (FakeShield, SIDA, ForgeryGPT, LEGION) on a 2060.
- ❌ Reporting **per-image optimal-threshold F1** as your headline number.
- ❌ Training on raw **CASIA v2** without corrected GT and without re-encoding both classes identically — you will learn the TIFF/JPEG shortcut `[search]`.
- ❌ Random train/test splits over COCO-derived sets — split by source image.
- ❌ Claiming a classic IMDL model "detects AI inpainting" without a CocoGlide/TGIF number.
- ❌ Downloading full tampCOCO (800k) `[verified]`, GIM (1.14M) or DEAL-300K (300k) on limited disk.
- ❌ Assuming Baidu NetDisk checkpoint links (IMDL-BenCo, Mesorch, MVSS, CAT-Net mirrors) will work — always prefer the Google Drive / HuggingFace mirrors, and vendor a copy.

---

## 8. Open questions before implementation

1. Are **DinoLizer** code/weights public yet? v1 said "upon acceptance"; a revision is dated 2026-07 — re-check `[verified]`.
2. **MVSS-Net license** — email the authors.
3. **SAFIRE on CocoGlide** — no published number found; measure it. If source-partitioning transfers to diffusion regions, SAFIRE becomes pick #1.
4. **MIML access** — email application; start now if wanted for v2 `[search]`.

## 9. arXiv IDs

IMDL-BenCo 2406.10580 · ForensicHub 2505.11003 · IML-ViT 2307.14863 · TruFor 2212.10957 · HiFi-Net 2303.17111 · FOCAL 2308.09307 · SAFIRE 2412.08197 · SparseViT 2412.14598 · Omni-IML 2411.14823 · GIM 2406.16531 · TGIF 2407.11566 · TGIF2 2603.28613 · DinoLizer 2511.20722 · EfficientIML 2509.08583 · ForMa 2502.09941 · RITA 2509.20006 · DEAL-300K 2511.23377 · End4 2509.13214 · InpDiffusion 2501.02816 · localized-deepfake study 2512.16688 · FakeShield 2410.02761 · SIDA 2412.04292 · ForgeryGPT 2410.10238 · IFDL-VLM 2603.12930 · PhotoHolmes 2412.14969
