# 01 — Whole-Image AI-Generated Image Detection: Literature Survey

**Scope:** module (1) of `imgforensics` — binary "is this whole photo AI-generated?" Localization of local edits and classic manipulation are covered by separate surveys.
**Compiled:** 2026-09-09. **Target hardware:** Windows 11, Python 3.10, RTX 2060 6 GB; Colab/Kaggle for training.

**Verification legend:** `[verified]` = page opened this session, claim taken from it. `[search]` = link appeared in search results but was not opened. `[from memory]` = background knowledge, not re-checked — a lead to confirm, not a fact. Every number is quoted from a named source; where a source gave none, the cell says "not stated".

---

## 1. TL;DR

1. **Cross-generator generalization is the whole problem.** A Feb-2026 benchmark of 23 detector variants (16 methods) over 12 datasets / 2.6 M images / 291 generators: "a 37 percentage-point gap separates the best detector (75.0% mean accuracy) from the worst (37.5%)," no universal winner [verified, arXiv:2602.07814].
2. **Detectors decay with generator age.** Same study: accuracy drops "from approximately 79% for 2020–2021 generators to around 38% for 2024 models"; Flux Dev, Firefly v4 and Midjourney v7 get "only 18–30% average detection accuracy" [verified].
3. **The best off-the-shelf zero-shot detector is the data-centric one.** Community Forensics: "75.0% mean accuracy with 82.1% median", but only "35–42%" on the newest commercial generators [verified].
4. **Training data beats architecture.** Within detector families, "training data explains 20–60% performance variance" despite identical architectures, "often exceeding variance between different architectures" [verified].
5. **Backbone choice is settled enough to copy:** DINOv2-L/14 beat CLIP-L/14 as a frozen feature space in the wild, 94.90 vs 91.92 average AUC [verified, arXiv:2507.10236].
6. **Never resize. Crop.** Resizing "erase[s] the subtle high-frequency traces left by the generation process"; texture cropping lifted DMID 78.78 → 89.46 AUC over center cropping [verified].
7. **Augmentation is not optional:** DMID went 78.31 → 89.46 AUC with JPEG/WEBP + geometric + noise augmentation [verified].
8. **Dataset bias is the #1 way to fool yourself.** GenImage-trained detectors "actually learn from existing Biases in compression and image size" — real=JPEG vs fake=PNG, plus resolution bias [verified, arXiv:2403.17608].
9. **Thresholds, not features, are often the bug.** Many detector–dataset pairs show "high AUC (>0.8) but low accuracy (<0.5) ... poorly calibrated thresholds" [verified]. Report AUC *and* thresholded accuracy.
10. **Baseline for this repo:** frozen DINOv2/CLIP features + light trainable head, Community-Forensics-Small subset, crop-not-resize, JPEG augmentation. Under a day on a 6 GB card. Stretch: SPAI or B-Free.

---

## 2. Problem framing

### 2.1 Generator families

| Family | Examples | Forensic signature | Status 2026 |
|---|---|---|---|
| GAN | ProGAN, StyleGAN1–3, BigGAN, StarGAN | Strong periodic upsampling grid artifacts; checkerboard spectra | Essentially solved; ~79% mean accuracy for 2020–21 generators [verified, 2602.07814] |
| Latent diffusion (LDM) | SD 1.4/1.5/2.1/XL/3.5, Midjourney, Wukong | VAE decoder fingerprint; weaker but present spectral peaks | Detectable with training, degrades on unseen samplers |
| Pixel-space diffusion | ADM, GLIDE, Imagen | Different noise imprint than LDM; no VAE trace | Reconstruction-error methods that assume an autoencoder fail here |
| Autoregressive / unified MM | GPT-4o image gen, GPT-Image-2, Qwen-Image, Nano-Banana (Gemini) | Token-by-token generation, no upsampling stack — most low-level cues vanish | The hard case. `[search]` GPT-Image-2-Bench, Nano-banana subsets appear in 2026 benchmarks |
| Flow-matching / rectified flow | FLUX.1 dev/pro, SD3.5 | Similar to LDM but retrained VAE | "18–30% average detection accuracy" for Flux Dev [verified] |

**Why cross-generator generalization is the open problem.** Every detector is a classifier of *generator-specific low-level statistics*. When the generator changes its upsampler, VAE, or sampler — or drops the convolution stack entirely (autoregressive) — the learned statistic disappears. Rankings are also unstable across evaluation sets: "Spearman rank correlations between dataset pairs range from 0.01 to 0.87" [verified, 2602.07814], so a GenImage leaderboard win predicts almost nothing about Chameleon.

**Laundering.** Platforms recompress, resize and strip metadata. Cited inside that benchmark: detectors "achieving above 95% accuracy on academic benchmarks drop to below 60% on social media deepfakes due to compression artifacts, resolution variability, and adversarial perturbations" [verified]. Directly measured: Synthbuster AUC 96.98 → 79.96 in the wild [verified, arXiv:2507.10236]. NTIRE 2026 made this the whole task: 294,500 images (108,750 real + 185,750 generated), 42 generators, **36 transformations**, scored by ROC AUC on transformed and untransformed test images; 511 registrants, 20 valid submissions [verified, arXiv:2604.11487].

---

## 3. Method families

VRAM figures are **my estimates for a 6 GB RTX 2060** unless a source is cited; treat them as planning numbers.

### 3.1 CNN artifact classifiers

| Method | Year / venue | Core idea | Backbone | Training data | Reported generalization | Code | Weights | License | VRAM (infer / train) |
|---|---|---|---|---|---|---|---|---|---|
| CNNDetection (Wang et al.) | 2020 CVPR | A ResNet50 trained on one GAN + blur/JPEG aug generalizes to many CNN generators | ResNet50 | ProGAN on 20 LSUN categories; test set = 13 CNN synthesis algorithms [verified] | Historic baseline; near-random on modern diffusion (see §5) | [PeterWang512/CNNDetection](https://github.com/PeterWang512/CNNDetection) [verified] | Yes — `weights/download_weights.sh`, variants blur_jpg_prob0.1 / 0.5 [verified] | LICENSE.txt, type not stated in README [verified] | ~1 GB / ~4 GB |
| DMID (Corvi et al.) | 2023 ICASSP | Re-train the CNNDetection recipe with diffusion-aware augmentation + frequency analysis | ResNet50 | LDM + COCO/UCID/ImageNet reals [search] | Baseline in ITW-SM: 89.46 AUC avg with texture crop + aug [verified, 2507.10236] | [grip-unina/DMimageDetection](https://github.com/grip-unina/DMimageDetection) [search] | Not confirmed | Not confirmed | ~1 GB / ~4 GB |
| AIDE | 2025 ICLR | Mixture of experts: DCT/SRM low-level scoring + OpenCLIP semantic branch | ResNet50 + ConvNeXt (OpenCLIP) [verified] | ProGAN/GenImage variants | Introduced Chameleon, on which "almost all models classify AI-generated images as real ones" [verified, repo README] | [shilinyan99/AIDE](https://github.com/shilinyan99/AIDE) [verified] | Yes — Google Drive model zoo [verified] | MIT (code) [verified] | ~3 GB / >8 GB — likely needs Colab |

### 3.2 Frequency-domain methods

| Method | Year / venue | Core idea | Notes |
|---|---|---|---|
| Frank et al., Durall et al. | 2020 ICML / CVPR | GAN upsampling leaves periodic peaks in the DFT / azimuthal spectrum | Foundational; brittle to resizing and JPEG [from memory] |
| FreqDetect, Gram, Fusing | 2020–2022 | Spectral / texture statistics fed to a shallow classifier | All packaged with weights in SIDBench [verified] |
| **SPAI** — Any-Resolution Detection by Spectral Learning | 2025 CVPR | Learn the spectral distribution of **real** images via masked frequency modeling; deviation = fake score; any resolution | "5.5% absolute improvement in AUC over the previous state-of-the-art across 13 recent generative approaches" plus robustness to online perturbations [search]. ViT-B/16 MFM backbone, LDM fakes + COCO/LSUN reals; **"inference ... with less than 8GB of GPU RAM"**, training "originally targeted Nvidia L40S 48GB GPUs" [verified]. Code + weights [mever-team/spai](https://github.com/mever-team/spai), **Apache 2.0** [verified] |
| CoDA / "Secret Lies in Color" | 2025 CVPR + 2026 arXiv | Color-distribution probing instead of the DFT | `[search]` arXiv:2605.24306; CVPR 2025 poster exists |

### 3.3 Foundation-model feature methods

| Method | Year / venue | Core idea | Backbone | Reported generalization | Code / weights | License | VRAM |
|---|---|---|---|---|---|---|---|
| UnivFD (Ojha et al.) | 2023 CVPR | Do **not** fine-tune: nearest-neighbour / linear probe on frozen CLIP features | CLIP ViT-L/14 | The reference "frozen feature space" result [from memory]; weights shipped in SIDBench [verified] | [SIDBench HF](https://huggingface.co/dkarageo/sidbench) [verified] | Apache-2.0 (SIDBench) [verified] | ~2 GB / ~3 GB |
| FatFormer | 2024 CVPR | Forgery-Aware Adapter + language-guided alignment on CLIP | CLIP ViT-L | Not re-verified | `[search]` | unknown | ~3 GB / ~10 GB |
| **RINE** | 2024 ECCV | Use **intermediate** CLIP encoder blocks (low-level info) with a learned per-block importance head | CLIP ViT-L/14, frozen | "+10.6% absolute performance improvement" averaged over 20 test datasets; best models "requiring just a single epoch for training (approximately 8 minutes)" [search, project page]; 94.90 AUC avg in ITW-SM re-implementation [verified] | [mever-team/rine](https://github.com/mever-team/rine); weights in SIDBench [verified] | Apache-2.0 via SIDBench [verified] | ~2 GB / **~4 GB — fits the 2060** |
| C2P-CLIP | 2025 AAAI | Inject a category-common text prompt to reshape the CLIP image encoder | CLIP | "12.4% improvement in detection accuracy compared to the original CLIP" [search] | `[search]` | unknown | ~3 GB / ~8 GB |
| Effort | 2025 ICML **Oral** | SVD the ViT weights into two orthogonal subspaces; freeze principal components, adapt the residual — a forensics-specific LoRA | Any ViT (CLIP) | "significant superiority ... with very little training cost" [search] | [YZY-stack/Effort-AIGI-Detection](https://github.com/YZY-stack/Effort-AIGI-Detection) [search] | unknown | ~3 GB / ~6–8 GB |
| **B-Free** | 2025 CVPR | Remove *content/format/resolution* bias by generating fakes from the same real images via self-conditioned SD reconstruction + inpainting augmentation | ViT with 4 registers, DINOv2 pretraining, 504×504 crops [verified] | Evaluated across 27 generative models incl. FLUX and SD3.5; "more calibrated results" [search/README] | [grip-unina/B-Free](https://github.com/grip-unina/B-Free) [verified] | **Non-commercial**: "used, reproduced and modified only for informational and nonprofit purposes" [verified] | ~3 GB / >12 GB (504² crops) |
| TAP | 2026 CVPRW | Tunable attention pooling over VFM **patch** tokens instead of CLS | Best VFM "outperforms the original CLIP by more than 12% in accuracy" [verified, abstract] | arXiv:2604.26772 [verified] | not stated | ~2–3 GB |
| SSAFE | 2026 arXiv (Jun) | Frozen multimodal encoders already separate real/fake; train only a linear classifier; curate *which generators* to train on | Curated training set of **only 10,000 images** vs AIGIBench 288 K / OpenFake 4 M [verified] | Introduces RealWorldBench | arXiv:2606.08634 [verified] | no code link on abs page [verified] | **~2 GB / ~2 GB** |
| Fleet | 2026 ICML | Reject the static-feature hypothesis; continuous **few-shot** adaptation to each new generator via constrained routing correction | — | "static hypothesis suffers a severe performance drop against rapidly evolving generators" [search] | [ICTMCG/Fleet](https://github.com/ICTMCG/Fleet) [search] | unknown | — |

### 3.4 Reconstruction-error methods

| Method | Year / venue | Core idea | Practical note |
|---|---|---|---|
| DIRE | 2023 ICCV | DDIM-invert then re-generate; fakes reconstruct with lower error | Very slow (full inversion per image); weights in SIDBench [verified] |
| **AEROBLADE** | 2024 CVPR | Skip inversion — pass the image through the **LDM autoencoder**; "generated images can be more accurately reconstructed by the AE than real images". **Training-free**, "nearly matches ... detectors that rely on extensive training" [search] | [jonasricker/aeroblade](https://github.com/jonasricker/aeroblade) [search]. Cheap sanity baseline; fails by construction on non-latent (ADM/GLIDE) and autoregressive generators |
| LaRE² | 2024 CVPR | Move the reconstruction error into **latent** space for speed | `[search]` |
| SeDID / FakeInversion | 2024 | Stepwise error at intermediate diffusion timesteps / inversion into a reference text-to-image model | `[from memory]` — FakeInversion not confirmed in this session's searches |
| **DRCT** | 2024 ICML | Use diffusion reconstruction to mint **hard positives** from real images, then contrastive training — a *training augmentation*, not an inference method | DRCT-2M, 16 diffusion models; "over a 10% accuracy improvement in cross-set tests" [search]. [beibuwandeluori/DRCT](https://github.com/beibuwandeluori/DRCT) |

### 3.5 Local low-level cue methods

| Method | Year / venue | Core idea | Code / weights |
|---|---|---|---|
| LGrad | 2023 CVPR | Use gradients from a pretrained CNN as a generator-agnostic artifact representation | In SIDBench with weights [verified] |
| **NPR** | 2024 CVPR | Neighboring Pixel Relationships expose the **upsampling** operator; tiny, fast | "12.8% improvement" over prior work on 28 generative models [search]. [chuangchuangtan/NPR-DeepfakeDetection](https://github.com/chuangchuangtan/NPR-DeepfakeDetection); weights in SIDBench [verified]. **Trains in ~1 GB VRAM** — cheapest baseline. Weak on autoregressive generators (no upsampler) |
| PatchCraft | 2024 | Contrast rich- vs poor-texture patches | In SIDBench [verified] |
| DE-FAKE | 2023 CCS | Use the text-image relationship (CLIP similarity) for T2I detection | In SIDBench [verified] |
| **SAFE** | 2024–25 arXiv:2408.06741 | Replace down-sampling with **cropping** in preprocessing so artifacts are not distorted; three simple transforms + simple artifact features | Benchmarked in arXiv:2602.07814 [verified]. Repo not confirmed this session |

### 3.6 Data-centric / large-scale training

| Method | Year / venue | Core idea | Key numbers |
|---|---|---|---|
| **Community Forensics** | 2025 CVPR | Scrape **thousands** of community-fine-tuned T2I models and train one classifier on all of them | 2.7 M images from **4,803 models**; "detection performance improves as the number of models in the training set increases" and diversity helps [search/abstract]. Best zero-shot in the 2026 benchmark: 75.0% mean / 82.1% median accuracy [verified] |
| AIDE / Chameleon | 2025 ICLR | Sanity check: build a human-fooling test set | On Chameleon "all detectors suffer from significant performance drops"; most "only marginally exceed ... random guessing (50%)" [search] |
| AI-GenBench | 2025 IJCNN (Verimedia) | **Temporal** protocol: train on historically-ordered generators, test on future ones | Open data + code, [MI-BioLab/AI-GenBench](https://github.com/MI-BioLab/AI-GenBench) [verified] |
| Calibration | 2026 arXiv:2602.01973 | Post-hoc learnable logit shift on a small target-domain validation set; backbone frozen | "significantly improves robustness without retraining"; [muliyangm/AIGI-Det-Calib](https://github.com/muliyangm/AIGI-Det-Calib), CC BY 4.0 [verified] |

### 3.7 VLM/MLLM explainable detectors (brief — see the localization survey)

| Method | Venue | Note |
|---|---|---|
| FakeShield | ICLR 2025 | First MLLM framework for explainable detection **and** localization; MMTD-Set built with GPT-4o. [zhipeixu/FakeShield](https://github.com/zhipeixu/FakeShield) [search] |
| SIDA | CVPR 2025 | LLaVA+LISA; SID-Set = 300 K images; outputs mask + textual explanation. [hzlsaber/SIDA](https://github.com/hzlsaber/SIDA) [search] |

Both are 7B+ models: **not trainable on 6 GB**, inference needs heavy quantization. An aspirational explanation layer, not the classifier.

---

## 4. Datasets

| Dataset | Size | Generators | Format | Download | Access | License | Pitfalls |
|---|---|---|---|---|---|---|---|
| **ForenSynths** (CNNDetection) | Train: ProGAN over 20 LSUN classes; test: 13 CNN algorithms [verified] | GANs only | PNG [from memory] | ~70 GB train [from memory] | Open | LICENSE.txt in repo [verified] | GAN-only; obsolete as a *sole* training set |
| **GenImage** | ~1.33 M real + 1.35 M fake [search] | 8: SD1.4, SD1.5, Midjourney, ADM, GLIDE, Wukong, VQDM, BigGAN [search] | Mixed | **~500 GB** [verified] | Open (Baidu / Google Drive / Harvard Dataverse) [verified] | CC BY-NC-SA 4.0 [search] | **Severe.** Detectors "actually learn from existing Biases in compression and image size" [verified] |
| **UnbiasedGenImage** | Filtered subsets of GenImage | same 8 | JPEG-QF-aligned, 512×512 subset | subset of the 500 GB | Open [verified] | not stated [verified] | The fix: constrains natural images to 450–550 px and aligns JPEG quality across classes [verified]. **Use this, not raw GenImage** |
| DiffusionForensics (DIRE) | ~ | LDM/ADM/SD family | PNG | large | Open | `[from memory]` | Real/fake source mismatch |
| **Synthbuster** | 9 × 1,000 fakes + 1,000 real | DALL·E 2/3, Firefly, MJ v5, SD 1.3/1.4/2/XL, Glide [search] | Reals are **uncompressed** RAISE [search] | small (a few GB) | Open; RAISE reals via RAISE portal [search] | not confirmed | Uncompressed reals vs compressed fakes = format bias risk. Cheap and excellent as a *test-only* set |
| **Chameleon** | 11 K AI-generated + ~15 K real, mostly >720p to 4K [search] | Unlisted commercial | mixed | small | **Gated** — email request [verified] | Academic research only, commercial use prohibited [verified] | Hard by design; every detector collapses |
| **WildRF** | Reddit 2,150/2,150; X 340/340; Facebook 160/160 [search] | Unknown, in-the-wild | Platform JPEG | tiny | Open `[search]` | unknown | Test-only; small |
| **Community Forensics** | 2.7 M from 4,803 models, **~1.1 TB**; Systematic 1.92 M / Manual 774 K / Commercial 14.9 K / PublicEval 51.8 K [verified] | 4,803 LDM checkpoints | Parquet, PNG **or** JPEG bytes [verified] | 1.1 TB full; **"-Small" ≈ 11%** with redistributable reals [verified] | Open on HF, no gate observed [verified] | CC-BY-4.0 "for research purposes only"; per-image licenses mostly CreativeML OpenRAIL-M [verified] | Reals: LAION 40.3%, ImageNet 40.3%, CelebA, COCO, FFHQ [verified]. Mixed formats → check the real/fake format split |
| **ITW-SM** | 10,000 (5 K real / 5 K AI) from Facebook, Instagram, LinkedIn, X; 0.1 MP–8.4 K [verified] | in-the-wild | original platform compression preserved [verified] | tiny | mever-team.github.io/itw-sm [verified] | unknown | The most realistic small test set found |
| **NTIRE 2026 in-the-wild set** | 294,500 (108,750 real + 185,750 fake), 42 generators, 36 transforms [verified] | 42 open + closed | mixed/transformed | not stated | Challenge; check the CVPRW paper [verified] | unknown | Purpose-built for laundering robustness |
| ImagiNet / TWIGMA / OpenFake / AIGIBench | TWIGMA = web-scraped AI images w/ Twitter metadata [search]; AIGIBench 288 K; OpenFake 4 M [verified] | many | mixed | varies | Open `[search]` | varies | TWIGMA reals are usually paired from OpenImages — a *different* pipeline than the fakes |
| **Real sources** | COCO, LAION subsets, ImageNet, RAISE (uncompressed RAW), OpenImages, LSUN, FFHQ | — | COCO/ImageNet = JPEG; RAISE = TIFF/RAW | small–large | Open | varies | **The pitfall:** if reals are JPEG and fakes are PNG, you have built a JPEG detector, not an AI detector |

**Two pitfalls to burn into the eval code:**
1. **Format bias** — real=JPEG / fake=PNG. Fix: re-encode *both* classes to identical JPEG-quality distributions, or adopt UnbiasedGenImage's QF alignment [verified].
2. **Resolution/content bias** — classes drawn from different resolution distributions or different semantics. B-Free's answer: synthesize fakes *from* the same real images so "any differences ... stem solely from the subtle artifacts introduced by AI generation" [search].

---

## 5. Robustness findings

| Perturbation | What survives | Evidence |
|---|---|---|
| **JPEG recompression** | Only methods trained *with* JPEG augmentation. CNNDetection anticipated this in 2020 with blur+JPEG aug (blur σ 0–3, JPEG Q 30–100) [verified]. DMID: 78.31 → 89.46 AUC when JPEG/WEBP + geometric + noise augmentation is added [verified] | arXiv:2507.10236 [verified]; CNNDetection README [verified] |
| **Resizing** | Almost nothing low-level. Resizing "erase[s] the subtle high-frequency traces left by the generation process" [verified]. SAFE and SPAI both attack this by cropping / any-resolution processing rather than downscaling | arXiv:2507.10236 [verified]; SPAI [search] |
| **Cropping** | Fine, and *preferable* to resizing. Texture-based cropping > center cropping: DMID 89.46 vs 78.78 AUC [verified] | arXiv:2507.10236 [verified] |
| **Social-media re-upload** | Largest single drop observed. Synthbuster 96.98 AUC → 79.96 in-the-wild [verified]. Cited elsewhere: ">95% on academic benchmarks drop to below 60% on social media deepfakes" [verified quote inside 2602.07814] | both [verified] |
| **New generators** | Nothing generalizes well. 79% (2020–21) → 38% (2024) mean accuracy [verified]; Flux/Firefly/MJ v7 at 18–30% [verified] | arXiv:2602.07814 [verified] |
| **Threshold drift** | Ranking survives better than the decision boundary: "high AUC (>0.8) but low accuracy (<0.5)" pairs are common [verified]; a learned logit shift on a small target validation set recovers much of it [verified, arXiv:2602.01973] | both [verified] |

Family verdict: **frozen foundation features + heavy augmentation + crop-not-resize** is the most laundering-robust practical combination; **pure frequency/upsampling cues (NPR, classic DFT)** are the most fragile to resize and recompression; **reconstruction methods** tolerate content shift but are structurally blind to non-latent and autoregressive generators.

---

## 6. Commercial / closed references

For positioning only. **Hive AI** and **Sightengine** are the two commercial APIs that appear consistently in independent testing; in a NewsGuard report dated **May 8, 2026** covering 45 images (15 authentic, 15 lightly edited, 15 significantly edited, from Reuters/AP/NYT/Guardian and Google Earth), both had a **0% false-positive rate** on the 15 authentic images, while ScamAI hit 40%, ZeroGPT 20% and AI or Not 6.67%; across all five tools the report found they "collectively declared authentic images to be AI-generated 13.33 percent of the time" [verified]. **Illuminarty** is a free web detector with no published evaluation `[from memory]`. **Google SynthID Detector** is a different animal: it reads an *embedded watermark* rather than doing forensics, so it only covers Google's own Imagen/Veo/Gemini/Lyria stack, it shipped via a **waitlist prioritizing journalists and researchers rather than general availability**, and Google acknowledges it can be bypassed by extreme modification [search]. **C2PA/Content Credentials** share that weakness — absence of a signature proves nothing. Positioning for this repo: an open, offline, auditable detector reporting calibrated probabilities *and its own failure table* is genuinely differentiated, since none of the above are inspectable.

---

## 7. Recommendations for this project

### (a) Baseline — trains on 6 GB or free Colab in hours

**Frozen DINOv2 ViT-B/14 (or CLIP ViT-L/14) features + a small trainable head, RINE-style over intermediate blocks.**

- Why: DINOv2-L/14 beat CLIP-L/14 94.90 vs 91.92 avg AUC as a frozen space [verified]; RINE's best models needed "just a single epoch ... approximately 8 minutes" [search]; SSAFE showed a linear probe over a frozen encoder trained on **10,000 curated images** competes with 288 K–4 M-image pipelines [verified].
- Preprocessing (non-negotiable): **crop, never resize**; 224–336 px crops at native resolution; augment with JPEG Q30–95, WEBP, blur, downscale-then-upscale, noise, cut-out.
- Frozen backbone → **~2–4 GB VRAM** at batch 16 with AMP. Cache backbone features once; the head then trains in minutes.
- **Expected quality, honestly:** in-distribution AUC will look near-perfect and mean nothing. On held-out *modern* generators, expect roughly the 75%-mean / 82%-median accuracy band of the best public detector [verified, arXiv:2602.07814], falling toward 20–40% on 2025–26 commercial generators. Ship that number.

### (b) Stretch options

1. **SPAI** (CVPR 2025) — Apache 2.0 code *and* weights, "inference ... with less than 8GB of GPU RAM" [verified], any-resolution, robust to online perturbations. Run its released weights as an ensemble member on day one; retraining targeted 48 GB L40S [verified], so Colab A100 or not at all.
2. **Effort** (ICML 2025 Oral) — orthogonal-subspace ViT adaptation, "very little training cost" [search]; the closest thing to a LoRA that fits a 2060.
3. **B-Free-style data generation** — mint bias-free training pairs by inpainting your own real images with SD 2.1. Its license is **non-commercial only** [verified], so reimplement the *idea*; don't vendor the code into an MIT/Apache repo.

### (c) Dataset-subset plan (disk-aware)

| Purpose | Set | Approx. download |
|---|---|---|
| Train | **Community Forensics "-Small"** (~11% of 2.7 M, with redistributable reals) [verified] — take a further stratified slice of ~150–300 K images | ~40–120 GB for the Small set; slice to **~20–30 GB** on disk |
| Train (reals, bias control) | COCO train2017 | ~19 GB |
| Val (seen generators) | held-out generators from the same Community Forensics slice | 0 (reuse) |
| **Test A — controlled** | Synthbuster + RAISE-1k reals | a few GB |
| **Test B — hard** | Chameleon (email request; academic only) [verified] | small |
| **Test C — in the wild** | ITW-SM (10 K) [verified] and/or WildRF | < 5 GB |
| Optional | UnbiasedGenImage 512×512 QF-aligned subset instead of raw GenImage (never pull the full ~500 GB) [verified] | 10–30 GB |

Total working footprint: **~60–90 GB**, which is manageable; do the heavy training on Colab/Kaggle against a HF-streamed Community Forensics and keep only the eval sets locally.

### (d) Evaluation protocol

1. **Train on one family only** (Community Forensics LDM slice) → **test on strictly unseen families**: Synthbuster, Chameleon, ITW-SM. Never report a number whose test generator appeared in training.
2. **Report AUC *and* accuracy at a threshold fixed on a validation split**, plus the false-positive rate on reals separately. Accuracy and AUC agree at Pearson r=0.82 but high-AUC/low-accuracy failures are common [verified] — AUC alone hides the deployment failure.
3. **Robustness sweep as a first-class table:** clean / JPEG Q90 / Q75 / Q50 / resize 0.5× / crop / WEBP, in the NTIRE-2026 spirit (36 transformations, ROC AUC transformed and untransformed) [verified].
4. **Temporal split:** adopt AI-GenBench's historically-ordered protocol [verified] and plot per-generator-year accuracy — that 79%→38% decay curve is the most honest chart you can publish.
5. **Calibration:** report ECE, and offer the post-hoc logit shift of arXiv:2602.01973 (CC BY 4.0 code) as an opt-in step [verified].
6. **Baselines:** run SIDBench (12 methods, weights 8.74 GB on HF `dkarageo/sidbench`, **Apache-2.0**) [verified] rather than hand-rolling baseline implementations.

### (e) What to avoid

- **Raw GenImage** without QF/resolution alignment — you will learn JPEG, not AI [verified].
- **Resizing** anywhere in the preprocessing pipeline [verified].
- **Only in-distribution numbers**, or one headline accuracy over a mixed test set — rankings shift wildly across sets (Spearman 0.01–0.87) [verified].
- **Reconstruction-error methods as the primary detector** — AEROBLADE-style approaches assume a latent autoencoder and are blind to pixel-space diffusion and autoregressive generators.
- **An MLLM detector** (FakeShield/SIDA) as the 6 GB classifier; keep them for the future explanation layer.
- **Vendoring non-commercially-licensed code** (B-Free) into an open portfolio repo.
- **Promising "detects AI images"** in the README. Promise calibrated evidence plus a published failure table — a solo project should be louder about limits, not quieter.

---

## 8. Priority reading order

arXiv:2602.07814 (benchmark) → arXiv:2507.10236 (ITW-SM design choices) → arXiv:2403.17608 (dataset bias) → SPAI + B-Free repos → Community Forensics + Chameleon → arXiv:2604.11487 (NTIRE 2026) → SSAFE / TAP / Fleet for the 2026 frontier.
