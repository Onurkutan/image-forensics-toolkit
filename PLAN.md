# Plan

One-screen board for the next session. The full phase plan and every result note live in
[docs/ROADMAP.md](docs/ROADMAP.md); the benchmark tables in [docs/benchmarks/](docs/benchmarks/);
the workbench design in [docs/design/01_toolbox_architecture.md](docs/design/01_toolbox_architecture.md).

## Status

- Now: **Phase 4b, stage 1 shipped** (commits `720b47b`, `e382f92`). `dino_inpaint` (frozen DINOv2
  patch tokens, a 1.2 M-parameter patch head, 448 px crops) trained on TGIF's regenerated subsets
  reaches pixel best-F1 0.580 / 0.565 on the full sd2-fr / sdxl-fr test split (CAT-Net 0.340 / 0.200)
  and 0.637 on CocoGlide, a generator it never saw (CAT-Net 0.605); the three-member `max`
  ensemble is the best pooled map (0.708) but stacks false positives (61% of authentic images above
  0.5 somewhere). Everything is in `docs/benchmarks/10_dino_inpaint_summary.md`; checkpoint
  `weights/dino_inpaint_01` (research-only, TGIF is CC BY-SA).
- Next step: **6e slice 2, progressive JPEG coefficients.** 80% of ITW-SM files are progressive and
  `catnet_v2` currently re-encodes them at quality 100 before reading the DCT stream; libjpeg reads
  progressive coefficients, so the `jpeglib` path can take them, which changes results and needs a
  re-benchmark of `catnet_v2` on ITW-SM-style inputs. The 4b calibration question is settled (see
  Decisions): no per-checkpoint bias; verdicts through the fusion layer.
- Queued after that: a 4b stage-1b run with hard negatives (more varied authentic images than
  TGIF's 4,140 crops, weighted towards the categories that fire, e.g. `donut`) against the object
  prior the paired check measured; optional stage-2 LoRA and the ablations (block 11 only, hard
  targets, context kernel 1); the localization robustness suite on TGIF; 6c publish (HF Space and
  head weights, one-line heads-up first); the per-year decay chart.
- Blockers: none. (TGIF's manual download is done; TGIF2 FLUX/random are not downloaded.)
- Last update: 2026-09-25

## Milestones

Validation for every milestone: `pytest` (non-ml) and `pytest -m ml` green locally and in CI; a
benchmark milestone also reproduces its table from saved records with `imgforensics fusion eval`.

### M1 Classical signals (roadmap Phase 1) -- done 2026-09-09
### M2 Data and evaluation harness (Phase 2) -- done 2026-09-09
### M3 Whole-image AI-generated detector (Phase 3)
- [x] experiments 01-03, Synthbuster, ITW-SM (cross-dataset exit criterion met 2026-09-15)
- [x] experiment 04 probe: simulated laundering does not explain the real-class false positives (2026-09-21)
- [ ] per-year decay chart
- [ ] a genuine in-the-wild real-image source in training (the next attempt on the real-class FPR),
      disjoint from ITW-SM and WildRF-test; refit the fusers on the fixed `metadata` signal then
### M4 Manipulation localization (Phase 4)
- [x] 4a: `iml_vit`, `catnet_v2`, `localizer_ensemble`, CocoGlide tables
- [x] TGIF: download, layout with mask pairing, manifest, first localizer benchmarks (2026-09-21)
- [x] 4b stage 1: `dino_inpaint`, own DINOv2 inpainting localizer trained on TGIF's regenerated
      subsets; passes the pre-registered bars on TGIF and CocoGlide (2026-09-23)
- [x] 4b follow-ups measured: full test split, three-member ensemble, threshold study (2026-09-25)
- [x] 4b checks: paired object test (a synthesis detector with an object prior, weaker than
      CAT-Net's on CocoGlide) and threshold study (0.5 is near the pixel optimum) (2026-09-25)
- [ ] 4b stage 1b: hard negatives against the object prior; optional stage-2 LoRA and ablations
- [ ] full TGIF test split and localization robustness suite for the record
### M5 Fusion and explanation (Phase 5) -- done 2026-09-10
### M6 Product and release (Phase 6)
- [x] 6a service layer, 6b FastAPI, 6c Gradio demo code, 6d web workbench client
- [x] 6e slice 1: `jpeglib` fast path for the coefficient reader (`catnet_v2` 462 to 163 ms per CocoGlide image)
- [ ] 6e slice 2: progressive JPEG coefficients (libjpeg reads them; the numpy decoder cannot)
- [ ] 6e slice 3: ONNX export of the head for CPU inference
- [ ] 6c publish: Hugging Face Space and the head weights on the Hub under a research-only card
      (needs Onur's Hub login; one-line heads-up before publishing)
- [ ] documentation site or extended README with the benchmark report; v1.0 tag

## Decisions

- 2026-09-25 -- No calibration bias is stored in the `dino_inpaint` checkpoint, and image-level
  verdicts go through the fusion layer rather than the maps' top-1% score. Why: the validation
  threshold sweep puts the pixel optimum at 0.25 with only +0.03 F1 over 0.5, while lowering it
  lights more authentic images; the real-image false positives are a tail of confident mistakes
  driven by an object prior (paired check: authentic maps align with the inpainted object at
  3.8-4.9x chance on TGIF, 1.5x on CocoGlide, where CAT-Net's prior is 1.9x), which is a training
  problem, not a threshold one. Alternatives considered: a per-checkpoint logit bias (small gain,
  more false positives); both bias and fusion (adds a knob nothing needs).
- 2026-09-23 -- 4b stage 1 (head only, frozen backbone) is accepted as the localizer; stage 2 (LoRA)
  is optional, not a prerequisite. Why: the pre-registered bars were all cleared, and the validation
  curve shows the frozen feature space saturating in one epoch, so LoRA is the priced way past that
  ceiling rather than a fix. Training on the regenerated subsets only was deliberate: the spliced
  subsets' edge cue is what CAT-Net already reads, and mixing it in would have hidden the number
  that mattered. Alternatives considered: all four subsets (a general inpainting localizer, worse
  readability), SegFormer-B2 (the roadmap's fallback, not needed).
- 2026-09-21 -- Experiment 04 (train on simulated social-media laundering) not run. Why: laundering
  COCO reals moves the exp03 head's FPR only from 2.6% to at most 8.9%, and on ITW-SM both heads
  call the *least* compressed reals fake most often; the failure is content/source, not
  re-encoding. Alternatives considered: run it anyway (six GPU hours for a bounded, partial answer).
- 2026-09-21 -- TGIF ground truth: spliced images pair with the bordered `ps_mask` their pipeline
  used, regenerated ones with the crop-size mask; all three `orig` variants are reals, with
  `extra.variant`. Duplicate reals (identical full/1024 variants, photos filed under several COCO
  categories) are real and no split-leak; training on TGIF must dedupe by sha256.
- 2026-09-21 -- `jpeglib` (MPL-2.0, cp38-abi3 wheels) as the optional fast coefficient reader in the
  `ml` extra; the numpy decoder stays as the fallback and the guarantee is byte-identity on
  well-formed baseline streams (corrupt entropy data can part the two; documented and pinned).
  Alternatives: `jpegio` (no Windows wheel, no CPython > 3.10), optimizing the numpy decoder (x2-3 at
  best against x33).
- 2026-09-16 -- Relicensed from MIT to PolyForm Noncommercial 1.0.0. Why: the project is
  non-commercial and commercial use by others is not wanted. Alternatives considered: CC BY-NC-SA 4.0
  (not written for software), a custom text (avoid). Versions up to commit `0735322` stay MIT for
  anyone who obtained them; vendored IML-ViT/dwtDct (MIT) and CAT-Net (Apache-2.0) keep their notices.
- 2026-09-16 -- A bare Photoshop APP13 segment is no longer an editor marker in `metadata`; Meta's
  `FBMD` fingerprint is a platform marker that never scores. Fusers were not refitted: their weight on
  `metadata` was -0.06 / -0.005 and the cold-applied tables moved by at most 0.001
  (`docs/benchmarks/08_itwsm_summary.md`, follow-up).
- 2026-09-10 -- Web workbench over a PySide6 desktop client; no build toolchain, static client
  inside `imgforensics.api`.
- 2026-09-10 -- Shipped defaults are the experiment 02 head and the fusion 01 fuser. The
  experiment 03 head (no WildRF in training) is the better curated-output detector (Synthbuster 0.980
  vs 0.969) and the worse social-media one (ITW-SM 0.819 vs 0.888): two operating points, one trade-off.
- Working rules -- crop, never resize; honest evaluation (unseen sets, FPR reported, baselines in every
  table); nothing third-party committed; at most 2-3 commits a day, batched by milestone; README changes
  are read by Onur before push.
- Tried and dropped -- parameter-free localizer ensembles (none beats CAT-Net alone on CocoGlide or
  TGIF; `max` is the default); `--workers` for feature extraction (no gain, the main-process loop
  around the backbone is the bottleneck).

## Open questions

- DinoLizer code and weights availability (re-check before 4b).
- Whether the TGIF2 FLUX subsets (196k images, a separate download) are worth the disk as an
  unseen-generator test for 4b.
- Chameleon dataset access needs an academic request; the plan does not depend on it.
