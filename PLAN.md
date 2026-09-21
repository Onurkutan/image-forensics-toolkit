# Plan

One-screen board for the next session. The full phase plan and every result note live in
[docs/ROADMAP.md](docs/ROADMAP.md); the benchmark tables in [docs/benchmarks/](docs/benchmarks/);
the workbench design in [docs/design/01_toolbox_architecture.md](docs/design/01_toolbox_architecture.md).

## Status

- Now: Phase 6 shipped through 6d; 6e's first slice (a `jpeglib` fast path for the JPEG coefficient
  reader, decoder time down ~33x) is in; TGIF is on disk with a layout, a manifest and the first
  localizer benchmarks (`docs/benchmarks/09_tgif_localizers_summary.md`). Experiment 04 (laundering
  simulation) was probed and dropped: the head's real-class failure is content, not compression
  (`docs/benchmarks/09_exp04_laundering_probe_summary.md`).
- Next step: **Phase 4b, the own inpainting localizer** -- a DINOv2 patch-level localizer (LoRA or a
  light head) trained on TGIF's training split, aimed at the fully regenerated images where the
  released localizers score at the predict-everything baseline; evaluate on TGIF test (per subset),
  CocoGlide and the localization robustness suite, with `catnet_v2` as the baseline in every table.
  First sub-steps: dedupe the reals by sha256 for training, decide the patch-label rule from the
  masks, and check DinoLizer's code availability once more.
- Queued after that: 6e slice 2, progressive-JPEG coefficients through libjpeg (80% of ITW-SM files
  are progressive and `catnet_v2` currently re-encodes them, so this changes results and needs a
  re-benchmark); the full 9,261-image TGIF test split and its robustness suite; 6c publish (HF Space
  and head weights, one-line heads-up first); the per-year decay chart.
- Blockers: none. (TGIF's manual download is done; TGIF2 FLUX/random are not downloaded.)
- Last update: 2026-09-21

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
- [ ] 4b: own DINOv2 inpainting localizer trained on TGIF (fully regenerated images are the target)
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
