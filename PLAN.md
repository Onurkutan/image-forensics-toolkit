# Plan

One-screen board for the next session. The full phase plan and every result note live in
[docs/ROADMAP.md](docs/ROADMAP.md); the benchmark tables in [docs/benchmarks/](docs/benchmarks/);
the workbench design in [docs/design/01_toolbox_architecture.md](docs/design/01_toolbox_architecture.md).

## Status

- Now: Phase 6 shipped through 6d (workbench at `imgforensics serve`). Latest work: the
  `metadata` signal's APP13 fix (commit `0735322`) and the relicense to PolyForm Noncommercial 1.0.0.
- Next step: **exp04, the laundering simulation** -- put a social re-share pipeline over the COCO
  reals in training to attack the real-class false positives (ITW-SM FPR 0.442, WildRF cross-dataset
  0.547); refit the fusers on the fixed `metadata` signal in the same run.
- Blockers: TGIF manual download for Phase 4b (no scriptable URL; steps in
  `src/imgforensics/data/acquire.yaml`), waiting on Onur.
- Last update: 2026-09-21

## Milestones

Validation for every milestone: `pytest` (non-ml) and `pytest -m ml` green locally and in CI; a
benchmark milestone also reproduces its table from saved records with `imgforensics fusion eval`.

### M1 Classical signals (roadmap Phase 1) -- done 2026-09-09
### M2 Data and evaluation harness (Phase 2) -- done 2026-09-09
### M3 Whole-image AI-generated detector (Phase 3)
- [x] experiments 01-03, Synthbuster, ITW-SM (cross-dataset exit criterion met 2026-09-15)
- [ ] per-year decay chart
- [ ] exp04 laundering simulation: design, run, refit fusers, record in `docs/benchmarks/09_*`
### M4 Manipulation localization (Phase 4)
- [x] 4a: `iml_vit`, `catnet_v2`, `localizer_ensemble`, CocoGlide tables
- [ ] 4b: own DINOv2 + LoRA inpainting localizer (needs TGIF; DinoLizer code availability to re-check)
### M5 Fusion and explanation (Phase 5) -- done 2026-09-10
### M6 Product and release (Phase 6)
- [x] 6a service layer, 6b FastAPI, 6c Gradio demo code, 6d web workbench client
- [ ] 6c publish: Hugging Face Space and the head weights on the Hub under a research-only card
      (needs Onur's Hub login; one-line heads-up before publishing)
- [ ] 6e CPU inference: ONNX export of the head, faster JPEG coefficient decoder
- [ ] documentation site or extended README with the benchmark report; v1.0 tag

## Decisions

- 2026-09-16 -- Relicensed from MIT to PolyForm Noncommercial 1.0.0. Why: the project is
  non-commercial and commercial use by others is not wanted. Alternatives considered: CC BY-NC-SA 4.0
  (not written for software), a custom text (avoid). Versions up to commit `0735322` stay MIT for
  anyone who obtained them; vendored IML-ViT/dwtDct (MIT) and CAT-Net (Apache-2.0) keep their notices.
- 2026-09-16 -- A bare Photoshop APP13 segment is no longer an editor marker in `metadata`; Meta's
  `FBMD` fingerprint is a platform marker that never scores. Fusers were not refitted: their weight on
  `metadata` was -0.06 / -0.005 and the cold-applied tables moved by at most 0.001
  (`docs/benchmarks/08_itwsm_summary.md`, follow-up). The refit rides with exp04.
- 2026-09-10 -- Web workbench over a PySide6 desktop client; no build toolchain, static client
  inside `imgforensics.api`.
- 2026-09-10 -- Shipped defaults are the experiment 02 head and the fusion 01 fuser. The
  experiment 03 head (no WildRF in training) is the better curated-output detector (Synthbuster 0.980
  vs 0.969) and the worse social-media one (ITW-SM 0.819 vs 0.888): two operating points, one trade-off.
- Working rules -- crop, never resize; honest evaluation (unseen sets, FPR reported, baselines in every
  table); nothing third-party committed; at most 2-3 commits a day, batched by milestone; README changes
  are read by Onur before push.
- Tried and dropped -- parameter-free localizer ensembles (none beats CAT-Net alone; `max` matches it
  and is the default); `--workers` for feature extraction (no gain, the main-process loop around the
  backbone is the bottleneck).

## Open questions

- DinoLizer code and weights availability (re-check before 4b).
- Chameleon dataset access needs an academic request; the plan does not depend on it.
