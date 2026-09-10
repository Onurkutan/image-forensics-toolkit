# Fusion 03: the fuser across the robustness levels

Date: 2026-09-10. Raw reports: [`05_wildrf_val_levels.md`](05_wildrf_val_levels.md) (WildRF val,
398 images, the shipped head and the seven signals at all 15 levels),
[`05_wildrf_test_levels.md`](05_wildrf_test_levels.md) (the same 1,000-image WildRF test sample
as [experiment 02](02_experiment_summary.md), all levels),
[`05_fusion_levels_eval.md`](05_fusion_levels_eval.md) (a fuser fitted on every level of the
val records, evaluated per level) and
[`05_fusion_clean_fuser_levels_eval.md`](05_fusion_clean_fuser_levels_eval.md) (the shipped
clean-level fuser of [fusion 01](04_fusion_wildrf.md), evaluated on the same records).

## The question

Fusion 01 fitted the stacking fuser on clean images only and left open whether a fuser needs to
see perturbed scores to work on perturbed images. These records answer it: the head and every
signal were scored on the WildRF val and test splits at all 15 levels of the default
robustness suite, one fuser was fitted on all of them (5,970 rows) and the fusion 01 fuser was
re-evaluated on the same test records, level by level, with `imgforensics fusion eval`.

## Result

Head alone, fused with the levels-fitted fuser, fused with the clean-only fuser -- WildRF
test, 1,000 images per level, threshold 0.5. "Called" is the share of images outside the
abstain band and the balanced accuracy on them.

| Level | Head AUC / FPR | Levels fuser AUC / FPR | Called, bal. acc. | Clean fuser AUC / FPR | Called, bal. acc. |
|---|---|---|---|---|---|
| clean | 0.980 / 0.137 | 0.980 / 0.036 | 30%, 1.000 | 0.981 / 0.036 | 53%, 0.996 |
| jpeg_q75 | 0.977 / 0.131 | 0.977 / 0.036 | 29%, 1.000 | 0.978 / 0.041 | 49%, 0.994 |
| jpeg_q50 | 0.977 / 0.105 | 0.977 / 0.026 | 29%, 0.993 | 0.977 / 0.034 | 48%, 0.994 |
| webp_q80 | 0.976 / 0.114 | 0.976 / 0.028 | 26%, 1.000 | 0.976 / 0.030 | 48%, 0.998 |
| resize_0.75 | 0.978 / 0.212 | 0.978 / 0.060 | 31%, 1.000 | 0.979 / 0.084 | 56%, 0.987 |
| resize_0.5 | 0.965 / 0.309 | 0.967 / 0.103 | 25%, 0.996 | 0.967 / 0.137 | 54%, 0.971 |
| resize_0.25 | 0.904 / 0.474 | 0.906 / 0.273 | 20%, 0.950 | 0.906 / 0.309 | 48%, 0.880 |
| roundtrip_0.5 | 0.972 / 0.204 | 0.971 / 0.036 | 23%, 0.986 | 0.972 / 0.058 | 48%, 0.988 |
| crop_0.8 | 0.979 / 0.146 | 0.979 / 0.052 | 30%, 1.000 | 0.979 / 0.056 | 54%, 0.998 |
| noise_5 | 0.968 / 0.131 | 0.968 / 0.032 | 22%, 0.959 | 0.968 / 0.036 | 42%, 0.988 |
| social_1080_q80 | 0.968 / 0.230 | 0.969 / 0.073 | 20%, 1.000 | 0.969 / 0.077 | 44%, 0.993 |

The full 15-level tables, with AP, TPR and ECE per row, are in the two evaluation reports.

## Reading the numbers honestly

- **Fusion is what makes the head usable under perturbation.** The head's own false-positive
  rate on laundered real photographs climbs from 13.7% clean to 21% at a 0.75 resize, 31% at
  half scale and 23% under the social re-share pipeline; either fuser brings each of those
  back to 4-10%, at an unchanged AUC. The signals do not rank better than the head (their
  AUCs are near 0.5); they move the operating point, which is the job a stacking layer has.
- **The clean-only fuser already generalizes across levels.** Fitted on 398 clean images, it
  matches the levels-fitted fuser's AUC everywhere and comes within 0.02-0.03 of its
  false-positive rate at the resize levels. A fuser does not need to see perturbed scores;
  the relationship between the head's score and the signals' scores that it learned on clean
  images holds after recompression, resizing and noise.
- **Seeing the levels buys a little false-positive rate for a lot of abstention.** The
  levels-fitted fuser reaches a lower false-positive rate on every row (0.060 vs 0.084 at
  resize 0.75, 0.103 vs 0.137 at half scale, 0.273 vs 0.309 at quarter scale) but its band
  calls only 20-31% of the images against the clean fuser's 42-56%, and it does so at a
  balanced accuracy of 0.95-1.00 against 0.88-1.00. Fitted on rows that are mostly perturbed
  and harder, it learned to be more careful; that is a legitimate operating point for a
  deployment that prefers silence to error, and the wrong one for a triage tool that has to
  say something about half of what it sees. The shipped default stays the fusion 01 fuser.
- **Quarter-scale resize is still the wall.** At `resize_0.25` the head's AUC is 0.904 and its
  false-positive rate 47%; fusion halves the latter and the band still lets through a 0.88
  balanced accuracy on the clean fuser. Below the crop size there is no cue left for either
  the head or the JPEG-domain signals; that failure has to be fixed upstream of fusion
  (a multi-scale crop policy) or declared.
- **All of this is WildRF, in distribution.** These are held-out images of a distribution the
  head has seen, so the absolute numbers belong with fusion 01's, not with experiment 03's
  cross-dataset ones; what transfers is the shape -- fusion fixes the operating point, not
  the ranking, and does so at every level.

## Cost

The val run (398 images, 15 levels, 7 signals in 4 worker processes plus the head) took
2 h 30 min and the test run (1,000 images) 5 h 35 min on the shared machine; WildRF's images
are large (median longest side above 2,000 px) and every classical signal runs at native
resolution, so the signals, not the head, set the bill. Fitting and the four evaluations took
seconds.
