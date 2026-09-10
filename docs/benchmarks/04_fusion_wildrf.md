# Fusion 01: stacking the DINOv2 head with the classical signals on WildRF

Date: 2026-09-10. Fitted with `imgforensics fusion fit` on the WildRF val split (398 images,
199 fake / 199 real, clean level only): every classical signal plus the experiment 02 head
(`dinov2_head`, [`02_experiment_summary.md`](02_experiment_summary.md)). Evaluated on the same
1,000-image sample of the WildRF test split used in experiment 02, with the signals recorded
in [`02_wildrf_test_signals.md`](02_wildrf_test_signals.md).

## Result

Reproduce this table from the saved records with
`imgforensics fusion eval data/benchmarks/02_wildrf_test_signals.json data/benchmarks/02_wildrf_test.json --fuser weights/fuser_wildrf.json`;
the command's output is in [`04_fusion_wildrf_eval.md`](04_fusion_wildrf_eval.md).


| Scorer | AUC | Balanced acc. at 0.5 | FPR at 0.5 | TPR at 0.5 |
|---|---|---|---|---|
| `dinov2_head` alone | 0.980 | 0.916 | 0.137 | 0.970 |
| Fused, all images | 0.981 | 0.929 | 0.036 | 0.895 |
| Fused, outside the abstain band (534 of 1,000) | n/a | 0.996 | 0.004 | 0.997 |

The fuser leaves the ranking alone (AUC 0.981 vs 0.980) and moves the operating point: the
false-positive rate on laundered social-media photographs drops from 13.7% to 3.6% at the
cost of recall (97.0% to 89.5%). With the abstain band applied, the 53% of images it is
willing to call are called almost perfectly; the other 47% are reported as `uncertain`.

## What the fuser learned

Logit weights, fitted on 318 training images with L2 regularization and class balancing:

| Detector | Weight | Reading |
|---|---|---|
| `dinov2_head` | +0.86 | the main evidence |
| `jpeg_ghost` | -0.76 | inverted: on WildRF a strong ghost response is a *real* photograph's compression history |
| `ela` | +0.36 | recompression error rises on the fakes here |
| `copy_move` | -0.31 | duplicated-block matches occur more on the reals (repetitive textures) |
| `double_jpeg` | +0.07 | weak |
| `metadata`, `sd_watermark`, `c2pa` | about 0 | abstain on almost every laundered image |

These signs are properties of WildRF, not of the signals: on Community Forensics the same
`ela` runs inverted. A fuser is a dataset-specific calibration layer and must be refitted
whenever the deployment distribution changes; the artifact records the sha256 of the
records it was fitted on for that reason.

## Calibration and band

Held-out ECE 0.048 before and after (temperature 1.0: scaling did not improve it, so the
identity was kept). The band search with a 0.9 balanced-accuracy target chose
[0.010, 0.990], the widest band that still met the target on the held-out split, where it
reached 1.000 outside the band with a 41% abstain rate; on the test sample the abstain rate
is 47%. A lower target gives a narrower band and fewer abstentions.

## Caveats

- 398 fitting images is small; the weights above are indicative, not final.
- Fit and test come from the same dataset (different splits), so this is an
  in-distribution calibration, like the head's own WildRF number.
- The fuser was fitted on the clean level only. A fuser fitted across robustness levels
  would need the signals recorded at those levels too.
