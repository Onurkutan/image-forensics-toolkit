# Benchmark report: CocoGlide

- Entries evaluated: 1024
- Robustness levels: clean
- Fixed threshold: 0.5
- Threshold caveat: threshold tuned in-sample

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| localizer_ensemble | fixed | 0.642 | 0.620 | 0.610 | 0.610 | 0.332 | 0.553 | 0.068 | 0.240 |
| localizer_ensemble | tuned | 0.642 | 0.620 | 0.619 | 0.619 | 0.398 | 0.637 | 0.068 | 0.240 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| localizer_ensemble | clean | 0.642 | 0.610 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| localizer_ensemble | glide | only one label present; AUC undefined |
| localizer_ensemble | none | only one label present; AUC undefined |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| localizer_ensemble | CocoGlide | 0.642 |

## Pixel metrics

| detector | level | n | mean f1@0.5 | mean best-f1 | mean ap | mean iou |
|---|---|---|---|---|---|---|
| localizer_ensemble | clean | 512 | 0.135 | 0.613 | 0.555 | 0.096 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| localizer_ensemble | 1024 | 742.98 |
