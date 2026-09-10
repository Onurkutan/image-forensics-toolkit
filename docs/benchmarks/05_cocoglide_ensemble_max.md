# Benchmark report: CocoGlide

- Entries evaluated: 1024
- Robustness levels: clean
- Fixed threshold: 0.5
- Threshold caveat: threshold tuned in-sample

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| localizer_ensemble | fixed | 0.670 | 0.676 | 0.559 | 0.559 | 0.762 | 0.879 | 0.245 | 0.296 |
| localizer_ensemble | tuned | 0.670 | 0.676 | 0.630 | 0.630 | 0.352 | 0.611 | 0.245 | 0.296 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| localizer_ensemble | clean | 0.670 | 0.559 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| localizer_ensemble | glide | only one label present; AUC undefined |
| localizer_ensemble | none | only one label present; AUC undefined |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| localizer_ensemble | CocoGlide | 0.670 |

## Pixel metrics

| detector | level | n | mean f1@0.5 | mean best-f1 | mean ap | mean iou |
|---|---|---|---|---|---|---|
| localizer_ensemble | clean | 512 | 0.385 | 0.608 | 0.565 | 0.301 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| localizer_ensemble | 1024 | 742.13 |
