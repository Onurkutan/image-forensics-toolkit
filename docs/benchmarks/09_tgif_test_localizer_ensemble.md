# Benchmark report: TGIF

- Entries evaluated: 2000
- Robustness levels: clean
- Fixed threshold: 0.5
- Threshold caveat: threshold tuned in-sample

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| localizer_ensemble | fixed | 0.678 | 0.950 | 0.648 | 0.623 | 0.410 | 0.656 | 0.249 | 0.230 |
| localizer_ensemble | tuned | 0.678 | 0.950 | 0.517 | 0.679 | 0.113 | 0.471 | 0.249 | 0.230 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| localizer_ensemble | clean | 0.678 | 0.623 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| localizer_ensemble | none | only one label present; AUC undefined |
| localizer_ensemble | ps-sp | only one label present; AUC undefined |
| localizer_ensemble | sd2-fr | only one label present; AUC undefined |
| localizer_ensemble | sd2-sp | only one label present; AUC undefined |
| localizer_ensemble | sdxl-fr | only one label present; AUC undefined |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| localizer_ensemble | TGIF | 0.678 |

## Pixel metrics

| detector | level | n | mean f1@0.5 | mean best-f1 | mean ap | mean iou |
|---|---|---|---|---|---|---|
| localizer_ensemble | clean | 1778 | 0.439 | 0.580 | 0.571 | 0.388 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| localizer_ensemble | 2000 | 690.90 |
