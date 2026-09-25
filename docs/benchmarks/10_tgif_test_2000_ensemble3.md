# Benchmark report: TGIF

- Entries evaluated: 2000
- Robustness levels: clean
- Fixed threshold: 0.5
- Threshold caveat: threshold tuned in-sample

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| localizer_ensemble | fixed | 0.867 | 0.979 | 0.901 | 0.679 | 0.608 | 0.965 | 0.031 | 0.075 |
| localizer_ensemble | tuned | 0.867 | 0.979 | 0.760 | 0.798 | 0.153 | 0.750 | 0.031 | 0.075 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| localizer_ensemble | clean | 0.867 | 0.679 |

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
| localizer_ensemble | TGIF | 0.867 |

## Pixel metrics

| detector | level | n | mean f1@0.5 | mean best-f1 | mean ap | mean iou |
|---|---|---|---|---|---|---|
| localizer_ensemble | clean | 1778 | 0.602 | 0.708 | 0.722 | 0.495 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| localizer_ensemble | 2000 | 863.39 |
