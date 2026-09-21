# Benchmark report: TGIF

- Entries evaluated: 2000
- Robustness levels: clean
- Fixed threshold: 0.5
- Threshold caveat: threshold tuned in-sample

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| iml_vit | fixed | 0.531 | 0.901 | 0.311 | 0.524 | 0.203 | 0.251 | 0.546 | 0.449 |
| iml_vit | tuned | 0.531 | 0.901 | 0.425 | 0.547 | 0.297 | 0.390 | 0.546 | 0.449 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| iml_vit | clean | 0.531 | 0.524 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| iml_vit | none | only one label present; AUC undefined |
| iml_vit | ps-sp | only one label present; AUC undefined |
| iml_vit | sd2-fr | only one label present; AUC undefined |
| iml_vit | sd2-sp | only one label present; AUC undefined |
| iml_vit | sdxl-fr | only one label present; AUC undefined |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| iml_vit | TGIF | 0.531 |

## Pixel metrics

| detector | level | n | mean f1@0.5 | mean best-f1 | mean ap | mean iou |
|---|---|---|---|---|---|---|
| iml_vit | clean | 1778 | 0.062 | 0.273 | 0.241 | 0.046 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| iml_vit | 2000 | 363.92 |
