# Benchmark report: TGIF

- Entries evaluated: 2000
- Robustness levels: clean
- Fixed threshold: 0.5
- Threshold caveat: threshold tuned in-sample

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| dino_inpaint | fixed | 0.776 | 0.965 | 0.714 | 0.691 | 0.338 | 0.720 | 0.245 | 0.206 |
| dino_inpaint | tuned | 0.776 | 0.965 | 0.652 | 0.702 | 0.234 | 0.638 | 0.245 | 0.206 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| dino_inpaint | clean | 0.776 | 0.691 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| dino_inpaint | none | only one label present; AUC undefined |
| dino_inpaint | ps-sp | only one label present; AUC undefined |
| dino_inpaint | sd2-fr | only one label present; AUC undefined |
| dino_inpaint | sd2-sp | only one label present; AUC undefined |
| dino_inpaint | sdxl-fr | only one label present; AUC undefined |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| dino_inpaint | TGIF | 0.776 |

## Pixel metrics

| detector | level | n | mean f1@0.5 | mean best-f1 | mean ap | mean iou |
|---|---|---|---|---|---|---|
| dino_inpaint | clean | 1778 | 0.339 | 0.521 | 0.530 | 0.244 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| dino_inpaint | 2000 | 184.85 |
