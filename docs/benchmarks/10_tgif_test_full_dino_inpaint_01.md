# Benchmark report: TGIF

- Entries evaluated: 9261
- Robustness levels: clean
- Fixed threshold: 0.5
- Threshold caveat: threshold tuned in-sample

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| dino_inpaint | fixed | 0.781 | 0.965 | 0.717 | 0.700 | 0.323 | 0.722 | 0.245 | 0.205 |
| dino_inpaint | tuned | 0.781 | 0.965 | 0.656 | 0.715 | 0.208 | 0.639 | 0.245 | 0.205 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| dino_inpaint | clean | 0.781 | 0.700 |

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
| dino_inpaint | TGIF | 0.781 |

## Pixel metrics

| detector | level | n | mean f1@0.5 | mean best-f1 | mean ap | mean iou |
|---|---|---|---|---|---|---|
| dino_inpaint | clean | 8232 | 0.342 | 0.525 | 0.536 | 0.247 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| dino_inpaint | 9261 | 190.67 |
