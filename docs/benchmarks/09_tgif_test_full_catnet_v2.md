# Benchmark report: TGIF

- Entries evaluated: 9261
- Robustness levels: clean
- Fixed threshold: 0.5
- Threshold caveat: threshold tuned in-sample

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| catnet_v2 | fixed | 0.660 | 0.950 | 0.584 | 0.660 | 0.242 | 0.563 | 0.358 | 0.346 |
| catnet_v2 | tuned | 0.660 | 0.950 | 0.496 | 0.683 | 0.076 | 0.443 | 0.358 | 0.346 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| catnet_v2 | clean | 0.660 | 0.660 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| catnet_v2 | none | only one label present; AUC undefined |
| catnet_v2 | ps-sp | only one label present; AUC undefined |
| catnet_v2 | sd2-fr | only one label present; AUC undefined |
| catnet_v2 | sd2-sp | only one label present; AUC undefined |
| catnet_v2 | sdxl-fr | only one label present; AUC undefined |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| catnet_v2 | TGIF | 0.660 |

## Pixel metrics

| detector | level | n | mean f1@0.5 | mean best-f1 | mean ap | mean iou |
|---|---|---|---|---|---|---|
| catnet_v2 | clean | 8232 | 0.457 | 0.594 | 0.601 | 0.414 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| catnet_v2 | 9261 | 432.66 |
