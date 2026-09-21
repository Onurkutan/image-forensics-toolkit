# Benchmark report: TGIF

- Entries evaluated: 2000
- Robustness levels: clean
- Fixed threshold: 0.5
- Threshold caveat: threshold tuned in-sample

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| catnet_v2 | fixed | 0.669 | 0.952 | 0.593 | 0.686 | 0.194 | 0.566 | 0.356 | 0.342 |
| catnet_v2 | tuned | 0.669 | 0.952 | 0.513 | 0.697 | 0.068 | 0.461 | 0.356 | 0.342 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| catnet_v2 | clean | 0.669 | 0.686 |

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
| catnet_v2 | TGIF | 0.669 |

## Pixel metrics

| detector | level | n | mean f1@0.5 | mean best-f1 | mean ap | mean iou |
|---|---|---|---|---|---|---|
| catnet_v2 | clean | 1778 | 0.458 | 0.592 | 0.599 | 0.416 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| catnet_v2 | 2000 | 366.55 |
