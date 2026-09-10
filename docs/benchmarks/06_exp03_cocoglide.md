# Benchmark report: CocoGlide

- Entries evaluated: 1024
- Robustness levels: clean
- Fixed threshold: 0.5
- Threshold caveat: threshold tuned in-sample

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| dinov2_head | fixed | 0.627 | 0.650 | 0.553 | 0.553 | 0.043 | 0.148 | 0.429 | 0.430 |
| dinov2_head | tuned | 0.627 | 0.650 | 0.605 | 0.605 | 0.129 | 0.340 | 0.429 | 0.430 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| dinov2_head | clean | 0.627 | 0.553 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| dinov2_head | glide | only one label present; AUC undefined |
| dinov2_head | none | only one label present; AUC undefined |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| dinov2_head | CocoGlide | 0.627 |

## Pixel metrics

| detector | level | n | mean f1@0.5 | mean best-f1 | mean ap | mean iou |
|---|---|---|---|---|---|---|
| dinov2_head | clean | 512 | 0.076 | 0.362 | 0.278 | 0.058 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| dinov2_head | 1024 | 28.91 |
