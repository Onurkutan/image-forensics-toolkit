# Benchmark report: CocoGlide

- Entries evaluated: 1024
- Robustness levels: clean
- Fixed threshold: 0.5
- Threshold caveat: threshold tuned in-sample

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| dinov2_head | fixed | 0.644 | 0.677 | 0.565 | 0.565 | 0.023 | 0.154 | 0.417 | 0.418 |
| dinov2_head | tuned | 0.644 | 0.677 | 0.620 | 0.620 | 0.211 | 0.451 | 0.417 | 0.418 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| dinov2_head | clean | 0.644 | 0.565 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| dinov2_head | glide | only one label present; AUC undefined |
| dinov2_head | none | only one label present; AUC undefined |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| dinov2_head | CocoGlide | 0.644 |

## Pixel metrics

| detector | n | mean f1@0.5 | mean best-f1 | mean ap | mean iou |
|---|---|---|---|---|---|
| dinov2_head | 512 | 0.078 | 0.363 | 0.278 | 0.059 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| dinov2_head | 1024 | 20.72 |
