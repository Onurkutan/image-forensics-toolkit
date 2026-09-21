# Benchmark report: COCO

- Entries evaluated: 1000
- Robustness levels: clean, jpeg_q75, resize_0.5, social_1080_q80, launder_s0.7_q80, launder_s0.5_q80
- Fixed threshold: 0.5

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| dinov2_head | - | only one label present at this level; AUC undefined | | | | | | | |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| dinov2_head | clean | - | only one label present; AUC undefined |
| dinov2_head | jpeg_q75 | - | only one label present; AUC undefined |
| dinov2_head | resize_0.5 | - | only one label present; AUC undefined |
| dinov2_head | social_1080_q80 | - | only one label present; AUC undefined |
| dinov2_head | launder_s0.7_q80 | - | only one label present; AUC undefined |
| dinov2_head | launder_s0.5_q80 | - | only one label present; AUC undefined |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| dinov2_head | none | only one label present; AUC undefined |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| dinov2_head | COCO | only one label present; AUC undefined |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| dinov2_head | 6000 | 37.93 |
