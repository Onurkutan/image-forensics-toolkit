# Benchmark report: COCO

- Entries evaluated: 1000
- Robustness levels: clean
- Fixed threshold: 0.5

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| dinov2_head | - | only one label present at this level; AUC undefined | | | | | | | |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| dinov2_head | clean | - | only one label present; AUC undefined |

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
| dinov2_head | 1000 | 35.49 |
