# Benchmark report: WildRF

- Entries evaluated: 398
- Robustness levels: clean
- Fixed threshold: 0.5

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| dinov2_head | fixed | 0.750 | 0.705 | 0.668 | 0.668 | 0.563 | 0.899 | 0.237 | 0.269 |
| dinov2_head | tuned | 0.750 | 0.705 | 0.683 | 0.683 | 0.503 | 0.869 | 0.237 | 0.269 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| dinov2_head | clean | 0.750 | 0.668 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| dinov2_head | none | 0.750 |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| dinov2_head | WildRF | 0.750 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| dinov2_head | 398 | 169.58 |
