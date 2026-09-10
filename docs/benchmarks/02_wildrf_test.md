# Benchmark report: WildRF

- Entries evaluated: 1000
- Robustness levels: clean
- Fixed threshold: 0.5

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| dinov2_head | fixed | 0.980 | 0.983 | 0.920 | 0.916 | 0.137 | 0.970 | 0.070 | 0.064 |
| dinov2_head | tuned | 0.980 | 0.983 | 0.936 | 0.936 | 0.060 | 0.933 | 0.070 | 0.064 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| dinov2_head | clean | 0.980 | 0.916 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| dinov2_head | none | 0.980 |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| dinov2_head | WildRF | 0.980 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| dinov2_head | 1000 | 107.57 |
