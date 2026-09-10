# Benchmark report: WildRF

- Entries evaluated: 1000
- Robustness levels: clean
- Fixed threshold: 0.5

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| dinov2_head | fixed | 0.804 | 0.829 | 0.688 | 0.673 | 0.547 | 0.893 | 0.216 | 0.238 |
| dinov2_head | tuned | 0.804 | 0.829 | 0.735 | 0.732 | 0.311 | 0.775 | 0.216 | 0.238 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| dinov2_head | clean | 0.804 | 0.673 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| dinov2_head | none | 0.804 |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| dinov2_head | WildRF | 0.804 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| dinov2_head | 1000 | 135.73 |
