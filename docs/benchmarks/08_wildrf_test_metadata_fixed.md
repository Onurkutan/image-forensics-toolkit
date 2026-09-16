# Benchmark report: WildRF

- Entries evaluated: 1000
- Robustness levels: clean
- Fixed threshold: 0.5

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| metadata | fixed | 0.503 | 0.537 | 0.469 | 0.503 | 0.000 | 0.006 | 0.033 | 0.250 |
| metadata | tuned | 0.503 | 0.537 | 0.469 | 0.503 | 0.000 | 0.006 | 0.033 | 0.250 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| metadata | clean | 0.503 | 0.503 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| metadata | none | 0.503 |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| metadata | WildRF | 0.503 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| metadata | 1000 | 8.10 |
