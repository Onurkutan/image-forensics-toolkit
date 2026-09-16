# Benchmark report: ITW-SM

- Entries evaluated: 2000
- Robustness levels: clean
- Fixed threshold: 0.5
- Threshold caveat: threshold tuned in-sample

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| metadata | fixed | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.001 | 0.000 | 0.250 |
| metadata | tuned | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.001 | 0.000 | 0.250 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| metadata | clean | 0.500 | 0.500 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| metadata | facebook | only one label present; AUC undefined |
| metadata | instagram | only one label present; AUC undefined |
| metadata | linkedin | only one label present; AUC undefined |
| metadata | none | only one label present; AUC undefined |
| metadata | x | only one label present; AUC undefined |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| metadata | ITW-SM | 0.500 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| metadata | 2000 | 0.37 |
