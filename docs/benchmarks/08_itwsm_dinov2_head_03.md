# Benchmark report: ITW-SM

- Entries evaluated: 10000
- Robustness levels: clean
- Fixed threshold: 0.5
- Threshold caveat: threshold tuned in-sample

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| dinov2_head | fixed | 0.819 | 0.817 | 0.673 | 0.673 | 0.585 | 0.931 | 0.243 | 0.250 |
| dinov2_head | tuned | 0.819 | 0.817 | 0.740 | 0.740 | 0.212 | 0.691 | 0.243 | 0.250 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| dinov2_head | clean | 0.819 | 0.673 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| dinov2_head | facebook | only one label present; AUC undefined |
| dinov2_head | instagram | only one label present; AUC undefined |
| dinov2_head | linkedin | only one label present; AUC undefined |
| dinov2_head | none | only one label present; AUC undefined |
| dinov2_head | x | only one label present; AUC undefined |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| dinov2_head | ITW-SM | 0.819 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| dinov2_head | 10000 | 70.09 |
