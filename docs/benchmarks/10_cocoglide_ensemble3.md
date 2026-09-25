# Benchmark report: CocoGlide

- Entries evaluated: 1024
- Robustness levels: clean
- Fixed threshold: 0.5
- Threshold caveat: threshold tuned in-sample

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| localizer_ensemble | fixed | 0.743 | 0.776 | 0.548 | 0.548 | 0.848 | 0.943 | 0.312 | 0.318 |
| localizer_ensemble | tuned | 0.743 | 0.776 | 0.690 | 0.690 | 0.232 | 0.613 | 0.312 | 0.318 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| localizer_ensemble | clean | 0.743 | 0.548 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| localizer_ensemble | glide | only one label present; AUC undefined |
| localizer_ensemble | none | only one label present; AUC undefined |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| localizer_ensemble | CocoGlide | 0.743 |

## Pixel metrics

| detector | level | n | mean f1@0.5 | mean best-f1 | mean ap | mean iou |
|---|---|---|---|---|---|---|
| localizer_ensemble | clean | 512 | 0.489 | 0.652 | 0.643 | 0.381 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| localizer_ensemble | 1024 | 531.69 |
