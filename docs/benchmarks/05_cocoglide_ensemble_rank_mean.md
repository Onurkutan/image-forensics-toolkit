# Benchmark report: CocoGlide

- Entries evaluated: 1024
- Robustness levels: clean
- Fixed threshold: 0.5
- Threshold caveat: threshold tuned in-sample

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| localizer_ensemble | fixed | 0.457 | 0.484 | 0.500 | 0.500 | 1.000 | 1.000 | 0.450 | 0.456 |
| localizer_ensemble | tuned | 0.457 | 0.484 | 0.506 | 0.506 | 0.236 | 0.248 | 0.450 | 0.456 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| localizer_ensemble | clean | 0.457 | 0.500 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| localizer_ensemble | glide | only one label present; AUC undefined |
| localizer_ensemble | none | only one label present; AUC undefined |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| localizer_ensemble | CocoGlide | 0.457 |

## Pixel metrics

| detector | level | n | mean f1@0.5 | mean best-f1 | mean ap | mean iou |
|---|---|---|---|---|---|---|
| localizer_ensemble | clean | 512 | 0.431 | 0.577 | 0.525 | 0.308 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| localizer_ensemble | 1024 | 751.88 |
