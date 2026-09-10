# Benchmark report: CocoGlide

- Entries evaluated: 1024
- Robustness levels: clean
- Fixed threshold: 0.5
- Threshold caveat: threshold tuned in-sample

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| iml_vit | fixed | 0.535 | 0.539 | 0.523 | 0.523 | 0.484 | 0.531 | 0.174 | 0.291 |
| iml_vit | tuned | 0.535 | 0.539 | 0.541 | 0.541 | 0.316 | 0.398 | 0.174 | 0.291 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| iml_vit | clean | 0.535 | 0.523 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| iml_vit | glide | only one label present; AUC undefined |
| iml_vit | none | only one label present; AUC undefined |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| iml_vit | CocoGlide | 0.535 |

## Pixel metrics

| detector | n | mean f1@0.5 | mean best-f1 | mean ap | mean iou |
|---|---|---|---|---|---|
| iml_vit | 512 | 0.059 | 0.486 | 0.423 | 0.037 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| iml_vit | 1024 | 462.31 |
