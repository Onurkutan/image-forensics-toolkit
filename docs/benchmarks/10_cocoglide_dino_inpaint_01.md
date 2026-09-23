# Benchmark report: CocoGlide

- Entries evaluated: 1024
- Robustness levels: clean
- Fixed threshold: 0.5
- Threshold caveat: threshold tuned in-sample

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| dino_inpaint | fixed | 0.768 | 0.803 | 0.704 | 0.704 | 0.326 | 0.734 | 0.174 | 0.228 |
| dino_inpaint | tuned | 0.768 | 0.803 | 0.706 | 0.706 | 0.322 | 0.734 | 0.174 | 0.228 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| dino_inpaint | clean | 0.768 | 0.704 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| dino_inpaint | glide | only one label present; AUC undefined |
| dino_inpaint | none | only one label present; AUC undefined |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| dino_inpaint | CocoGlide | 0.768 |

## Pixel metrics

| detector | level | n | mean f1@0.5 | mean best-f1 | mean ap | mean iou |
|---|---|---|---|---|---|---|
| dino_inpaint | clean | 512 | 0.425 | 0.637 | 0.672 | 0.333 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| dino_inpaint | 1024 | 20.05 |
