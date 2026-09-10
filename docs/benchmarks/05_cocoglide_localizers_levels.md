# Benchmark report: CocoGlide

- Entries evaluated: 512
- Robustness levels: clean, jpeg_q95, jpeg_q85, jpeg_q75, jpeg_q60, jpeg_q50, webp_q80, noise_2, noise_5
- Fixed threshold: 0.5
- Threshold caveat: threshold tuned in-sample

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| catnet_v2 | fixed | 0.633 | 0.656 | 0.551 | 0.551 | 0.588 | 0.690 | 0.241 | 0.295 |
| catnet_v2 | tuned | 0.633 | 0.656 | 0.625 | 0.624 | 0.241 | 0.490 | 0.241 | 0.295 |
| iml_vit | fixed | 0.497 | 0.512 | 0.492 | 0.492 | 0.533 | 0.518 | 0.203 | 0.305 |
| iml_vit | tuned | 0.497 | 0.512 | 0.525 | 0.525 | 0.335 | 0.384 | 0.203 | 0.305 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| catnet_v2 | clean | 0.633 | 0.551 |
| catnet_v2 | jpeg_q95 | 0.738 | 0.677 |
| catnet_v2 | jpeg_q85 | 0.724 | 0.658 |
| catnet_v2 | jpeg_q75 | 0.717 | 0.647 |
| catnet_v2 | jpeg_q60 | 0.643 | 0.598 |
| catnet_v2 | jpeg_q50 | 0.629 | 0.594 |
| catnet_v2 | webp_q80 | 0.561 | 0.518 |
| catnet_v2 | noise_2 | 0.609 | 0.557 |
| catnet_v2 | noise_5 | 0.540 | 0.529 |
| iml_vit | clean | 0.497 | 0.492 |
| iml_vit | jpeg_q95 | 0.485 | 0.490 |
| iml_vit | jpeg_q85 | 0.498 | 0.506 |
| iml_vit | jpeg_q75 | 0.505 | 0.498 |
| iml_vit | jpeg_q60 | 0.522 | 0.512 |
| iml_vit | jpeg_q50 | 0.516 | 0.508 |
| iml_vit | webp_q80 | 0.504 | 0.480 |
| iml_vit | noise_2 | 0.514 | 0.504 |
| iml_vit | noise_5 | 0.504 | 0.481 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| catnet_v2 | glide | only one label present; AUC undefined |
| catnet_v2 | none | only one label present; AUC undefined |
| iml_vit | glide | only one label present; AUC undefined |
| iml_vit | none | only one label present; AUC undefined |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| catnet_v2 | CocoGlide | 0.633 |
| iml_vit | CocoGlide | 0.497 |

## Pixel metrics

| detector | level | n | mean f1@0.5 | mean best-f1 | mean ap | mean iou |
|---|---|---|---|---|---|---|
| catnet_v2 | clean | 255 | 0.350 | 0.598 | 0.569 | 0.274 |
| catnet_v2 | jpeg_q95 | 255 | 0.288 | 0.619 | 0.596 | 0.235 |
| catnet_v2 | jpeg_q85 | 255 | 0.313 | 0.631 | 0.604 | 0.254 |
| catnet_v2 | jpeg_q75 | 255 | 0.388 | 0.609 | 0.571 | 0.310 |
| catnet_v2 | jpeg_q60 | 255 | 0.320 | 0.614 | 0.577 | 0.253 |
| catnet_v2 | jpeg_q50 | 255 | 0.330 | 0.605 | 0.563 | 0.259 |
| catnet_v2 | webp_q80 | 255 | 0.280 | 0.559 | 0.514 | 0.213 |
| catnet_v2 | noise_2 | 255 | 0.276 | 0.592 | 0.565 | 0.214 |
| catnet_v2 | noise_5 | 255 | 0.147 | 0.516 | 0.470 | 0.106 |
| iml_vit | clean | 255 | 0.064 | 0.474 | 0.406 | 0.041 |
| iml_vit | jpeg_q95 | 255 | 0.044 | 0.446 | 0.381 | 0.030 |
| iml_vit | jpeg_q85 | 255 | 0.058 | 0.468 | 0.403 | 0.038 |
| iml_vit | jpeg_q75 | 255 | 0.073 | 0.478 | 0.407 | 0.048 |
| iml_vit | jpeg_q60 | 255 | 0.081 | 0.485 | 0.410 | 0.053 |
| iml_vit | jpeg_q50 | 255 | 0.082 | 0.490 | 0.418 | 0.053 |
| iml_vit | webp_q80 | 255 | 0.054 | 0.451 | 0.383 | 0.036 |
| iml_vit | noise_2 | 255 | 0.068 | 0.475 | 0.406 | 0.043 |
| iml_vit | noise_5 | 255 | 0.072 | 0.481 | 0.414 | 0.047 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| catnet_v2 | 4608 | 302.72 |
| iml_vit | 4608 | 346.71 |
