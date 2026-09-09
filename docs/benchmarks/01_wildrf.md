# Benchmark report: WildRF

- Entries evaluated: 1000
- Robustness levels: clean
- Fixed threshold: 0.5

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| c2pa | fixed | 0.500 | 0.561 | 0.439 | 0.500 | 0.000 | 0.000 | 0.061 | 0.250 |
| c2pa | tuned | 0.500 | 0.561 | 0.439 | 0.500 | 0.000 | 0.000 | 0.061 | 0.250 |
| copy_move | fixed | 0.517 | 0.572 | 0.460 | 0.517 | 0.016 | 0.050 | 0.106 | 0.256 |
| copy_move | tuned | 0.517 | 0.572 | 0.460 | 0.517 | 0.016 | 0.050 | 0.106 | 0.256 |
| dinov2_head | fixed | 0.459 | 0.539 | 0.556 | 0.500 | 0.957 | 0.957 | 0.430 | 0.434 |
| dinov2_head | tuned | 0.459 | 0.539 | 0.439 | 0.500 | 0.000 | 0.000 | 0.430 | 0.434 |
| double_jpeg | fixed | 0.721 | 0.736 | 0.425 | 0.482 | 0.048 | 0.012 | 0.180 | 0.259 |
| double_jpeg | tuned | 0.721 | 0.736 | 0.712 | 0.734 | 0.089 | 0.556 | 0.180 | 0.259 |
| ela | fixed | 0.579 | 0.608 | 0.589 | 0.562 | 0.663 | 0.786 | 0.049 | 0.240 |
| ela | tuned | 0.579 | 0.608 | 0.600 | 0.551 | 0.847 | 0.950 | 0.049 | 0.240 |
| jpeg_ghost | fixed | 0.328 | 0.456 | 0.507 | 0.459 | 0.932 | 0.850 | 0.216 | 0.282 |
| jpeg_ghost | tuned | 0.328 | 0.456 | 0.420 | 0.472 | 0.103 | 0.046 | 0.216 | 0.282 |
| metadata | fixed | 0.476 | 0.558 | 0.419 | 0.476 | 0.055 | 0.007 | 0.087 | 0.255 |
| metadata | tuned | 0.476 | 0.558 | 0.439 | 0.500 | 0.000 | 0.000 | 0.087 | 0.255 |
| sd_watermark | fixed | 0.497 | 0.563 | 0.441 | 0.502 | 0.000 | 0.004 | 0.114 | 0.258 |
| sd_watermark | tuned | 0.497 | 0.563 | 0.437 | 0.497 | 0.009 | 0.004 | 0.114 | 0.258 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| c2pa | clean | 0.500 | 0.500 |
| copy_move | clean | 0.517 | 0.517 |
| dinov2_head | clean | 0.459 | 0.500 |
| double_jpeg | clean | 0.721 | 0.482 |
| ela | clean | 0.579 | 0.562 |
| jpeg_ghost | clean | 0.328 | 0.459 |
| metadata | clean | 0.476 | 0.476 |
| sd_watermark | clean | 0.497 | 0.502 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| c2pa | none | 0.500 |
| copy_move | none | 0.517 |
| dinov2_head | none | 0.459 |
| double_jpeg | none | 0.721 |
| ela | none | 0.579 |
| jpeg_ghost | none | 0.328 |
| metadata | none | 0.476 |
| sd_watermark | none | 0.497 |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| c2pa | WildRF | 0.500 |
| copy_move | WildRF | 0.517 |
| dinov2_head | WildRF | 0.459 |
| double_jpeg | WildRF | 0.721 |
| ela | WildRF | 0.579 |
| jpeg_ghost | WildRF | 0.328 |
| metadata | WildRF | 0.476 |
| sd_watermark | WildRF | 0.497 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| c2pa | 1000 | 1.09 |
| copy_move | 1000 | 208.70 |
| dinov2_head | 1000 | 106.63 |
| double_jpeg | 1000 | 312.61 |
| ela | 1000 | 319.55 |
| jpeg_ghost | 1000 | 366.33 |
| metadata | 1000 | 12.12 |
| sd_watermark | 1000 | 931.21 |
