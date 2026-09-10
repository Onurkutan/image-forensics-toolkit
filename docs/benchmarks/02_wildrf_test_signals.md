# Benchmark report: WildRF

- Entries evaluated: 1000
- Robustness levels: clean
- Fixed threshold: 0.5

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| c2pa | fixed | 0.500 | 0.534 | 0.466 | 0.500 | 0.000 | 0.000 | 0.034 | 0.250 |
| c2pa | tuned | 0.500 | 0.534 | 0.466 | 0.500 | 0.000 | 0.000 | 0.034 | 0.250 |
| copy_move | fixed | 0.506 | 0.537 | 0.474 | 0.506 | 0.021 | 0.034 | 0.083 | 0.256 |
| copy_move | tuned | 0.506 | 0.537 | 0.474 | 0.506 | 0.021 | 0.034 | 0.083 | 0.256 |
| double_jpeg | fixed | 0.655 | 0.655 | 0.445 | 0.476 | 0.069 | 0.021 | 0.144 | 0.259 |
| double_jpeg | tuned | 0.655 | 0.655 | 0.652 | 0.669 | 0.088 | 0.425 | 0.144 | 0.259 |
| ela | fixed | 0.532 | 0.546 | 0.557 | 0.541 | 0.693 | 0.775 | 0.092 | 0.251 |
| ela | tuned | 0.532 | 0.546 | 0.579 | 0.557 | 0.766 | 0.880 | 0.092 | 0.251 |
| jpeg_ghost | fixed | 0.345 | 0.436 | 0.492 | 0.464 | 0.940 | 0.869 | 0.211 | 0.288 |
| jpeg_ghost | tuned | 0.345 | 0.436 | 0.466 | 0.500 | 0.000 | 0.000 | 0.211 | 0.288 |
| metadata | fixed | 0.465 | 0.527 | 0.435 | 0.465 | 0.092 | 0.022 | 0.076 | 0.258 |
| metadata | tuned | 0.465 | 0.527 | 0.466 | 0.500 | 0.000 | 0.000 | 0.076 | 0.258 |
| sd_watermark | fixed | 0.500 | 0.535 | 0.467 | 0.501 | 0.000 | 0.002 | 0.084 | 0.256 |
| sd_watermark | tuned | 0.500 | 0.535 | 0.467 | 0.501 | 0.000 | 0.002 | 0.084 | 0.256 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| c2pa | clean | 0.500 | 0.500 |
| copy_move | clean | 0.506 | 0.506 |
| double_jpeg | clean | 0.655 | 0.476 |
| ela | clean | 0.532 | 0.541 |
| jpeg_ghost | clean | 0.345 | 0.464 |
| metadata | clean | 0.465 | 0.465 |
| sd_watermark | clean | 0.500 | 0.501 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| c2pa | none | 0.500 |
| copy_move | none | 0.506 |
| double_jpeg | none | 0.655 |
| ela | none | 0.532 |
| jpeg_ghost | none | 0.345 |
| metadata | none | 0.465 |
| sd_watermark | none | 0.500 |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| c2pa | WildRF | 0.500 |
| copy_move | WildRF | 0.506 |
| double_jpeg | WildRF | 0.655 |
| ela | WildRF | 0.532 |
| jpeg_ghost | WildRF | 0.345 |
| metadata | WildRF | 0.465 |
| sd_watermark | WildRF | 0.500 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| c2pa | 1000 | 1.95 |
| copy_move | 1000 | 345.11 |
| double_jpeg | 1000 | 552.46 |
| ela | 1000 | 537.05 |
| jpeg_ghost | 1000 | 598.85 |
| metadata | 1000 | 13.02 |
| sd_watermark | 1000 | 1612.61 |
