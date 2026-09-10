# Benchmark report: WildRF

- Entries evaluated: 398
- Robustness levels: clean
- Fixed threshold: 0.5

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| c2pa | fixed | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | 0.250 |
| c2pa | tuned | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | 0.250 |
| copy_move | fixed | 0.502 | 0.501 | 0.503 | 0.503 | 0.030 | 0.035 | 0.052 | 0.253 |
| copy_move | tuned | 0.502 | 0.501 | 0.503 | 0.503 | 0.030 | 0.035 | 0.052 | 0.253 |
| dinov2_head | fixed | 0.988 | 0.988 | 0.930 | 0.930 | 0.131 | 0.990 | 0.066 | 0.056 |
| dinov2_head | tuned | 0.988 | 0.988 | 0.947 | 0.947 | 0.055 | 0.950 | 0.066 | 0.056 |
| double_jpeg | fixed | 0.742 | 0.717 | 0.477 | 0.477 | 0.055 | 0.010 | 0.238 | 0.248 |
| double_jpeg | tuned | 0.742 | 0.717 | 0.759 | 0.759 | 0.101 | 0.618 | 0.238 | 0.248 |
| ela | fixed | 0.628 | 0.581 | 0.608 | 0.608 | 0.628 | 0.844 | 0.067 | 0.238 |
| ela | tuned | 0.628 | 0.581 | 0.618 | 0.618 | 0.588 | 0.824 | 0.067 | 0.238 |
| jpeg_ghost | fixed | 0.303 | 0.398 | 0.432 | 0.432 | 0.945 | 0.809 | 0.275 | 0.308 |
| jpeg_ghost | tuned | 0.303 | 0.398 | 0.500 | 0.500 | 0.060 | 0.060 | 0.275 | 0.308 |
| metadata | fixed | 0.480 | 0.500 | 0.480 | 0.480 | 0.040 | 0.000 | 0.024 | 0.255 |
| metadata | tuned | 0.480 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.024 | 0.255 |
| sd_watermark | fixed | 0.500 | 0.503 | 0.503 | 0.503 | 0.000 | 0.005 | 0.051 | 0.252 |
| sd_watermark | tuned | 0.500 | 0.503 | 0.503 | 0.503 | 0.000 | 0.005 | 0.051 | 0.252 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| c2pa | clean | 0.500 | 0.500 |
| copy_move | clean | 0.502 | 0.503 |
| dinov2_head | clean | 0.988 | 0.930 |
| double_jpeg | clean | 0.742 | 0.477 |
| ela | clean | 0.628 | 0.608 |
| jpeg_ghost | clean | 0.303 | 0.432 |
| metadata | clean | 0.480 | 0.480 |
| sd_watermark | clean | 0.500 | 0.503 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| c2pa | none | 0.500 |
| copy_move | none | 0.502 |
| dinov2_head | none | 0.988 |
| double_jpeg | none | 0.742 |
| ela | none | 0.628 |
| jpeg_ghost | none | 0.303 |
| metadata | none | 0.480 |
| sd_watermark | none | 0.500 |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| c2pa | WildRF | 0.500 |
| copy_move | WildRF | 0.502 |
| dinov2_head | WildRF | 0.988 |
| double_jpeg | WildRF | 0.742 |
| ela | WildRF | 0.628 |
| jpeg_ghost | WildRF | 0.303 |
| metadata | WildRF | 0.480 |
| sd_watermark | WildRF | 0.500 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| c2pa | 398 | 2.53 |
| copy_move | 398 | 332.13 |
| dinov2_head | 398 | 124.51 |
| double_jpeg | 398 | 600.57 |
| ela | 398 | 618.89 |
| jpeg_ghost | 398 | 578.77 |
| metadata | 398 | 18.35 |
| sd_watermark | 398 | 1821.83 |
