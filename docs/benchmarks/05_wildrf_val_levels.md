# Benchmark report: WildRF

- Entries evaluated: 398
- Robustness levels: clean, jpeg_q95, jpeg_q85, jpeg_q75, jpeg_q60, jpeg_q50, webp_q80, resize_0.75, resize_0.5, resize_0.25, roundtrip_0.5, crop_0.8, noise_2, noise_5, social_1080_q80
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
| c2pa | jpeg_q95 | 0.500 | 0.500 |
| c2pa | jpeg_q85 | 0.500 | 0.500 |
| c2pa | jpeg_q75 | 0.500 | 0.500 |
| c2pa | jpeg_q60 | 0.500 | 0.500 |
| c2pa | jpeg_q50 | 0.500 | 0.500 |
| c2pa | webp_q80 | 0.500 | 0.500 |
| c2pa | resize_0.75 | 0.500 | 0.500 |
| c2pa | resize_0.5 | 0.500 | 0.500 |
| c2pa | resize_0.25 | 0.500 | 0.500 |
| c2pa | roundtrip_0.5 | 0.500 | 0.500 |
| c2pa | crop_0.8 | 0.500 | 0.500 |
| c2pa | noise_2 | 0.500 | 0.500 |
| c2pa | noise_5 | 0.500 | 0.500 |
| c2pa | social_1080_q80 | 0.500 | 0.500 |
| copy_move | clean | 0.502 | 0.503 |
| copy_move | jpeg_q95 | 0.507 | 0.508 |
| copy_move | jpeg_q85 | 0.500 | 0.500 |
| copy_move | jpeg_q75 | 0.505 | 0.505 |
| copy_move | jpeg_q60 | 0.495 | 0.495 |
| copy_move | jpeg_q50 | 0.512 | 0.513 |
| copy_move | webp_q80 | 0.490 | 0.490 |
| copy_move | resize_0.75 | 0.497 | 0.497 |
| copy_move | resize_0.5 | 0.505 | 0.505 |
| copy_move | resize_0.25 | 0.487 | 0.487 |
| copy_move | roundtrip_0.5 | 0.492 | 0.492 |
| copy_move | crop_0.8 | 0.485 | 0.485 |
| copy_move | noise_2 | 0.487 | 0.487 |
| copy_move | noise_5 | 0.493 | 0.492 |
| copy_move | social_1080_q80 | 0.500 | 0.500 |
| dinov2_head | clean | 0.988 | 0.930 |
| dinov2_head | jpeg_q95 | 0.987 | 0.930 |
| dinov2_head | jpeg_q85 | 0.987 | 0.922 |
| dinov2_head | jpeg_q75 | 0.984 | 0.922 |
| dinov2_head | jpeg_q60 | 0.986 | 0.922 |
| dinov2_head | jpeg_q50 | 0.984 | 0.932 |
| dinov2_head | webp_q80 | 0.982 | 0.920 |
| dinov2_head | resize_0.75 | 0.984 | 0.892 |
| dinov2_head | resize_0.5 | 0.971 | 0.842 |
| dinov2_head | resize_0.25 | 0.908 | 0.734 |
| dinov2_head | roundtrip_0.5 | 0.979 | 0.905 |
| dinov2_head | crop_0.8 | 0.987 | 0.925 |
| dinov2_head | noise_2 | 0.985 | 0.922 |
| dinov2_head | noise_5 | 0.973 | 0.902 |
| dinov2_head | social_1080_q80 | 0.976 | 0.862 |
| double_jpeg | clean | 0.742 | 0.477 |
| double_jpeg | jpeg_q95 | 0.260 | 0.259 |
| double_jpeg | jpeg_q85 | 0.422 | 0.422 |
| double_jpeg | jpeg_q75 | 0.493 | 0.492 |
| double_jpeg | jpeg_q60 | 0.478 | 0.477 |
| double_jpeg | jpeg_q50 | 0.500 | 0.500 |
| double_jpeg | webp_q80 | 0.505 | 0.500 |
| double_jpeg | resize_0.75 | 0.495 | 0.495 |
| double_jpeg | resize_0.5 | 0.498 | 0.500 |
| double_jpeg | resize_0.25 | 0.497 | 0.508 |
| double_jpeg | roundtrip_0.5 | 0.508 | 0.500 |
| double_jpeg | crop_0.8 | 0.416 | 0.389 |
| double_jpeg | noise_2 | 0.545 | 0.500 |
| double_jpeg | noise_5 | 0.500 | 0.500 |
| double_jpeg | social_1080_q80 | 0.470 | 0.470 |
| ela | clean | 0.628 | 0.608 |
| ela | jpeg_q95 | 0.608 | 0.593 |
| ela | jpeg_q85 | 0.520 | 0.510 |
| ela | jpeg_q75 | 0.536 | 0.525 |
| ela | jpeg_q60 | 0.465 | 0.475 |
| ela | jpeg_q50 | 0.482 | 0.480 |
| ela | webp_q80 | 0.442 | 0.500 |
| ela | resize_0.75 | 0.374 | 0.503 |
| ela | resize_0.5 | 0.406 | 0.482 |
| ela | resize_0.25 | 0.400 | 0.490 |
| ela | roundtrip_0.5 | 0.518 | 0.520 |
| ela | crop_0.8 | 0.455 | 0.508 |
| ela | noise_2 | 0.447 | 0.492 |
| ela | noise_5 | 0.500 | 0.500 |
| ela | social_1080_q80 | 0.420 | 0.425 |
| jpeg_ghost | clean | 0.303 | 0.432 |
| jpeg_ghost | jpeg_q95 | 0.376 | 0.492 |
| jpeg_ghost | jpeg_q85 | 0.377 | 0.482 |
| jpeg_ghost | jpeg_q75 | 0.378 | 0.475 |
| jpeg_ghost | jpeg_q60 | 0.318 | 0.440 |
| jpeg_ghost | jpeg_q50 | 0.316 | 0.455 |
| jpeg_ghost | webp_q80 | 0.402 | 0.500 |
| jpeg_ghost | resize_0.75 | 0.282 | 0.442 |
| jpeg_ghost | resize_0.5 | 0.289 | 0.445 |
| jpeg_ghost | resize_0.25 | 0.351 | 0.487 |
| jpeg_ghost | roundtrip_0.5 | 0.388 | 0.497 |
| jpeg_ghost | crop_0.8 | 0.324 | 0.482 |
| jpeg_ghost | noise_2 | 0.324 | 0.440 |
| jpeg_ghost | noise_5 | 0.423 | 0.480 |
| jpeg_ghost | social_1080_q80 | 0.331 | 0.467 |
| metadata | clean | 0.480 | 0.480 |
| metadata | jpeg_q95 | 0.500 | 0.500 |
| metadata | jpeg_q85 | 0.500 | 0.500 |
| metadata | jpeg_q75 | 0.500 | 0.500 |
| metadata | jpeg_q60 | 0.500 | 0.500 |
| metadata | jpeg_q50 | 0.500 | 0.500 |
| metadata | webp_q80 | 0.500 | 0.500 |
| metadata | resize_0.75 | 0.500 | 0.500 |
| metadata | resize_0.5 | 0.500 | 0.500 |
| metadata | resize_0.25 | 0.500 | 0.500 |
| metadata | roundtrip_0.5 | 0.500 | 0.500 |
| metadata | crop_0.8 | 0.500 | 0.500 |
| metadata | noise_2 | 0.500 | 0.500 |
| metadata | noise_5 | 0.500 | 0.500 |
| metadata | social_1080_q80 | 0.500 | 0.500 |
| sd_watermark | clean | 0.500 | 0.503 |
| sd_watermark | jpeg_q95 | 0.497 | 0.500 |
| sd_watermark | jpeg_q85 | 0.497 | 0.500 |
| sd_watermark | jpeg_q75 | 0.497 | 0.500 |
| sd_watermark | jpeg_q60 | 0.497 | 0.500 |
| sd_watermark | jpeg_q50 | 0.497 | 0.500 |
| sd_watermark | webp_q80 | 0.497 | 0.500 |
| sd_watermark | resize_0.75 | 0.497 | 0.500 |
| sd_watermark | resize_0.5 | 0.485 | 0.500 |
| sd_watermark | resize_0.25 | 0.533 | 0.500 |
| sd_watermark | roundtrip_0.5 | 0.497 | 0.500 |
| sd_watermark | crop_0.8 | 0.497 | 0.500 |
| sd_watermark | noise_2 | 0.500 | 0.503 |
| sd_watermark | noise_5 | 0.500 | 0.503 |
| sd_watermark | social_1080_q80 | 0.497 | 0.500 |

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
| c2pa | 5970 | 1.94 |
| copy_move | 5970 | 302.28 |
| dinov2_head | 5970 | 120.46 |
| double_jpeg | 5970 | 369.47 |
| ela | 5970 | 500.62 |
| jpeg_ghost | 5970 | 543.11 |
| metadata | 5970 | 51.05 |
| sd_watermark | 5970 | 1455.55 |
