# Benchmark report: WildRF

- Entries evaluated: 1000
- Robustness levels: clean, jpeg_q95, jpeg_q85, jpeg_q75, jpeg_q60, jpeg_q50, webp_q80, resize_0.75, resize_0.5, resize_0.25, roundtrip_0.5, crop_0.8, noise_2, noise_5, social_1080_q80
- Fixed threshold: 0.5

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| c2pa | fixed | 0.500 | 0.534 | 0.466 | 0.500 | 0.000 | 0.000 | 0.034 | 0.250 |
| c2pa | tuned | 0.500 | 0.534 | 0.466 | 0.500 | 0.000 | 0.000 | 0.034 | 0.250 |
| copy_move | fixed | 0.506 | 0.537 | 0.474 | 0.506 | 0.021 | 0.034 | 0.083 | 0.256 |
| copy_move | tuned | 0.506 | 0.537 | 0.474 | 0.506 | 0.021 | 0.034 | 0.083 | 0.256 |
| dinov2_head | fixed | 0.980 | 0.983 | 0.920 | 0.916 | 0.137 | 0.970 | 0.070 | 0.064 |
| dinov2_head | tuned | 0.980 | 0.983 | 0.936 | 0.936 | 0.060 | 0.933 | 0.070 | 0.064 |
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
| copy_move | clean | 0.506 | 0.506 |
| copy_move | jpeg_q95 | 0.506 | 0.506 |
| copy_move | jpeg_q85 | 0.511 | 0.511 |
| copy_move | jpeg_q75 | 0.507 | 0.507 |
| copy_move | jpeg_q60 | 0.507 | 0.507 |
| copy_move | jpeg_q50 | 0.509 | 0.509 |
| copy_move | webp_q80 | 0.504 | 0.504 |
| copy_move | resize_0.75 | 0.502 | 0.502 |
| copy_move | resize_0.5 | 0.501 | 0.501 |
| copy_move | resize_0.25 | 0.499 | 0.499 |
| copy_move | roundtrip_0.5 | 0.509 | 0.509 |
| copy_move | crop_0.8 | 0.506 | 0.506 |
| copy_move | noise_2 | 0.505 | 0.505 |
| copy_move | noise_5 | 0.503 | 0.503 |
| copy_move | social_1080_q80 | 0.508 | 0.508 |
| dinov2_head | clean | 0.980 | 0.916 |
| dinov2_head | jpeg_q95 | 0.980 | 0.918 |
| dinov2_head | jpeg_q85 | 0.979 | 0.919 |
| dinov2_head | jpeg_q75 | 0.977 | 0.915 |
| dinov2_head | jpeg_q60 | 0.977 | 0.926 |
| dinov2_head | jpeg_q50 | 0.977 | 0.927 |
| dinov2_head | webp_q80 | 0.976 | 0.919 |
| dinov2_head | resize_0.75 | 0.978 | 0.884 |
| dinov2_head | resize_0.5 | 0.965 | 0.835 |
| dinov2_head | resize_0.25 | 0.904 | 0.739 |
| dinov2_head | roundtrip_0.5 | 0.972 | 0.886 |
| dinov2_head | crop_0.8 | 0.979 | 0.911 |
| dinov2_head | noise_2 | 0.979 | 0.921 |
| dinov2_head | noise_5 | 0.968 | 0.906 |
| dinov2_head | social_1080_q80 | 0.968 | 0.872 |
| double_jpeg | clean | 0.655 | 0.476 |
| double_jpeg | jpeg_q95 | 0.294 | 0.295 |
| double_jpeg | jpeg_q85 | 0.440 | 0.440 |
| double_jpeg | jpeg_q75 | 0.480 | 0.480 |
| double_jpeg | jpeg_q60 | 0.478 | 0.478 |
| double_jpeg | jpeg_q50 | 0.495 | 0.495 |
| double_jpeg | webp_q80 | 0.499 | 0.490 |
| double_jpeg | resize_0.75 | 0.507 | 0.501 |
| double_jpeg | resize_0.5 | 0.504 | 0.498 |
| double_jpeg | resize_0.25 | 0.498 | 0.499 |
| double_jpeg | roundtrip_0.5 | 0.507 | 0.494 |
| double_jpeg | crop_0.8 | 0.423 | 0.401 |
| double_jpeg | noise_2 | 0.543 | 0.499 |
| double_jpeg | noise_5 | 0.498 | 0.499 |
| double_jpeg | social_1080_q80 | 0.481 | 0.481 |
| ela | clean | 0.532 | 0.541 |
| ela | jpeg_q95 | 0.565 | 0.535 |
| ela | jpeg_q85 | 0.481 | 0.487 |
| ela | jpeg_q75 | 0.484 | 0.488 |
| ela | jpeg_q60 | 0.470 | 0.476 |
| ela | jpeg_q50 | 0.479 | 0.474 |
| ela | webp_q80 | 0.422 | 0.473 |
| ela | resize_0.75 | 0.397 | 0.456 |
| ela | resize_0.5 | 0.387 | 0.469 |
| ela | resize_0.25 | 0.440 | 0.475 |
| ela | roundtrip_0.5 | 0.477 | 0.501 |
| ela | crop_0.8 | 0.462 | 0.486 |
| ela | noise_2 | 0.457 | 0.481 |
| ela | noise_5 | 0.494 | 0.498 |
| ela | social_1080_q80 | 0.450 | 0.454 |
| jpeg_ghost | clean | 0.345 | 0.464 |
| jpeg_ghost | jpeg_q95 | 0.406 | 0.494 |
| jpeg_ghost | jpeg_q85 | 0.377 | 0.486 |
| jpeg_ghost | jpeg_q75 | 0.379 | 0.472 |
| jpeg_ghost | jpeg_q60 | 0.297 | 0.448 |
| jpeg_ghost | jpeg_q50 | 0.293 | 0.446 |
| jpeg_ghost | webp_q80 | 0.421 | 0.503 |
| jpeg_ghost | resize_0.75 | 0.330 | 0.465 |
| jpeg_ghost | resize_0.5 | 0.323 | 0.472 |
| jpeg_ghost | resize_0.25 | 0.372 | 0.494 |
| jpeg_ghost | roundtrip_0.5 | 0.433 | 0.511 |
| jpeg_ghost | crop_0.8 | 0.397 | 0.490 |
| jpeg_ghost | noise_2 | 0.387 | 0.477 |
| jpeg_ghost | noise_5 | 0.488 | 0.493 |
| jpeg_ghost | social_1080_q80 | 0.337 | 0.464 |
| metadata | clean | 0.465 | 0.465 |
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
| sd_watermark | clean | 0.500 | 0.501 |
| sd_watermark | jpeg_q95 | 0.499 | 0.500 |
| sd_watermark | jpeg_q85 | 0.499 | 0.500 |
| sd_watermark | jpeg_q75 | 0.499 | 0.500 |
| sd_watermark | jpeg_q60 | 0.499 | 0.500 |
| sd_watermark | jpeg_q50 | 0.499 | 0.500 |
| sd_watermark | webp_q80 | 0.499 | 0.500 |
| sd_watermark | resize_0.75 | 0.498 | 0.500 |
| sd_watermark | resize_0.5 | 0.499 | 0.500 |
| sd_watermark | resize_0.25 | 0.509 | 0.500 |
| sd_watermark | roundtrip_0.5 | 0.499 | 0.500 |
| sd_watermark | crop_0.8 | 0.498 | 0.500 |
| sd_watermark | noise_2 | 0.500 | 0.501 |
| sd_watermark | noise_5 | 0.500 | 0.501 |
| sd_watermark | social_1080_q80 | 0.499 | 0.500 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| c2pa | none | 0.500 |
| copy_move | none | 0.506 |
| dinov2_head | none | 0.980 |
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
| dinov2_head | WildRF | 0.980 |
| double_jpeg | WildRF | 0.655 |
| ela | WildRF | 0.532 |
| jpeg_ghost | WildRF | 0.345 |
| metadata | WildRF | 0.465 |
| sd_watermark | WildRF | 0.500 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| c2pa | 15000 | 1.73 |
| copy_move | 15000 | 332.76 |
| dinov2_head | 15000 | 94.58 |
| double_jpeg | 15000 | 348.27 |
| ela | 15000 | 466.64 |
| jpeg_ghost | 15000 | 592.37 |
| metadata | 15000 | 47.37 |
| sd_watermark | 15000 | 1338.86 |
