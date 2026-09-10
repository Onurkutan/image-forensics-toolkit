# Benchmark report: COCO+Synthbuster

- Entries evaluated: 2800
- Robustness levels: clean, jpeg_q95, jpeg_q85, jpeg_q75, jpeg_q60, jpeg_q50, webp_q80, resize_0.75, resize_0.5, resize_0.25, roundtrip_0.5, crop_0.8, noise_2, noise_5, social_1080_q80
- Fixed threshold: 0.5

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| dinov2_head | fixed | 0.988 | 0.992 | 0.943 | 0.950 | 0.026 | 0.926 | 0.060 | 0.048 |
| dinov2_head | tuned | 0.988 | 0.992 | 0.943 | 0.950 | 0.026 | 0.926 | 0.060 | 0.048 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| dinov2_head | clean | 0.988 | 0.950 |
| dinov2_head | jpeg_q95 | 0.986 | 0.946 |
| dinov2_head | jpeg_q85 | 0.985 | 0.936 |
| dinov2_head | jpeg_q75 | 0.980 | 0.929 |
| dinov2_head | jpeg_q60 | 0.979 | 0.925 |
| dinov2_head | jpeg_q50 | 0.977 | 0.922 |
| dinov2_head | webp_q80 | 0.984 | 0.945 |
| dinov2_head | resize_0.75 | 0.986 | 0.947 |
| dinov2_head | resize_0.5 | 0.961 | 0.912 |
| dinov2_head | resize_0.25 | 0.584 | 0.464 |
| dinov2_head | roundtrip_0.5 | 0.972 | 0.925 |
| dinov2_head | crop_0.8 | 0.990 | 0.951 |
| dinov2_head | noise_2 | 0.986 | 0.943 |
| dinov2_head | noise_5 | 0.984 | 0.926 |
| dinov2_head | social_1080_q80 | 0.983 | 0.930 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| dinov2_head | dalle2 | only one label present; AUC undefined |
| dinov2_head | dalle3 | only one label present; AUC undefined |
| dinov2_head | firefly | only one label present; AUC undefined |
| dinov2_head | glide | only one label present; AUC undefined |
| dinov2_head | midjourney-v5 | only one label present; AUC undefined |
| dinov2_head | none | only one label present; AUC undefined |
| dinov2_head | stable-diffusion-1-3 | only one label present; AUC undefined |
| dinov2_head | stable-diffusion-1-4 | only one label present; AUC undefined |
| dinov2_head | stable-diffusion-2 | only one label present; AUC undefined |
| dinov2_head | stable-diffusion-xl | only one label present; AUC undefined |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| dinov2_head | COCO | only one label present; AUC undefined |
| dinov2_head | Synthbuster | only one label present; AUC undefined |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| dinov2_head | 42000 | 51.34 |
