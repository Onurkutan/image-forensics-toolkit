# Benchmark report: ITW-SM

- Entries evaluated: 2000
- Robustness levels: clean, jpeg_q95, jpeg_q85, jpeg_q75, jpeg_q60, jpeg_q50, webp_q80, resize_0.75, resize_0.5, resize_0.25, roundtrip_0.5, crop_0.8, noise_2, noise_5, social_1080_q80
- Fixed threshold: 0.5
- Threshold caveat: threshold tuned in-sample

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| dinov2_head | fixed | 0.817 | 0.821 | 0.665 | 0.665 | 0.600 | 0.931 | 0.248 | 0.254 |
| dinov2_head | tuned | 0.817 | 0.821 | 0.741 | 0.740 | 0.219 | 0.700 | 0.248 | 0.254 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| dinov2_head | clean | 0.817 | 0.665 |
| dinov2_head | jpeg_q95 | 0.814 | 0.657 |
| dinov2_head | jpeg_q85 | 0.812 | 0.661 |
| dinov2_head | jpeg_q75 | 0.812 | 0.675 |
| dinov2_head | jpeg_q60 | 0.805 | 0.669 |
| dinov2_head | jpeg_q50 | 0.796 | 0.651 |
| dinov2_head | webp_q80 | 0.798 | 0.647 |
| dinov2_head | resize_0.75 | 0.840 | 0.647 |
| dinov2_head | resize_0.5 | 0.843 | 0.653 |
| dinov2_head | resize_0.25 | 0.784 | 0.649 |
| dinov2_head | roundtrip_0.5 | 0.805 | 0.653 |
| dinov2_head | crop_0.8 | 0.830 | 0.663 |
| dinov2_head | noise_2 | 0.818 | 0.678 |
| dinov2_head | noise_5 | 0.811 | 0.695 |
| dinov2_head | social_1080_q80 | 0.800 | 0.644 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| dinov2_head | facebook | only one label present; AUC undefined |
| dinov2_head | instagram | only one label present; AUC undefined |
| dinov2_head | linkedin | only one label present; AUC undefined |
| dinov2_head | none | only one label present; AUC undefined |
| dinov2_head | x | only one label present; AUC undefined |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| dinov2_head | ITW-SM | 0.817 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| dinov2_head | 30000 | 56.43 |
