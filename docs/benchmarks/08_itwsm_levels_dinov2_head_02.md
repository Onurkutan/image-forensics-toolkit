# Benchmark report: ITW-SM

- Entries evaluated: 2000
- Robustness levels: clean, jpeg_q95, jpeg_q85, jpeg_q75, jpeg_q60, jpeg_q50, webp_q80, resize_0.75, resize_0.5, resize_0.25, roundtrip_0.5, crop_0.8, noise_2, noise_5, social_1080_q80
- Fixed threshold: 0.5
- Threshold caveat: threshold tuned in-sample

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| dinov2_head | fixed | 0.890 | 0.896 | 0.747 | 0.746 | 0.457 | 0.950 | 0.195 | 0.188 |
| dinov2_head | tuned | 0.890 | 0.896 | 0.804 | 0.804 | 0.159 | 0.768 | 0.195 | 0.188 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| dinov2_head | clean | 0.890 | 0.746 |
| dinov2_head | jpeg_q95 | 0.892 | 0.754 |
| dinov2_head | jpeg_q85 | 0.896 | 0.766 |
| dinov2_head | jpeg_q75 | 0.899 | 0.774 |
| dinov2_head | jpeg_q60 | 0.896 | 0.786 |
| dinov2_head | jpeg_q50 | 0.893 | 0.774 |
| dinov2_head | webp_q80 | 0.886 | 0.758 |
| dinov2_head | resize_0.75 | 0.892 | 0.694 |
| dinov2_head | resize_0.5 | 0.888 | 0.677 |
| dinov2_head | resize_0.25 | 0.838 | 0.671 |
| dinov2_head | roundtrip_0.5 | 0.893 | 0.726 |
| dinov2_head | crop_0.8 | 0.897 | 0.747 |
| dinov2_head | noise_2 | 0.893 | 0.762 |
| dinov2_head | noise_5 | 0.894 | 0.782 |
| dinov2_head | social_1080_q80 | 0.874 | 0.718 |

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
| dinov2_head | ITW-SM | 0.890 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| dinov2_head | 30000 | 57.07 |
