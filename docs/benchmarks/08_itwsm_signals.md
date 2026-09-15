# Benchmark report: ITW-SM

- Entries evaluated: 2000
- Robustness levels: clean
- Fixed threshold: 0.5
- Threshold caveat: threshold tuned in-sample

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| c2pa | fixed | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | 0.250 |
| c2pa | tuned | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | 0.250 |
| copy_move | fixed | 0.493 | 0.499 | 0.493 | 0.493 | 0.052 | 0.038 | 0.061 | 0.254 |
| copy_move | tuned | 0.493 | 0.499 | 0.502 | 0.502 | 0.005 | 0.009 | 0.061 | 0.254 |
| double_jpeg | fixed | 0.495 | 0.497 | 0.495 | 0.495 | 0.035 | 0.025 | 0.106 | 0.262 |
| double_jpeg | tuned | 0.495 | 0.497 | 0.500 | 0.500 | 0.000 | 0.000 | 0.106 | 0.262 |
| ela | fixed | 0.616 | 0.584 | 0.580 | 0.579 | 0.665 | 0.824 | 0.076 | 0.246 |
| ela | tuned | 0.616 | 0.584 | 0.601 | 0.601 | 0.382 | 0.584 | 0.076 | 0.246 |
| jpeg_ghost | fixed | 0.679 | 0.624 | 0.540 | 0.540 | 0.896 | 0.976 | 0.164 | 0.263 |
| jpeg_ghost | tuned | 0.679 | 0.624 | 0.648 | 0.648 | 0.516 | 0.812 | 0.164 | 0.263 |
| metadata | fixed | 0.583 | 0.551 | 0.583 | 0.583 | 0.308 | 0.474 | 0.078 | 0.249 |
| metadata | tuned | 0.583 | 0.551 | 0.583 | 0.583 | 0.308 | 0.474 | 0.078 | 0.249 |
| sd_watermark | fixed | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.050 | 0.253 |
| sd_watermark | tuned | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.050 | 0.253 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| c2pa | clean | 0.500 | 0.500 |
| copy_move | clean | 0.493 | 0.493 |
| double_jpeg | clean | 0.495 | 0.495 |
| ela | clean | 0.616 | 0.579 |
| jpeg_ghost | clean | 0.679 | 0.540 |
| metadata | clean | 0.583 | 0.583 |
| sd_watermark | clean | 0.500 | 0.500 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| c2pa | facebook | only one label present; AUC undefined |
| c2pa | instagram | only one label present; AUC undefined |
| c2pa | linkedin | only one label present; AUC undefined |
| c2pa | none | only one label present; AUC undefined |
| c2pa | x | only one label present; AUC undefined |
| copy_move | facebook | only one label present; AUC undefined |
| copy_move | instagram | only one label present; AUC undefined |
| copy_move | linkedin | only one label present; AUC undefined |
| copy_move | none | only one label present; AUC undefined |
| copy_move | x | only one label present; AUC undefined |
| double_jpeg | facebook | only one label present; AUC undefined |
| double_jpeg | instagram | only one label present; AUC undefined |
| double_jpeg | linkedin | only one label present; AUC undefined |
| double_jpeg | none | only one label present; AUC undefined |
| double_jpeg | x | only one label present; AUC undefined |
| ela | facebook | only one label present; AUC undefined |
| ela | instagram | only one label present; AUC undefined |
| ela | linkedin | only one label present; AUC undefined |
| ela | none | only one label present; AUC undefined |
| ela | x | only one label present; AUC undefined |
| jpeg_ghost | facebook | only one label present; AUC undefined |
| jpeg_ghost | instagram | only one label present; AUC undefined |
| jpeg_ghost | linkedin | only one label present; AUC undefined |
| jpeg_ghost | none | only one label present; AUC undefined |
| jpeg_ghost | x | only one label present; AUC undefined |
| metadata | facebook | only one label present; AUC undefined |
| metadata | instagram | only one label present; AUC undefined |
| metadata | linkedin | only one label present; AUC undefined |
| metadata | none | only one label present; AUC undefined |
| metadata | x | only one label present; AUC undefined |
| sd_watermark | facebook | only one label present; AUC undefined |
| sd_watermark | instagram | only one label present; AUC undefined |
| sd_watermark | linkedin | only one label present; AUC undefined |
| sd_watermark | none | only one label present; AUC undefined |
| sd_watermark | x | only one label present; AUC undefined |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| c2pa | ITW-SM | 0.500 |
| copy_move | ITW-SM | 0.493 |
| double_jpeg | ITW-SM | 0.495 |
| ela | ITW-SM | 0.616 |
| jpeg_ghost | ITW-SM | 0.679 |
| metadata | ITW-SM | 0.583 |
| sd_watermark | ITW-SM | 0.500 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| c2pa | 2000 | 2.11 |
| copy_move | 2000 | 290.13 |
| double_jpeg | 2000 | 239.15 |
| ela | 2000 | 214.54 |
| jpeg_ghost | 2000 | 540.92 |
| metadata | 2000 | 0.54 |
| sd_watermark | 2000 | 599.48 |
