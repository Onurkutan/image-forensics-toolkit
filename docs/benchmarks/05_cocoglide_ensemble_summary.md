# Localizer ensemble and per-level pixel metrics on CocoGlide

Date: 2026-09-10. Hardware: RTX 2060 (6 GB). Raw reports:
[`05_cocoglide_ensemble_mean.md`](05_cocoglide_ensemble_mean.md),
[`05_cocoglide_ensemble_max.md`](05_cocoglide_ensemble_max.md),
[`05_cocoglide_ensemble_rank_mean.md`](05_cocoglide_ensemble_rank_mean.md) (all 1,024 images,
clean level) and [`05_cocoglide_localizers_levels.md`](05_cocoglide_localizers_levels.md)
(`iml_vit` and `catnet_v2` on a seeded 512-image subset -- 255 of them with masks -- at the
nine geometry-preserving levels of the `localization` suite). The single-model clean-level
numbers come from [`03_cocoglide_catnet.md`](03_cocoglide_catnet.md) and
[`03_cocoglide_iml_vit.md`](03_cocoglide_iml_vit.md).

## Two questions

1. Does combining the two localizers' maps beat the better of them? The roadmap's 4a promised
   "a simple ensemble"; `localizer_ensemble` offers three parameter-free rules -- pixelwise
   mean, pixelwise maximum, and the mean of per-image percentile ranks.
2. How does each localizer's *map* degrade under recompression and noise? Until this run the
   benchmark runner scored pixels at the clean level only; it now records pixel metrics at
   every level whose perturbation keeps the pixel grid.

## The ensemble, clean level (512 masked images)

| Localizer | F1@0.5 | best-F1 | AP | IoU@0.5 | image AUC |
|---|---|---|---|---|---|
| `catnet_v2` alone | 0.364 | 0.605 | 0.566 | 0.288 | 0.666 |
| `iml_vit` alone | 0.059 | 0.486 | 0.423 | 0.037 | 0.535 |
| ensemble `max` | 0.385 | 0.608 | 0.565 | 0.301 | 0.670 |
| ensemble `mean` | 0.135 | 0.613 | 0.555 | 0.096 | 0.642 |
| ensemble `rank_mean` | 0.431 | 0.577 | 0.525 | 0.308 | 0.457 |

- **No parameter-free combination beats CAT-Net alone in any meaningful way.** `max` is CAT-Net
  plus two hundredths of F1@0.5 and IoU, at the same best-F1 and AP: where IML-ViT is
  confident it agrees with CAT-Net, and where it is not it adds nothing. `mean` gains 0.008
  of best-F1 and loses two thirds of the F1@0.5 (0.364 to 0.135): IML-ViT's near-zero
  probabilities pull every pixel below the threshold, which is the calibration failure the
  first 4a slice measured, now propagated into the average. `rank_mean` has the highest
  F1@0.5 (0.431) for the reason its docstring gives -- a rank map marks the upper half of
  every image at 0.5 by construction, and on CocoGlide's large edits that guess pays -- and
  the lowest best-F1 and AP, and its image score is meaningless (AUC 0.457).
- **The default is therefore `max`.** It is the one rule that preserves the better member's
  calibration at a fixed threshold, and it is what a viewer wants from "show me both": a
  region that either model marks. `mean` and `rank_mean` stay selectable with
  `IMGFORENSICS_LOCALIZER_ENSEMBLE_MODE`. A combination that *does* improve on CAT-Net needs
  a fitted per-pixel stacking, which is dataset-specific and therefore a different feature.

## Per level (255 masked images of the 512 subset)

Pixel best-F1 / F1@0.5 / AP; image-level AUC in the last column.

| Level | `catnet_v2` | AUC | `iml_vit` | AUC |
|---|---|---|---|---|
| clean | 0.598 / 0.350 / 0.569 | 0.633 | 0.474 / 0.064 / 0.406 | 0.497 |
| jpeg_q95 | 0.619 / 0.288 / 0.596 | 0.738 | 0.446 / 0.044 / 0.381 | 0.485 |
| jpeg_q85 | 0.631 / 0.313 / 0.604 | 0.724 | 0.468 / 0.058 / 0.403 | 0.498 |
| jpeg_q75 | 0.609 / 0.388 / 0.571 | 0.717 | 0.478 / 0.073 / 0.407 | 0.505 |
| jpeg_q60 | 0.614 / 0.320 / 0.577 | 0.643 | 0.485 / 0.081 / 0.410 | 0.522 |
| jpeg_q50 | 0.605 / 0.330 / 0.563 | 0.629 | 0.490 / 0.082 / 0.418 | 0.516 |
| webp_q80 | 0.559 / 0.280 / 0.514 | 0.561 | 0.451 / 0.054 / 0.383 | 0.504 |
| noise_2 | 0.592 / 0.276 / 0.565 | 0.609 | 0.475 / 0.068 / 0.406 | 0.514 |
| noise_5 | 0.516 / 0.147 / 0.470 | 0.540 | 0.481 / 0.072 / 0.414 | 0.504 |

- **A real JPEG history helps CAT-Net, and it shows.** CocoGlide is PNG; at the clean level
  CAT-Net's DCT stream reads the quality-100 re-encode this toolkit performs. At `jpeg_q85` and
  `jpeg_q95` the image has been through an ordinary JPEG encoder once, which is the setting the
  model was trained on, and both the pixel ranking (best-F1 0.598 to 0.631, AP 0.569 to 0.604)
  and the image-level AUC (0.633 to 0.738) improve. The 4a report called the clean number a
  lower bound; this is the measurement behind that claim.
- **Down to quality 50 the map holds; WEBP and heavy noise take it apart.** Best-F1 stays
  within 0.60-0.63 through `jpeg_q50`, drops to 0.559 under WEBP (a different transform, no
  DCT history to read) and to 0.516 at sigma-5 noise, where F1@0.5 falls to 0.147: the map
  still ranks pixels usefully but no longer crosses the threshold.
- **IML-ViT is flat because it was never reading the compression.** Its best-F1 sits at
  0.45-0.49 at every level and its image AUC at 0.5: what it sees on CocoGlide does not change
  with recompression, and it did not see much to begin with.
- The 255-image subset's clean-level numbers are slightly below the 512-image ones above
  (0.598 vs 0.605 best-F1), the ordinary variation of a random half; compare rows within this
  table, not across tables.

## Cost

The three ensemble runs took 13 minutes each (1,024 images, both members resident, 3.8 GB VRAM
reserved); the per-level run 53 minutes (512 images, 9 levels, two models). CAT-Net's
pure-Python JPEG decoder remains the largest single cost.
