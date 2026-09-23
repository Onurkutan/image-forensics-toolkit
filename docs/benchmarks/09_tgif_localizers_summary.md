# TGIF: the pretrained localizers on text-guided inpainting, spliced and fully regenerated

Date: 2026-09-21. Hardware: RTX 2060 (6 GB). Raw reports:
[`09_tgif_test_catnet_v2.md`](09_tgif_test_catnet_v2.md),
[`09_tgif_test_iml_vit.md`](09_tgif_test_iml_vit.md),
[`09_tgif_test_localizer_ensemble.md`](09_tgif_test_localizer_ensemble.md); records in
`data/benchmarks/09_tgif_test_*.json`. Manifest: `tgif_test_2000.jsonl`, a stratified sample
(seed 0, label x generator) of the 9,261-image test split of `tgif.jsonl`.

## Setup

TGIF (Mareen et al., WIFS 2024; CC BY-SA 4.0) inpaints 3,124 COCO photographs with text-guided
diffusion and Photoshop generative fill, two mask types per photograph (a bounding box and a
segmentation mask) and three variations per generation. Four sub-datasets: **sd2-sp** and
**ps-sp** composite the inpainted region back into the full-size original ("spliced"), so only
the masked pixels change; **sd2-fr** (512 px) and **sdxl-fr** (1024 px) keep the generator's
whole output ("fully regenerated"), so every pixel has passed through the model's decoder and
the mask marks only where the content was changed. Ground truth follows the mask each pipeline
actually used: the bordered `ps_mask` for the spliced sets, the crop-size mask for the
regenerated ones (`src/imgforensics/data/layouts.yaml`, TGIF entry, records the decision).
Reals are the three stored variants of each original (full, 1024 crop, 512 crop), so each fake
subset has a real counterpart at its own size. Sample: 222 reals, 444-445 fakes per subset,
mask types balanced (915 bbox, 863 segm). All files are PNG, so `catnet_v2` reads a quality-100
re-encode of its own making, exactly as on CocoGlide. Nothing from TGIF was used to train or
tune anything here; the two localizers are the released CASIA/tampCOCO-era checkpoints.

## Result: pixel level, clean, threshold 0.5 (mean over the 1,778 fakes)

| Localizer | F1@0.5 | best-F1 | AP | IoU |
|---|---|---|---|---|
| `catnet_v2` | **0.458** | **0.592** | **0.599** | **0.416** |
| `iml_vit` | 0.062 | 0.273 | 0.241 | 0.046 |
| `localizer_ensemble` (`max`) | 0.439 | 0.580 | 0.571 | 0.388 |
| predict-everything baseline | 0.139 | 0.139 | 0.084 | 0.084 |

The mean hides two different datasets. Per subset, `catnet_v2` (ensemble in parentheses):

| Subset | n | F1@0.5 | best-F1 | AP | IoU | baseline F1 / AP |
|---|---|---|---|---|---|---|
| sd2-sp (spliced, SD2) | 444 | **0.895** (0.839) | **0.926** (0.891) | **0.959** (0.925) | **0.829** (0.758) | 0.110 / 0.063 |
| ps-sp (spliced, Photoshop/Firefly) | 445 | **0.863** (0.819) | **0.913** (0.877) | **0.948** (0.912) | **0.781** (0.726) | 0.106 / 0.060 |
| sd2-fr (fully regenerated, 512 px) | 445 | 0.046 (0.059) | 0.325 (0.343) | 0.304 (0.290) | 0.036 (0.042) | 0.226 / 0.148 |
| sdxl-fr (fully regenerated, 1024 px) | 444 | 0.028 (0.039) | 0.205 (0.207) | 0.185 (0.156) | 0.020 (0.027) | 0.114 / 0.065 |

`iml_vit` per subset: best-F1 0.290 / 0.334 / 0.302 / 0.167, F1@0.5 0.11 / 0.11 / 0.02 / 0.02.
Mask type barely matters: `catnet_v2` best-F1 0.609 on bbox masks, 0.574 on segm. The
predict-everything baseline is computed from the masks themselves: F1 is `2p/(1+p)` and AP and
IoU are `p`, the masked fraction of the image, averaged over the same fakes.

## Result: image level (top-1% heatmap score), each fake subset against the 222 reals

| Localizer | sd2-sp | ps-sp | sd2-fr | sdxl-fr | FPR on reals at 0.5 |
|---|---|---|---|---|---|
| `catnet_v2` | **0.971** | **0.962** | 0.399 | 0.344 | 0.194 |
| `iml_vit` | 0.615 | 0.592 | 0.495 | 0.424 | 0.203 |
| `localizer_ensemble` | 0.948 | 0.935 | 0.455 | 0.373 | 0.410 |

The pooled AUC in the raw reports (0.669 / 0.531 / 0.678) is the average of these two worlds.

## Result: the full test split, `catnet_v2` only (9,261 images, clean)

Run afterwards as the record for Phase 4b ([`09_tgif_test_full_catnet_v2.md`](09_tgif_test_full_catnet_v2.md),
433 ms per image): 1,029 reals and 2,058 fakes per subset. The 2,000-image sample above was
representative to within 0.02 on every number.

| Subset | F1@0.5 | best-F1 | AP | IoU | image AUC vs the 1,029 reals |
|---|---|---|---|---|---|
| sd2-sp | 0.891 | **0.927** | 0.960 | 0.823 | 0.959 |
| ps-sp | 0.863 | **0.910** | 0.945 | 0.783 | 0.956 |
| sd2-fr | 0.044 | 0.340 | 0.321 | 0.031 | 0.394 |
| sdxl-fr | 0.028 | 0.200 | 0.179 | 0.019 | 0.331 |

Pooled: pixel F1@0.5 0.457 / best-F1 0.594 / AP 0.601 / IoU 0.414 over the 8,232 fakes, image AUC
0.660, real-image FPR 0.242 at 0.5. These are the baselines the 4b localizer is measured against.

## Reading the numbers honestly

- **A composited inpaint is a splice, and CAT-Net finds it.** On the two spliced subsets the
  released CAT-Net v2 reaches pixel best-F1 0.93 and 0.91 and image AUC 0.97 and 0.96 with no
  adaptation at all, on a generator family and a dataset it never saw, at eight times the
  predict-everything baseline. The composite leaves the same kind of boundary and noise
  discontinuity a classic splice does, which is what the model was trained on. This is the
  strongest localization result in the project so far, and it is a lower bound: the DCT stream
  reads a quality-100 re-encode, not a real compression history.
- **A fully regenerated image is not a splice, and nothing here finds it.** When the whole image
  has been through the generator's decoder there is no second source to detect: best-F1 falls
  to 0.33 (SD2) and 0.21 (SDXL), a few points above the predict-everything baseline (0.23 and
  0.11), F1 at the fixed threshold is near zero, and the image-level AUC is *below* 0.5 on both:
  the regenerated images look more uniform to the localizer than the PNG originals do. This is
  the CocoGlide gap measured on its own dataset, and it is exactly the case the roadmap's 4b
  localizer exists for: an inpainting localizer has to read the generator's fingerprint inside
  the masked region, not a discontinuity at its edge.
- **The per-subset split is the number to carry, not the mean.** A single "TGIF best-F1 0.59"
  would describe neither half. The same holds for the pooled image AUC.
- **`iml_vit` and the ensemble add nothing here either.** IML-ViT ranks pixels a little above
  chance and never crosses 0.5, as on CocoGlide; the `max` ensemble inherits its false positives
  (real-image FPR 0.19 to 0.41) and loses two to five points everywhere. `catnet_v2` alone stays
  the recommendation among the released models. (The ensemble measured here had two members;
  since 2026-09-23 it also includes this project's `dino_inpaint`, and the three-member numbers
  are in [`10_dino_inpaint_summary.md`](10_dino_inpaint_summary.md).)
- **Real-image false positives are not size-driven.** `catnet_v2` calls 20% of the full-size
  originals, 18% of the 1024 crops and 20% of the 512 crops fake at 0.5; the top-1% rule is a
  heatmap statistic, not a detector, and the threshold is untuned.

## Cost

Manifest with sha256 and audit over 84,348 files: 14 minutes. The three runs on 2,000 images:
`catnet_v2` 24 minutes (367 ms per image with the new `jpeglib` coefficient reader, of which the
quality-100 re-encode and its decode are now a small part), `iml_vit` 22 minutes (364 ms),
`localizer_ensemble` 33 minutes (691 ms). The full 9,261-image test split with `catnet_v2` took
67 minutes (433 ms per image; the second half overlapped with a training run on the same GPU).
The localization robustness suite on this set is still to be run.
