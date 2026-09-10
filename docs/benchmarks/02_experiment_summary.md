# Experiment 02: diverse real sources, augmented-only training

Date: 2026-09-10. Hardware: RTX 2060 (6 GB). Config:
[`configs/experiments/02_diverse_reals_augmented_only.yaml`](../../configs/experiments/02_diverse_reals_augmented_only.yaml).
Raw reports: [`02_val_cf.md`](02_val_cf.md), [`02_wildrf_test.md`](02_wildrf_test.md),
[`02_coco_test.md`](02_coco_test.md), [`02_cocoglide.md`](02_cocoglide.md).

## Headline

Same backbone, same head, same harness as [experiment 01](01_experiment_summary.md); only the
data changed. The failure mode of experiment 01 (every foreign image called fake) is gone: the
false-positive rate on held-out COCO photographs fell from 99.9% to 2.7%, and WildRF went from
below chance to AUC 0.980. Local diffusion edits (CocoGlide) remain out of reach for a
whole-image detector, as expected; that is the localizer's job.

| Test set | Entries | AUC | Balanced acc. at 0.5 | FPR at 0.5 | TPR at 0.5 |
|---|---|---|---|---|---|
| Community Forensics val, 42 unseen generators | 1,000 of 4,764 | 1.000 | 0.997 | 0.000 | 0.995 |
| WildRF test split, in the wild | 1,000 of 2,241 | 0.980 | 0.916 | 0.137 | 0.970 |
| COCO val2017 held-out photographs | 1,000 | n/a | n/a | 0.027 | n/a |
| CocoGlide, GLIDE inpainting vs authentic COCO | 1,024 | 0.644 | 0.565 | 0.023 | 0.154 |

Experiment 01 on the same kind of sets: WildRF AUC 0.459 and COCO FPR 0.999.

## What changed

1. **Real-image diversity.** Training reals are no longer only FFHQ, LAION and ImageNet
   crops: 4,000 COCO val2017 photographs (JPEG, mixed scenes) and the WildRF train split
   (2,712 images, both classes, platform-laundered) were merged in. Train: 25,676 entries
   (8,551 fake, 17,523 real). Val: Community Forensics val plus the WildRF val split
   (5,162 entries).
2. **Augmented-only training views.** `train_views: augmented_only` drops the clean view of
   every training image, so both classes always pass through random JPEG, WEBP, blur,
   downscale-upscale, noise and cut-out. The pristine-PNG cue that experiment 01 could
   exploit is gone.
3. **Mixed-source calibration.** The temperature fitted on the mixed val split is 0.47 (not
   the 0.05 clamp of experiment 01), ECE 0.031 to 0.005 on val, 0.070 on WildRF test.

Everything else is identical: frozen DINOv2 ViT-B/14, CLS tokens of blocks 8-11 plus the
pooled output, four 224 px grid crops, the 1.06 M-parameter head, AdamW with cosine schedule.
Best epoch 9 of 14. The learned layer weights now lean on the earliest selected block
(0.31, 0.21, 0.19, 0.15, 0.15), where experiment 01 was flat.

## Reading the numbers honestly

- **WildRF is held-out images, not a held-out distribution.** Its train split was in
  training, so 0.980 is an in-distribution number for WildRF's mix of platforms and
  generators. Per platform on the test split: Reddit 0.987, Twitter 0.980, Facebook 0.949.
  The remaining cross-dataset test (Synthbuster's nine generator families, DALL-E,
  Firefly, Midjourney and Stable Diffusion variants) is pending its download; ITW-SM is
  pending access approval.
- **COCO reals are a held-out slice of a source seen in training.** The 2.7% false-positive
  rate says the head no longer treats "photograph" as "fake"; it does not say the same for
  photographs from an unseen source. The 13.7% false-positive rate on WildRF reals is the
  more realistic figure for laundered social-media photos.
- **CocoGlide** pairs authentic COCO images with the same images locally inpainted by GLIDE.
  The head sees 15% of the edits at 0.5 and keeps the false-positive rate at 2.3%. A local
  edit of a few percent of the pixels does not move a whole-image score; this is the case
  for the localizers in Phase 4 (`iml_vit`, `catnet_v2`), whose CocoGlide results are in
  [`03_cocoglide_iml_vit.md`](03_cocoglide_iml_vit.md).
- The same-family validation AUC (1.000) remains a sanity check only.

## Cost

Feature extraction for the 6,712 new images took 52 minutes, far slower than the 18 images/s
of experiment 01, because the augmented view is computed on the full-resolution image (some
WildRF images exceed 4,000 px) before cropping, and that runs on one CPU core. Training took
under two minutes; the three benchmarks about four minutes with the head alone.

## Next

- Run the Synthbuster and (when approved) ITW-SM tests as genuine cross-dataset numbers.
- Move augmentation onto crops rather than full images, or into worker processes, so
  extraction is bounded by the GPU again.
- Feed these records to the fusion layer together with the classical signals and the
  localizers.
