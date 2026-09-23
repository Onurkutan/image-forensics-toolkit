# Phase 4b: `dino_inpaint`, an own inpainting localizer over frozen DINOv2

Date: 2026-09-21 (training) and 2026-09-23 (benchmarks). Hardware: RTX 2060 (6 GB). Raw
reports: [`10_tgif_test_dino_inpaint_01.md`](10_tgif_test_dino_inpaint_01.md),
[`10_cocoglide_dino_inpaint_01.md`](10_cocoglide_dino_inpaint_01.md); records in
`data/benchmarks/10_*.json`; checkpoint `weights/dino_inpaint_01` (gitignored; head weights only,
4.7 MB, `commercial_ok: false` because TGIF is CC BY-SA 4.0). Config:
`configs/experiments/4b_dino_inpaint_stage1.yaml`. Baselines quoted from
[`09_tgif_localizers_summary.md`](09_tgif_localizers_summary.md) and
[`03_cocoglide_catnet.md`](03_cocoglide_catnet.md).

## What was built

The released localizers find spliced inpaints and miss fully regenerated ones: on TGIF's
regenerated subsets `catnet_v2` scores a pixel best-F1 of 0.33 (SD2) and 0.21 (SDXL) against a
predict-everything baseline of 0.23 and 0.11, because a regenerated image has no second source
and no splice edge. `dino_inpaint` is the roadmap's answer: the frozen DINOv2 ViT-B/14
(`vit_base_patch14_dinov2.lvd142m`, Apache-2.0, the backbone `dinov2_head` already uses) read at
448 px with `dynamic_img_size`, its patch tokens from blocks 5, 8 and 11 concatenated (2,304
channels per 14 px patch), and a fully convolutional patch head on top: per-layer LayerNorm, a
1x1 projection to 256 channels, one 3x3 convolution for context, a 1x1 logit, 1.19 M parameters.
The head predicts a 32x32 grid of "this patch was generated" logits per crop; at inference the
image is tiled at 448 with stride 336 (a side below 448 shrinks the tile to the next multiple of
14 and reflection-pads the remainder, never resizes), the tiles' sigmoids are averaged where they
overlap and upsampled 14x to a pixel heatmap. Image-level score: the mean of the top 1% of the
heatmap, as for the other localizers. Registered as `dino_inpaint` (kind `localizer`); abstains
with a reason when no checkpoint is found; `IMGFORENSICS_INPAINT_DIR` selects one.

## Training (stage 1, head only)

Training data: TGIF's train split, **regenerated subsets only** (14,640 sd2-fr and 14,640
sdxl-fr, every one with its exact crop-size mask) and 4,140 authentic crops (`orig_512` and
`orig_1024`, deduplicated by sha256: the same photograph is filed under every COCO category it
contains). The spliced subsets are deliberately excluded: a splice edge is an easier cue that
does not exist in the target case, and it would make the regenerated-subset number unreadable.
Per crop: real 0.30 / sd2-fr 0.35 / sdxl-fr 0.35; half of a fake's crops are placed to contain a
uniformly drawn mask pixel (uniformly among the placements that contain it, never centred on
it); soft patch targets (the mask's area inside each 14x14 patch); BCE with `pos_weight` 3 plus
half a soft Dice; horizontal flip and a symmetric JPEG/WEBP/noise augmentation on both classes.
20,000 crops per epoch, batch 8, AdamW 3e-4, fp16 autocast with loss scaling, early stopping on
the mean best-F1 of a fixed 500-image validation subset run through the inference path.

| Epoch | train loss | val best-F1 | val AP | val F1@0.5 | reals' area above 0.5 | crops/s |
|---|---|---|---|---|---|---|
| 1 | 0.787 | 0.586 | 0.585 | 0.371 | 0.023 | 23.5 |
| **2** | 0.682 | **0.597** | **0.603** | 0.411 | 0.028 | 24.2 |
| 3 | 0.626 | 0.584 | 0.588 | 0.431 | 0.027 | 24.9 |
| 4 | 0.583 | 0.575 | 0.579 | 0.429 | 0.033 | 27.5 |
| 5 | 0.550 | 0.589 | 0.598 | 0.450 | 0.033 | 29.0 |
| 6 | 0.516 | 0.576 | 0.579 | 0.412 | 0.023 | 32.1 |

The validation metric is computed at patch resolution (14x downsampled), so it is not the
pixel number below, but its shape is the finding: the frozen feature space gives almost
everything it has in the first epoch, the training loss keeps falling while validation does
not, and early stopping keeps epoch 2. 88 minutes on the RTX 2060, the first half shared with
a benchmark run.

## Result: TGIF test sample (2,000 images, clean), pixel level, threshold 0.5

`dino_inpaint` against `catnet_v2` (the best released localizer here) and the predict-everything
baseline, per subset; best per column in bold:

| Subset | n | F1@0.5 | best-F1 | AP | IoU |
|---|---|---|---|---|---|
| sd2-fr (regenerated, 512 px) | 445 | **0.398** / 0.046 / 0.226 | **0.567** / 0.325 / 0.226 | **0.574** / 0.304 / 0.148 | **0.291** / 0.036 / 0.148 |
| sdxl-fr (regenerated, 1024 px) | 444 | **0.461** / 0.028 / 0.114 | **0.575** / 0.205 / 0.114 | **0.574** / 0.185 / 0.065 | **0.338** / 0.020 / 0.065 |
| sd2-sp (spliced) | 444 | 0.324 / **0.895** / 0.110 | 0.555 / **0.926** / 0.110 | 0.578 / **0.959** / 0.063 | 0.235 / **0.829** / 0.063 |
| ps-sp (spliced, Photoshop/Firefly) | 445 | 0.173 / **0.863** / 0.106 | 0.386 / **0.913** / 0.106 | 0.393 / **0.948** / 0.060 | 0.113 / **0.781** / 0.060 |

Each cell is `dino_inpaint` / `catnet_v2` / baseline. Pooled over the 1,778 fakes: F1@0.5 0.339,
best-F1 0.521, AP 0.530, IoU 0.244 (`catnet_v2` 0.458 / 0.592 / 0.599 / 0.416, a mean that its
spliced subsets carry).

Image level (top-1% score, each subset against the 222 reals): sd2-fr **0.867**, sdxl-fr
**0.904**, sd2-sp 0.720, ps-sp 0.613 (`catnet_v2`: 0.399, 0.344, 0.971, 0.962). Real-image
false-positive rate at 0.5: 0.338 (`catnet_v2` 0.194). 185 ms per image.

## Result: CocoGlide (1,024 images, GLIDE inpainting, a generator never in training)

| Localizer | F1@0.5 | best-F1 | AP | IoU | image AUC | ms / image |
|---|---|---|---|---|---|---|
| `dino_inpaint` | **0.425** | **0.637** | **0.672** | **0.333** | **0.768** | 20 |
| `catnet_v2` | 0.364 | 0.605 | 0.566 | 0.288 | 0.666 | 193 |
| `iml_vit` | 0.059 | 0.486 | 0.423 | 0.037 | 0.535 | - |
| predict-everything | 0.355 | 0.355 | 0.252 | 0.252 | - | - |

CocoGlide's 256x256 images are one 266 px tile each. `catnet_v2`'s image AUC is its top-1% rule
on a heatmap; both real-image false-positive rates are untuned (0.326 here).

## Reading the numbers honestly

- **The target is met, on the rules fixed before the run.** The bar was sd2-fr best-F1 at
  least 0.50 and sdxl-fr at least 0.35, F1@0.5 above the trivial baseline on both, and CocoGlide
  best-F1 at least 0.45 with AP at least 0.35. Stage 1 lands at 0.567 / 0.575, 0.398 / 0.461
  against 0.226 / 0.114, and 0.637 / 0.672. A head over a frozen self-supervised backbone can
  read where a regenerated image's content was synthesized; the released splicing localizers
  cannot, and at the image level the head separates regenerated images from the originals at
  AUC 0.87 and 0.90 where they score below chance.
- **It generalizes to a generator it never saw, and there it beats CAT-Net.** CocoGlide is
  GLIDE inpainting composited into COCO photographs, three years older than anything in TGIF;
  `dino_inpaint` wins on every pixel number and on the image AUC. That is the cross-generator
  test the roadmap asked for, and the one number that justifies calling this a localizer rather
  than a TGIF fit. TGIF's own test split shares its three generators with training, so those
  rows are in-distribution for the generator and held-out for the images.
- **It is not a splicing localizer, and it was not meant to be.** On the spliced subsets it
  trails `catnet_v2` by 0.37 and 0.53 best-F1: the splice edge is a cue it never saw in training
  and its patches are 14 px wide. The two models are complementary, which is what the
  `localizer_ensemble`'s `max` rule exists for; the next measurement is the three-member ensemble.
- **Calibration is the open weakness.** A third of the authentic images clear 0.5 somewhere in
  their heatmap (the top-1% rule punishes any confident patch), and F1@0.5 sits well below
  best-F1 on every subset. The heatmap ranks well and the fixed threshold does not read it
  well; a threshold study on the validation split, not more training, is the next step there.
- **Stage 2 (LoRA) is not needed to pass, so it is not run tonight.** The validation curve says
  the frozen space saturates in one epoch; adapting the backbone is the way past that ceiling,
  and it is a separate, priced experiment (about 3 hours at batch 4) rather than a prerequisite.
- **Licensing.** The checkpoint records `commercial_ok: false`: its training data is CC BY-SA
  4.0, and the project treats weights derived from ShareAlike data as research-only, as
  `docs/ROADMAP.md` section 7 does for derived image sets. Only the head's 1.19 M parameters are
  saved; the backbone is fetched from its own source.

## Cost

Training 88 minutes (6 epochs, 27 crops/s average, 0.8 GB VRAM). Benchmarks: TGIF sample 18
minutes, CocoGlide 1 minute. The full 9,261-image TGIF test split is recorded below when run.
