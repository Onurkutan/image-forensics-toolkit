# Experiment 01: DINOv2 head on Community Forensics Small

Date: 2026-09-10. Hardware: RTX 2060 (6 GB). Runbook:
[`docs/experiments/01_dinov2_head_community_forensics.md`](../experiments/01_dinov2_head_community_forensics.md).
Raw reports: [`01_val_dinov2.md`](01_val_dinov2.md), [`01_val_signals.md`](01_val_signals.md),
[`01_wildrf.md`](01_wildrf.md), [`01_coco_reals.md`](01_coco_reals.md).

## Headline

The head separates held-out generators of its own training family perfectly and then calls
almost every image from any other source fake. It learned what its two real-image sources
look like, not what a real photograph looks like. This is a negative result, and the
evaluation harness reported it exactly as designed: a held-out-generator split alone would
have declared victory.

| Test set | Entries | AUC | Balanced acc. at 0.5 | Fraction called fake |
|---|---|---|---|---|
| Val, 42 unseen generators, same dataset family | 1,000 of 4,764 | 1.000 | 1.000 | fakes 1.00, reals 0.00 |
| WildRF, in the wild, mixed generators and platforms | 1,000 of 5,351 | 0.459 | 0.500 | fakes 0.96, reals 0.96 |
| COCO val2017, real photographs only | 1,000 of 5,000 | n/a | n/a | reals 0.999 |

## Setup

- Data: Community Forensics Small, first 8 shards (23 GB, 23,939 rows, 211 NSFW rows skipped),
  8,768 fakes from 227 generators at 512x512 and 14,960 reals (FFHQ, LAION, ImageNet) at
  1024x1024, all PNG. The bias audit flagged the resolution gap (Jensen-Shannon distance 1.0);
  reals were center-cropped to 512x512 at native resolution with `manifest crop`, after which
  the audit passed (same format, same resolution bucket, no duplicates).
- Split: generator-disjoint. Train 6,996 fakes / 11,968 reals (185 generators); val 1,772 fakes /
  2,992 reals (42 generators never seen in training).
- Features: frozen DINOv2 ViT-B/14, CLS tokens of blocks 8-11 plus the pooled output, four
  224 px grid crops per image, two views per training image (clean plus one augmented view
  from `configs/augment_default.yaml`). 37,928 training arrays in 37 minutes.
- Head: `MultiLayerHead`, 1.06 M parameters, AdamW, cosine schedule, label smoothing 0.05,
  class-balanced sampling. Best epoch 4 of 9 (early stopping); training took under a minute.
- Calibration on val: ECE 0.028 to 0.0003. The fitted temperature hit the lower clamp (0.05),
  which is what a perfectly separable validation split produces.

## What the numbers say

**Validation (same dataset family).** AUC 1.000 at every robustness level except two:
`resize_0.5` 0.998 and `resize_0.25` 0.567. At a quarter scale a 512 px image becomes 128 px,
smaller than the 224 px crop window, and the crop is mostly reflection padding. That is a real
limit of the crop-never-resize policy for small inputs, not of the head.

**WildRF (in the wild).** AUC 0.459, below chance. The head assigns a score above 0.5 to 95.7%
of the fakes and 95.7% of the reals; the mean score is 0.94 for fakes and 0.95 for reals. The
detector is not wrong about fakes, it is wrong about everything.

**COCO real photographs.** 99.9% false-positive rate at 0.5, mean score 0.998, median 1.0.
Ordinary JPEG photographs are the strongest "fake" signal the head knows.

**Classical signals.** On the clean-PNG validation set, `metadata`, `c2pa`, `sd_watermark`,
`copy_move` and `double_jpeg` abstain at 0.5, as expected for pristine files. `ela` (0.30) and
`jpeg_ghost` (0.38) run inverted: the real photographs carry residual JPEG history from before
they were stored as PNG, the generated images never were JPEG. On WildRF the picture changes:
`double_jpeg` reaches AUC 0.72 and `ela` 0.58, `jpeg_ghost` stays inverted at 0.33, the rest
abstain. Re-shared fakes on social platforms appear to carry more compression history than the
reals around them; this is a usable, dataset-dependent signal for the fusion layer, not a
detector.

**Cost.** Per 512 px image: DINOv2 head 43 ms, all seven signals about 270 ms on one CPU core.
On WildRF's larger images the signals average 2.2 s per image, `sd_watermark` alone 0.9 s,
because it must decode at native resolution. The GPU sat mostly idle during the benchmarks;
the signals are the bottleneck and are not yet parallelized.

## Diagnosis

The training reals come from three curated sources (FFHQ portraits, LAION, ImageNet) and every
training fake is a prompt-driven render. DINOv2 features are strongly semantic, so a head can
separate the two classes by content and by source style without touching any generation
artifact. Three cues are available and none of them survives a change of real-image source:

1. Content and style of the real sources versus prompt-driven renders.
2. Scene scale: a 224 px crop of a 1024 px photograph covers a smaller part of the scene than
   a 224 px crop of a 512 px render, even after center-cropping the photograph to 512.
3. Compression history: the reals were JPEG once, the fakes never; the clean view of every
   training image keeps that difference visible, and the inverted `ela` result confirms it.

The bias audit cannot see semantic or source bias, only format, resolution, quality and
duplicates. The held-out-generator split cannot see it either, because held-out generators
share the real-image sources with the training generators. Only a foreign test set can, which
is why WildRF and COCO are in the protocol.

## What this changes

Experiment 02 will keep the backbone, the head and the harness, and change the data:

- Real-image diversity: add COCO train photographs (JPEG, mixed scenes) and the WildRF train
  split (both classes, in the wild, platform-laundered), so "real" is no longer three curated
  sources. WildRF test stays untouched as the foreign test set.
- Train on augmented views only: drop the clean view from the training set so both classes
  always pass through random JPEG, WEBP, blur and noise, removing the pristine-PNG cue.
- Report the COCO false-positive rate and the WildRF AUC as the primary numbers; the
  same-family validation AUC is a sanity check, not a result.
- Calibrate on a mixed-source validation split, so the temperature is fitted on scores that are
  not perfectly separable.

Independently of the data changes: parallelize the classical signals across CPU cores and
feed feature extraction with worker processes, so a benchmark is bounded by the GPU rather
than by one core.

## Reproduction

The runbook lists every command. Feature cache and weights are not committed; the trained head
(`weights/dinov2_head/head.json`) records the training manifests' sha256, the augmentation
hash, the calibration and the research-only license inherited from Community Forensics.
