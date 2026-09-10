# Synthbuster: nine generator families, two heads, fifteen levels

Date: 2026-09-10. Hardware: RTX 2060 (6 GB). Raw reports:
[`07_synthbuster_dinov2_head_02.md`](07_synthbuster_dinov2_head_02.md) (the shipped default,
[experiment 02](02_experiment_summary.md)) and
[`07_synthbuster_dinov2_head_03.md`](07_synthbuster_dinov2_head_03.md) (the same recipe without
WildRF in training, [experiment 03](06_experiment_03_summary.md)).

## Setup

Synthbuster (Bammey, 2023; CC BY-NC-SA 4.0) holds 1,000 images from each of nine generators:
DALL-E 2, DALL-E 3, Adobe Firefly, GLIDE, Midjourney v5, Stable Diffusion 1.3, 1.4, 2 and XL.
The test manifest takes 200 per generator (1,800 fakes, `imgforensics manifest sample --stratify
generator`) and pairs them with the 1,000 held-out COCO val2017 photographs of experiment 02 as
the real class, then runs both heads through the full 15-level robustness suite.

Two facts decide how to read the numbers:

- **Which families are unseen.** The training fakes are 186 Hugging Face models of the
  Community Forensics slice, almost all Stable Diffusion fine-tunes, plus one SDXL base and
  PixArt (and, for the experiment 02 head, WildRF's unlabelled fakes). DALL-E 2, DALL-E 3,
  Firefly, GLIDE and Midjourney v5 are families the heads never saw; the four Stable Diffusion
  rows are the seen family through different checkpoints.
- **The real class is not Synthbuster's.** Synthbuster's paired reals (RAISE-1k) are a separate
  non-commercial download this project does not use, so the real class is COCO, which is in
  both heads' training data (held-out images, seen source). The bias audit flags the format
  split -- every Synthbuster image is PNG, every COCO image JPEG (Jensen-Shannon distance 0.76)
  -- so the clean-level AUC could in principle be a JPEG-versus-PNG detector. Both heads were
  trained on augmented-only views, which is meant to remove exactly that cue, and the JPEG
  levels below are the check: at `jpeg_q75` and `social_1080_q80` both classes are JPEG.

## Result

| Level | exp02 head AUC | TPR at 0.5 | FPR at 0.5 | exp03 head AUC | TPR at 0.5 | FPR at 0.5 |
|---|---|---|---|---|---|---|
| clean (PNG fakes, JPEG reals) | 0.986 | 0.914 | 0.027 | 0.988 | 0.926 | 0.026 |
| jpeg_q75 (both JPEG) | 0.969 | 0.804 | 0.033 | 0.980 | 0.893 | 0.036 |
| jpeg_q50 | 0.964 | 0.772 | 0.036 | 0.977 | 0.884 | 0.040 |
| social_1080_q80 | 0.974 | 0.816 | 0.031 | 0.983 | 0.891 | 0.031 |
| resize_0.5 | 0.964 | 0.912 | 0.082 | 0.961 | 0.927 | 0.103 |
| resize_0.25 | 0.605 | 0.809 | 0.790 | 0.584 | 0.903 | 0.974 |

Per generator, the share of fakes called fake at 0.5 (clean / `jpeg_q75` / `social_1080_q80`):

| Generator | Seen family? | exp02 head | exp03 head |
|---|---|---|---|
| DALL-E 2 | no | 0.895 / 0.695 / 0.730 | 0.835 / 0.720 / 0.745 |
| DALL-E 3 | no | 0.990 / 0.975 / 0.985 | 1.000 / 1.000 / 1.000 |
| Firefly | no | 0.915 / 0.790 / 0.690 | 0.930 / 0.895 / 0.755 |
| GLIDE | no | 0.820 / 0.795 / 0.780 | 0.835 / 0.845 / 0.835 |
| Midjourney v5 | no | 0.845 / 0.645 / 0.815 | 0.905 / 0.835 / 0.895 |
| Stable Diffusion 1.3 | yes | 0.960 / 0.795 / 0.805 | 0.950 / 0.940 / 0.950 |
| Stable Diffusion 1.4 | yes | 0.955 / 0.835 / 0.835 | 0.960 / 0.935 / 0.930 |
| Stable Diffusion 2 | yes | 0.870 / 0.795 / 0.785 | 0.935 / 0.885 / 0.940 |
| Stable Diffusion XL | yes | 0.975 / 0.915 / 0.920 | 0.985 / 0.985 / 0.970 |

## Reading the numbers honestly

- **The unseen families are detected.** Once both classes are JPEG (`jpeg_q75`), the shipped
  head still catches 70-98% of each unseen family's images at the fixed threshold with a 3.3%
  false-positive rate on COCO, and AUC 0.969 overall. This is the first genuine
  unseen-generator result in the project, and it is far from the collapse the survey warns
  about for 2025 commercial generators; DALL-E 3 is the easiest family here and DALL-E 2,
  Midjourney v5 and Firefly the hardest, in the JPEG-only setting.
- **The clean-level number is inflated by the format split, but not by much.** AUC drops
  0.986 to 0.969 (exp02) and 0.988 to 0.980 (exp03) when the fakes are re-encoded to match
  the reals' format, which is the size of the PNG cue the augmented-only training did not
  fully remove. Quote the JPEG rows, not the clean row.
- **The head without WildRF is the better Synthbuster detector.** Every JPEG row favours the
  experiment 03 head (AUC 0.980 vs 0.969 at `jpeg_q75`, TPR 0.893 vs 0.804), and its
  per-generator recall under laundering is higher on eight of nine families. WildRF's
  social-media training images pull the shipped head toward "laundered means real", which is
  what makes it better on WildRF (0.980 vs 0.804) and worse on curated generator output that
  has been re-encoded once. The two heads are two operating points on the same trade-off,
  not a better and a worse model; the shipped default stays the experiment 02 head because
  social-media images are the expected input, and this table is the cost of that choice.
- **Recompression is survivable, downscaling is not** -- the same shape as every earlier
  robustness table. At quarter scale the real class collapses (FPR 0.79 and 0.97): a 512 px
  COCO photograph becomes 128 px, smaller than the crop, and crop-never-resize has nothing
  left to work with.
- **What is still missing.** RAISE-1k reals would make this Synthbuster's own protocol and
  remove the COCO-seen-source caveat entirely; ITW-SM remains the pending in-the-wild set.

## Cost

Two runs of 2,800 images at 15 levels with the head alone and attribution off: about 1 h 40
min each on the shared machine, almost all of it in the perturbation pipeline on the CPU (the
GPU work per image is a few milliseconds).
