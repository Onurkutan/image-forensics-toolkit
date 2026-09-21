# Experiment 04, step 0: does simulated laundering explain the real-class false positives?

Date: 2026-09-21. Hardware: RTX 2060 (6 GB). Raw report:
[`09_exp03_laundering_probe.md`](09_exp03_laundering_probe.md) (records in
`data/benchmarks/09_exp03_laundering_probe.json`). Suite: `configs/robustness_laundering_probe.yaml`.

## The question

Experiments 02 and 03 left one failure standing: the head calls platform-laundered real
photographs fake (WildRF cross-dataset FPR 0.547 at 0.5, ITW-SM 0.442 with the shipped head and
0.585 with the experiment 03 head), while held-out COCO photographs stay at 2.6%. The planned
experiment 04 was to add a social-media laundering simulation -- downscale, then JPEG -- to the
training views of both classes so the head stops reading platform re-encoding as evidence of
generation. That plan assumes *hypothesis A*: the head reads the re-encoding. The alternative,
*hypothesis B*, is that it reads the content and source distribution of social-media photographs,
which COCO does not cover. Before spending a night of GPU on A, two cheap checks.

## Check 1: launder COCO photographs and watch the false-positive rate

The experiment 03 head (no WildRF in training) on the 1,000 held-out COCO photographs of
`coco_test.jsonl` (640 px long side), threshold 0.5:

| Level | What it does | FPR at 0.5 |
|---|---|---|
| clean | nothing | 0.026 |
| social_1080_q80 | JPEG q80 only (no resize below 1080) | 0.031 |
| jpeg_q75 | JPEG q75 only | 0.036 |
| launder_s0.7_q80 | resize to 448 px (scale 0.7), then JPEG q80 | 0.047 |
| launder_s0.5_q80 | resize to 320 px (scale 0.5), then JPEG q80 | 0.089 |
| resize_0.5 | resize to 320 px only, no JPEG | 0.103 |

Laundering the photographs the head already knows raises its false-positive rate from 2.6% to
at most 8.9%, and the JPEG step contributes almost nothing: the movement comes from the
half-scale resize alone, which the training augmentation's `downscale_upscale` covers only in
part. Six points of a fifty-point gap.

## Check 2: which ITW-SM photographs does the head get wrong?

Joining the saved ITW-SM records (`08_itwsm_dinov2_head_02.json`, `08_itwsm_dinov2_head_03.json`)
with the manifest's stored size and JPEG quality, real class only, FPR at 0.5:

| Long side | n | shipped head (exp02) | exp03 head |
|---|---|---|---|
| < 800 px | 133 | 0.323 | 0.474 |
| 800-1200 | 1,027 | 0.442 | 0.555 |
| 1200-1600 | 1,885 | 0.474 | 0.596 |
| 1600-2600 | 1,639 | 0.437 | 0.606 |
| >= 2600 | 316 | 0.335 | 0.557 |

| Stored JPEG quality | n | shipped head (exp02) | exp03 head |
|---|---|---|---|
| < 75 | 425 | 0.362 | 0.560 |
| 75-84 | 1,944 | 0.419 | 0.581 |
| 85-91 | 1,183 | 0.413 | 0.555 |
| >= 92 | 1,413 | 0.519 | 0.623 |

If the cue were "heavily re-encoded, therefore generated", the false-positive rate would rise as
quality falls and size shrinks. It does the opposite on both heads: the *least* compressed
photographs (quality 92 and above) are the ones called fake most often, and the smallest images
the least. The head is not reading the platform's compression.

## Reading

Hypothesis A is rejected as the main cause. Simulated laundering can explain a few points of the
real-class false-positive rate, not the fifty it would have to; and the images the head gets wrong
in the wild are the clean, large ones. What separates ITW-SM's and WildRF's real photographs from
COCO's is therefore what they are pictures *of* and where they come from -- phone photographs,
screenshots, graphics, portraits and re-posts that a COCO-trained real class has never seen --
which is what experiment 02 already showed from the other side: adding 2,712 WildRF-train reals
moved ITW-SM from 0.819 to 0.888 without any change to the model.

Experiment 04 as designed (a `social` augmentation on the training views, about six hours of
extraction, training and benchmarks) is therefore not run. The next experiment on this failure
needs a **real** in-the-wild real-image source in training, not a simulation of one, and it must
stay disjoint from ITW-SM and WildRF-test so the cross-dataset tests remain honest. One caveat on
check 1: the crop policy caps the simulated downscale at 0.5 of a 640 px image, while in-the-wild
uploads are often reduced further, so "mild laundering does not explain it" is the exact claim.

## Cost

Design and the two checks: about an hour, of which the GPU part (6,000 scored images) was five
minutes. The full experiment would have cost six hours.
