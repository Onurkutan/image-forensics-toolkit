# ITW-SM: the in-the-wild test, two heads, fifteen levels, and the fuser applied cold

Date: 2026-09-15. Hardware: RTX 2060 (6 GB). Raw reports:
[`08_itwsm_dinov2_head_02.md`](08_itwsm_dinov2_head_02.md) and
[`08_itwsm_dinov2_head_03.md`](08_itwsm_dinov2_head_03.md) (every image, clean level),
[`08_itwsm_levels_dinov2_head_02.md`](08_itwsm_levels_dinov2_head_02.md) and
[`08_itwsm_levels_dinov2_head_03.md`](08_itwsm_levels_dinov2_head_03.md) (a 2,000-image
sample at all 15 levels), [`08_itwsm_signals.md`](08_itwsm_signals.md) (the seven signals on
that sample), [`08_itwsm_fusion_wildrf_fuser.md`](08_itwsm_fusion_wildrf_fuser.md) and
[`08_itwsm_fusion_levels_fuser.md`](08_itwsm_fusion_levels_fuser.md) (the two WildRF-fitted
fusers applied to the sample without refitting).

## Setup

ITW-SM (Konstantinidou et al., 2026; CC BY-SA 4.0, gated) holds 10,000 images collected from
four platforms -- Facebook, Instagram, LinkedIn and X -- 5,000 real and 5,000 AI-generated,
from 0.1 to 45 megapixels, with the generators unlisted. Nothing from it was ever in
training, validation or calibration of either head or fuser, and its real class is a
different in-the-wild distribution from WildRF's (different platforms, different collection,
LinkedIn included). It is the cross-dataset test the roadmap's Phase 3 asked for on the
in-the-wild side, as Synthbuster ([`07_synthbuster_summary.md`](07_synthbuster_summary.md))
is on the generator side.

Two heads: the shipped default ([experiment 02](02_experiment_summary.md), WildRF's train
split in training) and the [experiment 03](06_experiment_03_summary.md) head (the same
recipe without WildRF). The 2,000-image sample is stratified by label and platform (`manifest
sample --stratify label,generator`; a fake's platform is read from its file name by the
layout, a real's never is, so the per-platform tables below are computed from the file names).

## Result: every image, clean level

| Head | AUC | Balanced acc. at 0.5 | FPR at 0.5 | TPR at 0.5 | Balanced acc. at the tuned threshold |
|---|---|---|---|---|---|
| experiment 02 (shipped) | **0.888** | 0.747 | 0.442 | 0.936 | 0.808 (FPR 0.142, TPR 0.757) |
| experiment 03 (no WildRF) | 0.819 | 0.673 | 0.585 | 0.931 | 0.740 (FPR 0.212, TPR 0.691) |

Per platform (experiment 02 head / experiment 03 head; AUC, FPR at 0.5):

| Platform | Images | exp02 AUC / FPR | exp03 AUC / FPR |
|---|---|---|---|
| Facebook | 2,351 | 0.911 / 0.506 | 0.882 / 0.621 |
| Instagram | 3,385 | 0.917 / 0.406 | 0.826 / 0.543 |
| X | 2,064 | 0.898 / 0.414 | 0.803 / 0.577 |
| LinkedIn | 2,200 | 0.802 / 0.437 | 0.769 / 0.596 |

## Result: robustness on the 2,000-image sample

AUC / balanced accuracy at 0.5, experiment 02 head (experiment 03 head in parentheses):

| Level | AUC | Balanced accuracy |
|---|---|---|
| clean | 0.890 (0.817) | 0.746 (0.665) |
| jpeg_q75 | 0.899 (0.812) | 0.774 (0.675) |
| jpeg_q50 | 0.893 (0.796) | 0.774 (0.651) |
| resize_0.5 | 0.888 (0.843) | 0.677 (0.653) |
| resize_0.25 | 0.838 (0.784) | 0.671 (0.649) |
| social_1080_q80 | 0.874 (0.800) | 0.718 (0.644) |

## Result: the WildRF fuser applied cold

The stacking fuser of [fusion 01](04_fusion_wildrf.md) (fitted on 398 WildRF val images) and
the levels-fitted fuser of [fusion 03](05_fusion_levels_summary.md), each applied to the ITW-SM
sample **without refitting**, clean level, threshold 0.5:

| Scorer | n | AUC | Balanced acc. | FPR | TPR | ECE |
|---|---|---|---|---|---|---|
| `dinov2_head` (exp02) alone | 2,000 | 0.890 | 0.746 | 0.457 | 0.950 | 0.195 |
| fusion 01 fuser, all images | 2,000 | 0.885 | 0.794 | 0.214 | 0.802 | 0.081 |
| fusion 01 fuser, outside its band (27.1% called) | 542 | 0.945 | 0.931 | 0.112 | 0.974 | 0.047 |
| fusion 03 fuser, all images | 2,000 | 0.878 | 0.793 | 0.197 | 0.784 | 0.063 |
| fusion 03 fuser, outside its band (8.1% called) | 162 | 0.955 | 0.935 | 0.000 | 0.870 | 0.044 |

## Reading the numbers honestly

- **0.888 is the number for "in the wild, never seen".** It sits between the in-distribution
  WildRF result (0.980) and the no-WildRF cross-dataset one (0.804), and the second head
  explains the gap: training on WildRF's social-media reals lifts ITW-SM from 0.819 to 0.888,
  so what was learned on one platform mix transfers in part to another. The failure mode is
  the same one experiment 03 named -- the real class: 93.6% of the generated images are
  called generated, and 44.2% of the real photographs are too. At the tuned threshold the
  trade is 14% false positives for 76% recall.
- **LinkedIn is the hard platform, on both heads.** AUC 0.80 against 0.90-0.92 elsewhere.
  Its images are the largest and the most heavily re-encoded in the set, and the per-platform
  drop is on the real side again.
- **Recompression does not hurt here, and the reason is the data.** JPEG q75 and q50 leave
  the AUC at 0.89-0.90 and the social re-share level at 0.874; a quarter-scale resize costs
  0.05. These images arrived already laundered by their platforms, so the suite's
  perturbations change less than they did on WildRF's larger originals; the robustness table
  is flat because the damage was done before the download.
- **The fuser transfers without refitting.** Applied cold, the fusion 01 fuser halves the
  head's false-positive rate (0.457 to 0.214), cuts the calibration error from 0.195 to
  0.081, and its abstain band calls 27% of the images at balanced accuracy 0.931 with an
  11% false-positive rate. The levels-fitted fuser is stricter again (8% called, no false
  positives among them, 0.935). Fusion 01's weights were WildRF's, and this is the first
  evidence that what they encode -- lean on the head, subtract the JPEG-history signals on
  laundered reals -- carries to a second in-the-wild distribution. A fuser refitted on ITW-SM
  itself would be in-distribution again and was not run.
- **The metadata signal has a platform artifact.** On this set `metadata` scores AUC 0.583
  with a 47% true-positive and a 31% false-positive rate, which is odd for a signal that
  abstains on almost every laundered image elsewhere. The cause is one marker: Instagram
  writes a Photoshop APP13 segment into every image it serves, real or generated, and the
  signal reads that segment as an editor marker (score 0.7, label `fake`) -- 436 of the 474
  flagged fakes and all of the flagged reals are Instagram images. The fuser absorbs most of
  it (the marker fires on both classes), but as a standalone card it is wrong on Instagram,
  and the signal should learn to tell a platform's APP13 from an editor's. That is the one
  code change this benchmark asks for.
- **Two signals carry a little in-the-wild signal.** `jpeg_ghost` (AUC 0.679) and `ela`
  (0.616) rank above chance here and in the same direction as on Synthbuster, where on
  WildRF `jpeg_ghost` ran inverted; `copy_move`, `double_jpeg`, `c2pa` and `sd_watermark`
  are at 0.5, as expected on laundered images.

## Cost

Fetch 24 minutes (10,004 files, 3.4 GB, gated); the two full-set clean runs 16 minutes each;
the two 15-level runs on 2,000 images 1 h 31 min each; the seven signals on the 2,000 images
17 minutes with four worker processes; the two fusion evaluations seconds.

## Follow-up (2026-09-16): the metadata signal after the APP13 fix

The one code change this benchmark asked for is in. The `metadata` signal now reads the
APP13 segment's image-resource blocks and the IPTC-IIM datasets of its 1028 (IPTC-NAA) block:
an editor marker needs a resource block beside 1028 or an IPTC OriginatingProgram naming a
known editor, and an APP13 whose only block is 1028 counts for nothing. Meta's `FBMD`
fingerprint in IPTC SpecialInstructions is recorded under `details["platform_markers"]`,
which never moves the score. A sweep of all 10,000 files shows what the old rule was reading:

| APP13 shape | Real | Generated |
|---|---|---|
| 1028 only, `FBMD` fingerprint (Instagram, some Facebook) | 1,314 | 2,401 |
| 1028 only, news-agency captions and credits (Facebook) | 176 | 7 |
| no APP13 (all of LinkedIn and X, the rest of Facebook) | 3,510 | 2,592 |
| any other image-resource block | 0 | 0 |

So the old rule called 3,898 images edited, 1,490 of them real photographs, on the strength
of a segment no editor wrote, and no ITW-SM image carries a block that would count now.
Re-running the signal on the same 2,000-image sample
([`08_itwsm_metadata_fixed.md`](08_itwsm_metadata_fixed.md)) and on the 1,000-image WildRF
test sample of [`02_wildrf_test_signals.md`](02_wildrf_test_signals.md)
([`08_wildrf_test_metadata_fixed.md`](08_wildrf_test_metadata_fixed.md)):

| Set | AUC before / after | FPR at 0.5 before / after | TPR at 0.5 before / after |
|---|---|---|---|
| ITW-SM sample | 0.583 / 0.500 | 0.308 / 0.000 | 0.474 / 0.001 |
| WildRF test sample | 0.465 / 0.503 | 0.092 / 0.000 | 0.022 / 0.006 |

On ITW-SM the signal now abstains on every image but one generated image with an AI marker.
On WildRF, 49 of the 55 images the old rule flagged carried the same `FBMD` fingerprint
(Facebook downloads and Reddit re-posts of them), 3 carried caption-only IPTC, and the 3 that
keep their 0.70 are generated images with an XMP edit history, which is a real editor trace.
The 0.583 in the tables above was never signal: it came from the fingerprint being slightly
more common on the generated half of the sample.

The two WildRF-fitted fusers, applied cold to the sample again with the new metadata records
([`08_itwsm_fusion_wildrf_fuser_fixed.md`](08_itwsm_fusion_wildrf_fuser_fixed.md),
[`08_itwsm_fusion_levels_fuser_fixed.md`](08_itwsm_fusion_levels_fuser_fixed.md)):

| Scorer | n | AUC | Balanced acc. | FPR | TPR | ECE |
|---|---|---|---|---|---|---|
| fusion 01 fuser, all images | 2,000 | 0.886 | 0.794 | 0.214 | 0.803 | 0.082 |
| fusion 01 fuser, outside its band (27.5% called) | 551 | 0.945 | 0.931 | 0.113 | 0.974 | 0.046 |
| fusion 03 fuser, all images | 2,000 | 0.878 | 0.793 | 0.197 | 0.784 | 0.063 |
| fusion 03 fuser, outside its band (8.1% called) | 162 | 0.955 | 0.935 | 0.000 | 0.870 | 0.044 |

Nothing moves by more than 0.001 and the fusion 01 band calls 551 images instead of 542: the
fusers give `metadata` a logit weight of -0.06 (fusion 01) and -0.005 (fusion 03), so an
input that dropped from 0.70 to 0.50 on 781 images shifts the fused logit by 0.05 at most.
The fusers were not refitted for this change; the next fit uses the new signal.
