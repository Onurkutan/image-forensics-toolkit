# Experiment 03: the same head without WildRF in training

Date: 2026-09-10. Hardware: RTX 2060 (6 GB). Config:
[`configs/experiments/03_no_wildrf.yaml`](../../configs/experiments/03_no_wildrf.yaml).
Raw reports: [`06_exp03_val.md`](06_exp03_val.md), [`06_exp03_wildrf_test.md`](06_exp03_wildrf_test.md),
[`06_exp03_coco_test.md`](06_exp03_coco_test.md), [`06_exp03_cocoglide.md`](06_exp03_cocoglide.md).

## Headline

[Experiment 02](02_experiment_summary.md) reported WildRF AUC 0.980 with the caveat that
WildRF's train split was in training, so the number was in-distribution. This experiment
removes that split and changes nothing else, which turns the same WildRF test sample into a
genuine cross-dataset test. The answer is **AUC 0.804**, and the way it fails is specific:
the head keeps calling generated images generated (TPR 0.893 at 0.5) but calls **more than
half of the platform-laundered real photographs fake** (FPR 0.547). COCO photographs, which
stay in training, are unaffected (FPR 0.026). The 0.980 of experiment 02 therefore measured
"this head has seen social-media reals", not "this head generalizes to social media".

| Test set | Entries | AUC | Balanced acc. at 0.5 | FPR at 0.5 | TPR at 0.5 |
|---|---|---|---|---|---|
| Val (Community Forensics val + 800 COCO reals) | 5,564 | 1.000 | 0.993 | 0.005 | 0.991 |
| WildRF test split, never seen | 1,000 of 2,241 | 0.804 | 0.673 | 0.547 | 0.893 |
| COCO val2017 held-out photographs | 1,000 | n/a | n/a | 0.026 | n/a |
| CocoGlide, GLIDE inpainting vs authentic COCO | 1,024 | 0.627 | 0.553 | 0.043 | 0.148 |

The same rows for experiment 02: WildRF 0.980 / 0.916 / 0.137 / 0.970, COCO FPR 0.027,
CocoGlide 0.644 / 0.565 / 0.023 / 0.154.

## What changed

1. **The WildRF train split (2,712 images) is out of the train manifest.** Train: 22,164
   entries (6,996 fake, 15,168 real: Community Forensics reals plus 3,200 COCO
   photographs).
2. **800 COCO reals moved from train to val** (the 800 lowest sha256 values, so the split is
   reproducible without a seed), replacing the WildRF val split as the "not perfectly
   separable" real source calibration needs. Val: 5,564 entries (1,772 fake, 3,792 real).
   Temperature 0.46, bias -0.53, ECE 0.032 to 0.005 on val, the same shape as experiment 02.
3. Nothing else: same backbone, crops, augmented-only views, head, optimizer and early
   stopping. Best epoch 13 of 18; layer weights 0.29, 0.21, 0.19, 0.15, 0.15.

## Reading the numbers honestly

- **Per platform, the drop is everywhere and largest where the sample is largest.** AUC on
  the test split's Reddit / Twitter / Facebook images: 0.787 / 0.835 / 0.871 (experiment 02:
  0.987 / 0.980 / 0.949), with false-positive rates of 0.590 / 0.516 / 0.368 (0.114 / 0.156 /
  0.235). Facebook, the hardest platform when WildRF was in training, is the least damaged
  one when it is not, which says the damage is about the *real* class, not about any
  platform's fakes.
- **The real-image distribution is the whole story, again.** Experiment 01 fell over because
  its reals were three curated sources; experiment 02 fixed it by adding COCO and WildRF
  reals; this experiment shows that COCO alone does not carry over to social-media
  photographs. The median score of a WildRF real photograph is 0.557 here and 0.036 in
  experiment 02; the median score of a WildRF fake barely moves (0.993 vs 0.999). The head is
  reading the laundering (platform re-encoding, resizing, aggressive JPEG) as evidence of
  generation, exactly the failure the literature survey warns about for every JPEG-domain
  cue, and a frozen-feature head inherits it.
- **What the 0.804 is good for.** It is the number to quote for "a DINOv2 head trained on
  curated data, applied to social-media images it has never seen", and it is the number
  the fusion layer and the abstain band have to be judged against, not 0.980. It is also
  why the shipped default stays the experiment 02 head (`weights/dinov2_head_02`): for a
  deployment that will see social-media images, training on some of them is the right
  call, and this experiment quantifies what happens otherwise.
- **CocoGlide** moves within noise (0.627 vs 0.644); a whole-image head still does not see
  a local edit, with or without WildRF.
- ITW-SM (the in-the-wild set this project has requested access to) and Synthbuster (nine
  generator families, download pending a Zenodo outage) remain the cross-dataset tests
  the roadmap asks for; this experiment is the one that could be run with the data on disk.

## Cost

Every feature was already cached from experiment 02, so training ran from the cache: 12
minutes for 18 epochs while another benchmark shared the machine. The four benchmarks took
11 minutes with the head alone and attribution switched off (`IMGFORENSICS_HEAD_ATTRIBUTION=0`).

## Fusion on the cross-dataset head

The stacking fuser was refitted on this head plus the seven classical signals, on the same
WildRF val split as [fusion 01](04_fusion_wildrf.md) (398 images, clean level; signal
records reused from that run, head records from
[`06_exp03_wildrf_val.md`](06_exp03_wildrf_val.md)), and evaluated with
`imgforensics fusion eval` on the same 1,000-image test sample
([`06_exp03_fusion_wildrf.md`](06_exp03_fusion_wildrf.md)). The experiment 02 fuser,
re-evaluated with the same command, reproduces every number of fusion 01
([`04_fusion_wildrf_eval.md`](04_fusion_wildrf_eval.md)).

| Scorer (WildRF test, clean) | n | AUC | Balanced acc. at 0.5 | FPR at 0.5 | TPR at 0.5 | ECE |
|---|---|---|---|---|---|---|
| `dinov2_head` (experiment 03) alone | 1,000 | 0.804 | 0.673 | 0.547 | 0.893 | 0.216 |
| Fused, all images | 1,000 | 0.831 | 0.750 | 0.238 | 0.738 | 0.044 |
| Fused, outside the abstain band (floor of 20 held-out images) | 202 | 0.898 | 0.887 | 0.190 | 0.964 | 0.038 |
| Fused, outside the band as first fitted (no floor) | 31 | 0.935 | 0.750 | 0.000 | 0.500 | 0.048 |

Three things the table says:

- **The signals absorb part of the damage.** The false-positive rate falls from 54.7% to
  23.8% and the calibration error from 0.216 to 0.044, at the cost of recall (89.3% to
  73.8%). The fitted weights show how: the head's logit weight drops from 0.86 (experiment
  02 fuser) to 0.29, while `ela` rises from 0.36 to 1.14, `double_jpeg` from 0.07 to 0.63
  and `jpeg_ghost` deepens from -0.76 to -1.37. On a distribution the head has not seen,
  the fuser hands the decision back to the JPEG-domain signals, which is what a stacking
  layer fitted on that distribution should do, and a reminder that those weights are
  WildRF's, not universal.
- **The abstain band, as first fitted, was a loophole.** With a 0.9 balanced-accuracy
  target and no constraint on support, the band search settled on [0.030, 0.980], inside
  which 96.9% of the test images fell; the 31 images it did call were too few to mean
  anything, and the "1.000 outside-band balanced accuracy" recorded at fit time rested on
  three held-out images.
- **With a minimum support the band becomes a real operating point.** `fit_fuser` now
  admits only bands that leave at least max(20, 10% of the held-out split) images outside
  them. Refitted under that rule the band is [0.070, 0.850], supported by exactly the floor
  of 20 held-out images at balanced accuracy 0.900. On the test sample it calls 202 of
  1,000 images (abstains on 79.8%) at balanced accuracy 0.887, false-positive rate 0.190 and
  recall 0.964. The experiment 02 fuser is unchanged by the rule (its band was already
  supported by 48 held-out images) and reproduces every number of fusion 01.
- **Read the 79.8% abstention as the honest number.** On social-media photographs this
  head has never seen, the fused system is willing to give a verdict on one image in five
  and is right about 89% of the time when it does; the rest it hands back as "uncertain".
  That is what the abstain band is for.

## Next

- Fit the fuser across robustness levels once the WildRF level records are in (being
  collected), and see whether the band holds under the social-media re-share level.
- Add a second laundered real source that is not WildRF to training (the roadmap's
  in-house social-media re-share pipeline over COCO is the licence-clean candidate) and
  re-run this test.
