# Benchmark report: CocoGlide

- Entries evaluated: 1024
- Robustness levels: clean
- Fixed threshold: 0.5
- Threshold caveat: threshold tuned in-sample

## Image metrics (clean)

| detector | threshold | auc | ap | accuracy | balanced_accuracy | fpr | tpr | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| catnet_v2 | fixed | 0.666 | 0.684 | 0.589 | 0.589 | 0.551 | 0.729 | 0.208 | 0.279 |
| catnet_v2 | tuned | 0.666 | 0.684 | 0.633 | 0.633 | 0.252 | 0.518 | 0.208 | 0.279 |

## Robustness (AUC / balanced accuracy at the fixed threshold)

| detector | level | auc | balanced_accuracy |
|---|---|---|---|
| catnet_v2 | clean | 0.666 | 0.589 |

## Per-generator AUC (clean)

| detector | generator | auc |
|---|---|---|
| catnet_v2 | glide | only one label present; AUC undefined |
| catnet_v2 | none | only one label present; AUC undefined |

## Per-source AUC (clean)

| detector | source | auc |
|---|---|---|
| catnet_v2 | CocoGlide | 0.666 |

## Pixel metrics

| detector | n | mean f1@0.5 | mean best-f1 | mean ap | mean iou |
|---|---|---|---|---|---|
| catnet_v2 | 512 | 0.364 | 0.605 | 0.566 | 0.288 |

## Timing

| detector | n | mean elapsed_ms |
|---|---|---|
| catnet_v2 | 1024 | 548.77 |

## Cost (appended by hand; the tables above are `imgforensics benchmark` output)

Re-measured on 2026-09-21 on this project's RTX 2060 (6 GB), batch 1, fp16
autocast, weights `CAT_full_v2.pth.tar` (114.3 M parameters), three
repetitions per input and the median of them. The last two columns are the
same run with `jpeglib` switched off, so the comparison is one machine on one
evening rather than two dates:

| input | peak VRAM allocated | peak VRAM reserved | per image | of which decoding | per image, numpy fallback | of which decoding |
|---|---|---|---|---|---|---|
| 256x256 (CocoGlide, PNG -> quality-100 JPEG), median of 10 | 473 MiB | 502 MiB | 193 ms | 26 ms | 495 ms | 314 ms |
| 1024x1024 quality-90 JPEG, one tile | 1000 MiB | 1244 MiB | 446 ms | 51 ms | 2.07 s | 1.52 s |
| 1600x1200 quality-90 JPEG, four tiles | 1000 MiB | 1244 MiB | 1.57 s | 66 ms | 4.26 s | 2.70 s |

Before the `jpeglib` fast path (2026-09-10, same machine and weights, and the
run the `## Timing` table above comes from): 549 ms per 256x256 image of
which ~340 ms decoding, ~2.0 s of which ~1.5 s at 1024x1024 (1152 MiB
reserved), ~2.9 s of which ~1.0 s at 1600x1200. Both synthetic rows are now
`tests/conftest.py`'s `natural_like_image` saved at quality 90, which
reproduces that run's 1024x1024 row to within 4% in time and exactly in
allocated VRAM; the 1600x1200 input of that run was a softer image and
decoded faster than this one does, so read that row against the columns
beside it rather than against 2026-09-10.

Peak VRAM is flat above one tile, as it must be: tiles are run one at a time.
It is also untouched by any of this -- the decoder never sees the GPU. The
dominant cost is now the model: coefficient decoding is 13%, 11% and 4% of a
prediction on the three rows above, against 63%, 73% and 63% with the
fallback. `imgforensics.localization._jpegcoef` hands the entropy decoding to
libjpeg through `jpeglib` and keeps its own pure-Python decoder for the cases
that package cannot be used on -- it is missing, or the stream is one the two
would not read identically -- so the numbers in the last two columns are what
a machine with no `jpeglib` wheel still gets. The fallback's old worst cases
still describe it: a quality-100 JPEG, where almost no coefficient quantizes
to zero, and high-entropy content, which both synthetic rows above are.

**Most in-the-wild JPEGs reach this model as a re-encode.** The 256x256 row's
input is a PNG, so `catnet_v2` decodes it and re-encodes it at quality 100
before reading its DCT stream -- upstream's own handling of non-JPEG input,
and a compression history this toolkit created rather than one the image
arrived with. A sweep of ITW-SM on 2026-09-21 found that caveat covers most
real files too: of its 9,981 files (69 of them not JPEGs at all) 8,015 are
progressive and 1,897 baseline, and every one of the baseline ones is
Facebook's re-encode. A progressive JPEG takes exactly the same route as the
PNG, pixels out and quality 100 back in, because neither decoder reads SOF2.
So on most social-media images this detector's second stream is reading the
toolkit's own quantization, not the platform's. libjpeg can read progressive
coefficients; wiring that in is the next slice of milestone 6e and will change
these results.

## Comparison with IML-ViT on the same 1,024 images

| | `iml_vit` | `catnet_v2` |
|---|---|---|
| image AUC | 0.535 | **0.666** |
| image AP | 0.539 | **0.684** |
| pixel F1@0.5 | 0.059 | **0.364** |
| pixel best-F1 | 0.486 | **0.605** |
| pixel AP | 0.423 | **0.566** |
| pixel IoU | 0.037 | **0.288** |
| mean ms/image | **462** | 549 |

Per manipulated-mask area (small < 10% of the image, medium 10-30%, large
> 30%; 192 / 154 / 166 images):

| detector | metric | small | medium | large |
|---|---|---|---|---|
| `iml_vit` | best-F1 | 0.316 | 0.457 | 0.711 |
| `catnet_v2` | best-F1 | 0.387 | 0.640 | 0.824 |
| `iml_vit` | F1@0.5 | 0.079 | 0.053 | 0.041 |
| `catnet_v2` | F1@0.5 | 0.159 | 0.404 | 0.564 |
