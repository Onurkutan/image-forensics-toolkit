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

Measured on this project's RTX 2060 (6 GB), batch 1, fp16 autocast, weights
`CAT_full_v2.pth.tar` (114.3 M parameters):

| input | peak VRAM allocated | peak VRAM reserved | per image | of which coefficient decoding |
|---|---|---|---|---|
| 256x256 (CocoGlide, PNG -> quality-100 JPEG) | 473 MiB | 502 MiB | 549 ms | ~340 ms |
| 1024x1024 JPEG, one tile | 1000 MiB | 1152 MiB | ~2.0 s | ~1.5 s |
| 1600x1200 JPEG, four tiles | 1000 MiB | 1244 MiB | ~2.9 s | ~1.0 s |

Peak VRAM is flat above one tile, as it must be: tiles are run one at a time.
The dominant cost is not the model but the pure-Python JPEG entropy decoder
that feeds its DCT stream (`imgforensics.localization._jpegcoef`, written
because `jpegio` has no Windows wheel and none past CPython 3.10). It is
worst on exactly the input this model asks for -- a quality-100 JPEG, where
almost no coefficient quantizes to zero -- and on high-entropy content; the
1024x1024 row above uses a synthetic noise-and-shapes image and is close to a
worst case.

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
