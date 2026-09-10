# Feature extraction throughput: window augmentation and worker processes

Date: 2026-09-10. Hardware: RTX 2060 (6 GB), 12 CPU cores, measured on an otherwise idle
machine (every benchmark and server stopped), two passes each, backbone weights already
cached, a fresh feature cache per run.

## What was measured

`imgforensics features extract` with two views per image (the un-augmented view 0 and one
augmented view, `configs/augment_default.yaml`), four 224 px grid crops, DINOv2 ViT-B/14 on
CUDA under fp16 autocast:

- **before** -- commit `5bc4779`, augmentation applied to the whole image before cropping,
  serial decoding;
- **after, serial** -- the current code, augmentation applied to a 2x crop window aligned to
  the image's 16-pixel grid, `--workers 1`;
- **after, 4 and 8 workers** -- the same with decoding, window cutting and augmentation in
  worker processes, the backbone in the main process.

Two samples of 200 training images: `small`, Community Forensics images at 512 px, and
`large`, WildRF images with a median longest side of 2,048 px.

## Result (images per second, pass 1 / pass 2)

| Sample | before (whole image) | after, serial | after, 4 workers | after, 8 workers |
|---|---|---|---|---|
| small (512 px) | 5.30 / 4.99 | 4.69 / 4.67 | 4.42 / 4.52 | 4.78 / 4.59 |
| large (2,048 px) | 1.27 / 1.23 | 3.39 / 3.52 | 3.16 / 3.41 | 3.27 / 3.40 |

## Reading the numbers honestly

- **The window is the win, and only on large images.** On 2,048 px images augmenting a crop
  window instead of the whole frame is 2.8x faster (1.25 to 3.45 images/s); on 512 px images
  the window is nearly the whole image and the extra bookkeeping costs about 10%. The
  experiment 02 extraction that motivated the change (52 minutes for 6,712 mixed-size images,
  see [`02_experiment_summary.md`](02_experiment_summary.md)) was dominated by exactly the
  large WildRF images this helps.
- **The worker pool buys nothing, on either sample.** Four or eight workers land within noise
  of the serial run. Once augmentation is cheap, decoding and augmenting are no longer the
  bottleneck, and what remains -- the backbone forward pass, the cache writes and the
  per-image batching in the main process -- is not parallelised by this option. `--workers`
  stays available and harmless, but the CHANGELOG's description of it as the fix for
  CPU-bound extraction is not what the measurement shows; the next lever is on the GPU side
  (larger cross-image batches, asynchronous cache writes).
- Throughput at 512 px, about 5 images/s for 8 crops each, is roughly 40 crops/s, an order of
  magnitude below what a ViT-B/14 forward pass at 224 px can do on this card; the extraction
  loop, not the model, sets the pace. That is the profiling target.
