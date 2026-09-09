# image-forensics-toolkit

[![CI](https://github.com/Onurkutan/image-forensics-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/Onurkutan/image-forensics-toolkit/actions/workflows/ci.yml)

A toolkit to detect AI-generated images, AI-inpainted regions, and classic manipulations
(splicing/copy-move), producing an image-level score plus an optional heatmap.

Status: early development (v0.1.0). Seven classical signals are implemented so
far: `metadata` (EXIF/XMP/editor/AI-generator markers, thumbnail consistency,
JPEG quality estimate and quantization-table classification), `ela` (Error
Level Analysis with heatmap), `c2pa` (C2PA manifest verification),
`sd_watermark` (Stable Diffusion invisible-watermark decode), `copy_move`
(block-matching duplicated-region detection with heatmap), `jpeg_ghost`
(JPEG-ghost recompression-quality mismatch with heatmap) and `double_jpeg`
(blocking-grid offset and aligned double-quantization periodicity). See
[Signals](#signals) below for what each one reads and its blind spots.

## Planned architecture

- **Detectors** (`imgforensics.detectors`): image-level classifiers that score a whole image as
  real or AI-generated.
- **Localization** (`imgforensics.localization`): pixel-level localizers that highlight
  manipulated or AI-inpainted regions with a heatmap.
- **Signals** (`imgforensics.signals`): classical low-level signal analyzers such as noise
  residuals, error level analysis, and JPEG artifact statistics.
- **Fusion** (`imgforensics.fusion`): combines detector and localizer outputs into a single
  image-level score with an explanation.

## Quickstart

```bash
git clone https://github.com/Onurkutan/image-forensics-toolkit.git
cd image-forensics-toolkit
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
# Optional: pip install -e ".[dev,provenance]" to also enable the c2pa signal
imgforensics analyze path/to/image.jpg
imgforensics analyze image.jpg --json --save-heatmaps out/
pytest
```

## Signals

Each signal is a self-contained, explainable check. Score is the probability
the image is generated/manipulated (0 = confidently real, 1 = confidently
fake); label is `real`, `fake` or `uncertain`. No signal alone should be read
as a verdict -- see `details` in its output for the evidence.

| Signal | Reads | Score means | Known blind spots |
|---|---|---|---|
| `metadata` | EXIF/XMP, PNG text chunks, Photoshop APP13, embedded EXIF thumbnail, JPEG quantization tables | High = a known AI-generator/editor marker or a thumbnail/image mismatch was found; low = camera EXIF with no edit trace; 0.5 = no usable metadata | Stripped by almost every sharing platform (upload to social media and this signal goes blind); markers are only as good as the list of known tool names |
| `ela` | Re-encodes the image at a fixed JPEG quality and diffs against the original | Bounded to [0.3, 0.65] -- an explanation aid (see the heatmap), never a standalone verdict | Useless on an already-uniformly-recompressed image; a second JPEG save by any platform equalises the error level everywhere |
| `c2pa` | Any C2PA manifest embedded in the file (via `c2pa-python`, optional `provenance` extra) | High = signed as AI-generated or the signed content hash no longer matches; low = signed camera capture with no edits; 0.5 = no manifest or an untrusted/self-signed signer with no other evidence | A missing manifest proves nothing -- most images, including AI-generated ones, carry no C2PA data at all; manifests are stripped by many platforms just like EXIF |
| `sd_watermark` | Decodes the DWT-DCT ("dwtDct") invisible watermark Stable Diffusion reference pipelines embed, checking bit-agreement against known payloads | High = a known payload's decoded bits matched; 0.45 = no known payload matched | This specific scheme embeds only in chroma and does not survive JPEG re-encoding (even at quality 100, due to chroma subsampling) or a resize; only covers the two reference-pipeline payloads, not every SD fork or later generators |
| `copy_move` | Block-matching search (downscaled, quantized zig-zag DCT features) for a duplicated region copied and pasted elsewhere in the same image, with a matched-region heatmap | 0.85 "fake" = a dominant shift with enough votes and matched area found; 0.60 "uncertain" = accepted but small matched area; 0.45 "uncertain" = no duplicate found (not evidence of authenticity) | Blind to a clone that was rotated or rescaled before pasting; heavy recompression can in principle merge distinct blocks; naturally repetitive textures (tiles, fences) are guarded against via a minimum-distance rule and a shift-consistency check, but are not impossible to fool |
| `jpeg_ghost` | Re-saves the image at a range of JPEG qualities and finds, per block, the quality whose re-save error is anomalously low compared to the rest of the image (Farid's JPEG ghosts), with a heatmap | Bounded to [0.3, 0.7] like `ela` -- an explanation aid, never a standalone verdict | Requires the image to be JPEG-derived; useless once a platform has uniformly re-encoded the whole image after the fact |
| `double_jpeg` | Two independent pixel-domain checks: whether the JPEG 8x8 blocking grid still starts at the image origin, with a secondary-phase check for a masked older grid underneath it (crop/composite detection); and, for JPEG inputs only, whether the DCT coefficient histogram -- normalized by the file's own quantization step, so the check measures a genuine second, coarser compression rather than just how lossy the current one is -- shows aligned double-quantization periodicity | 0.75 "fake" = blocking grid misaligned; 0.60 "uncertain" = a second, offset grid or double compression suspected (weak evidence alone); 0.45/0.40 "uncertain" = no JPEG history detectable / no evidence either way | Both checks are quantization-history fingerprints, not proof of malicious editing; the double-quantization check only detects a *coarser-then-finer* double compression (the reverse order, and same-or-finer-then-coarser, leave no detectable comb) and only on JPEG inputs; a suspected double compression is common for any re-shared image, and a misaligned grid only proves a crop-then-resave happened |

## Project layout

```
image-forensics-toolkit/
├── src/imgforensics/
│   ├── core/            # types, base detector class, registry
│   ├── detectors/       # image-level detectors
│   ├── localization/    # pixel-level localizers
│   ├── signals/         # classical low-level signals
│   ├── fusion/          # score fusion and explanation
│   ├── data/            # dataset loaders and download helpers
│   ├── utils/           # image I/O helpers
│   └── cli.py           # command-line interface
├── tests/
└── docs/
```

## Data and evaluation

A dataset manifest is a JSON Lines file of labeled images (path, label,
source, generator, split, mask path, sha256, resolution, format, JPEG
quality) plus a `*.meta.json` sidecar (dataset name, license,
`commercial_ok`). Build one from a folder tree with `real`/`fake`
subfolders via `imgforensics manifest build ROOT --dataset NAME --out
manifest.jsonl`, browse the external dataset registry with `imgforensics
datasets list` / `datasets show NAME`, and check a manifest's real/fake
halves for format, resolution, JPEG-quality, and duplicate-image bias with
`imgforensics audit manifest.jsonl [--strict]`. `imgforensics datasets
fetch NAME --dest DIR --accept-license` downloads a registered dataset per
its packaged recipe (`imgforensics datasets recipe NAME` prints it first;
`--dry-run` prints the license and plan without downloading) -- it always
prints the license and refuses to proceed without `--accept-license`, and
never asks for or stores Kaggle/Hugging Face credentials: a dataset that
needs one prints instructions and stops instead. `imgforensics datasets
prepare NAME --src DIR --out manifest.jsonl` turns a downloaded folder into
a manifest using a per-dataset layout description, falling back to
`label_from_parent_folder` when none is registered. A Hugging Face (`hf`)
step can bound its download to a `max_files` sample of the repository
instead of pulling it whole -- Community Forensics defaults to the first 8
sorted Parquet shards of the ~260 GB `-Small` repository (`--variant full`
for everything). `imgforensics manifest
sample IN --n N --out OUT` draws a deterministic, stratified subsample, and
`imgforensics manifest merge A B ... --out OUT` combines manifests.
`imgforensics manifest crop IN --out-dir DIR --out OUT --size N --mode
center|tiles [--label real ...]` writes native-resolution square crops
(never a resize) of the selected labels into a new manifest, closing a
resolution gap between classes -- e.g. 1024 px reals vs. 512 px fakes --
that would otherwise let a detector learn scene scale instead of
generation artifacts. See
[`imgforensics.eval.metrics`](src/imgforensics/eval/metrics.py) for the
image-level (AUC, AP, accuracy, ECE, ...) and pixel-level (F1, best-F1, AP,
IoU) metrics used to score detectors and localizers.

## Benchmarking

`imgforensics benchmark manifest.jsonl [--detector NAME ...] [--baselines]
[--robustness default|PATH|none] [--out results.json] [--report report.md]`
runs registered detectors, plus optional trivial baselines, over a manifest
at every level of a deterministic robustness suite (clean; JPEG/WEBP
re-encoding; resize; a resize round-trip; center crop; Gaussian noise; a
"social" resize+JPEG+metadata-strip pipeline; see
[`robustness_default.yaml`](src/imgforensics/eval/robustness_default.yaml)),
and prints Markdown tables: image metrics, per-level robustness, per-group
AUC, pixel metrics, and timing. Each detector's threshold is tuned on the
manifest's val split when one exists; otherwise the report is marked
"threshold tuned in-sample" as a caveat against reading it as held-out. At
any operating threshold, a score must be strictly above it to count as a
"fake" call, so a detector abstaining at the classical-signal midpoint of
0.5 is not scored as calling every image fake.

## Learned detectors (optional `ml` extra)

The learned half of the toolkit (Phase 3) is built on a **frozen** ViT
backbone: features are extracted once, cached on disk, and a small head is
trained on top of them. Nothing here is installed by default -- `torch`,
`torchvision`, `timm` and `safetensors` live in the optional `ml` extra, and
the rest of the package imports and runs without them.

```bash
# 1. torch first, from the wheel index your driver supports (cu130 shown; use cpu for CPU-only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# 2. then the package with the ml extra (timm, safetensors resolve from PyPI)
pip install -e ".[dev,ml]"

imgforensics features extract manifest.jsonl --cache-dir data/features
imgforensics features info --cache-dir data/features
```

Two backbones are registered: `dinov2_vitb14` (DINOv2 ViT-B/14, the default;
a frozen self-supervised space separates real from generated images better
than a language-aligned one) and `clip_vitl14` (OpenAI CLIP ViT-L/14, kept as
the comparison space). Both are read at several depths: `extract` returns the
CLS token of each selected transformer block plus the model's pooled output,
as an array of shape `(n_crops, n_layers, dim)`.

**Crop, never resize.** Every image reaches the backbone as native-resolution
224 px crops -- `center`, `grid` (the tiles closest to the image center) or
`random` (seeded from the image's own bytes, so the crops are reproducible per
image). Resizing would low-pass exactly the high-frequency generator artifacts
a detector keys on, so it never happens: images *smaller* than the crop size
are reflection-padded, not upscaled.

**Feature cache.** `imgforensics features extract` writes one `.npz` per
(image sha256, backbone, crop policy) under `--cache-dir` (default
`data/features`, gitignored), storing the features as float16 alongside the
layer indices, crop policy and backbone that produced them. Re-running skips
anything already cached, so changing the head -- or adding images to a
manifest -- costs no GPU time for the images already done. Both `features`
commands print an install hint and exit 1 when the `ml` extra is missing.

**Weights are downloaded, never committed.** The backbone weights (about
350 MB for DINOv2 ViT-B/14) are fetched from the Hugging Face Hub on first
use and cached there; set `IMGFORENSICS_WEIGHTS_DIR` to keep them in the
project's gitignored `weights/` directory instead. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for each model's license.

### Training a head

The backbone stays frozen; the only thing that trains is a ~1.06 M-parameter
head (per-layer LayerNorm + projection, learned layer-importance weights, and
a small MLP) over the cached features. On the cached features that is minutes,
not hours.

```bash
# 1. Cache features. --views 2 adds one augmented copy per image on top of the
#    un-augmented view 0, so the head never learns the training set's own
#    JPEG history. (Optional -- `train head` extracts anything missing itself.)
imgforensics features extract data/manifests/train.jsonl     --cache-dir data/features --views 2 --augment configs/augment_default.yaml

# 2. Train, calibrate and write the checkpoint.
imgforensics train head --config configs/head_dinov2.yaml [--epochs N] [--out DIR]
```

Augmented views are part of the cache key, so switching augmentation policies
never silently reuses the wrong features, and view 0 is shared with a plain
extraction because it is un-augmented by construction. `configs/augment_default.yaml`
is the packaged policy: JPEG Q30–95 (p 0.7), WEBP Q40–95, downscale-upscale,
blur, noise and cut-out, each at its own rate.

**Where the checkpoint lands.** `configs/head_dinov2.yaml` writes
`weights/dinov2_head/` (gitignored), holding three files: `head.safetensors`
(the weights), `head.json` (everything needed to reuse them) and
`training_log.jsonl` (one line per epoch). Training keeps the best
image-level validation AUC, early-stops, then fits a temperature and bias on
the validation split and reports the ECE before and after — a calibration
that does not reduce the ECE is not applied, and the checkpoint says so.

**How `analyze` picks it up.** The learned detector is registered as
`dinov2_head` and runs alongside the classical signals, but only when the `ml`
extra is installed. It looks for its checkpoint in `weights/dinov2_head/`, or
wherever `IMGFORENSICS_HEAD_DIR` points:

```bash
IMGFORENSICS_HEAD_DIR=weights/my_head imgforensics analyze image.jpg --detector dinov2_head
```

It crops the image per the checkpoint's own crop policy, scores each crop, and
reports the mean calibrated probability plus a heatmap with each crop's
probability painted over the region it came from. **With no checkpoint
installed it abstains** — score 0.5, label `uncertain`, and a `reason` saying
where it looked — rather than failing the run.

**The checkpoint records its training data's licenses.** `head.json` carries
each training and validation manifest's name, entry count, file sha256,
license string, and the `commercial_ok` flag ANDed across them (`null` when
any source is unverified, `false` when any is research-only). A head trained
on research-only data is itself research-only, and this is the only place that
fact survives the move from data to model.

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md). Step-by-step runbooks for individual
experiments (what to run, expected disk footprint, what to report) live in
[docs/experiments/](docs/experiments/).

## Research notes

See [docs/research/](docs/research/).

## License

The code in this repository is released under the MIT license, see [LICENSE](LICENSE).

This is a personal, non-commercial research project. Third-party models, weights and
datasets are not included in the repository; they are downloaded from their original sources
and keep their own licenses, some of which permit research use only. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the current library list; models and
datasets are added there as they are integrated.
