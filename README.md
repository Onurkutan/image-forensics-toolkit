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
- **Views** (`imgforensics.views`): maps that render the image under one transform and claim
  no verdict, for a person to read.
- **Fusion** (`imgforensics.fusion`): combines detector and localizer outputs into a single
  image-level score with an explanation.
- **Service** (`imgforensics.service`): the headless session layer -- tool catalogue,
  parameters, cached results, map tiles -- an interactive client talks to.

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
│   ├── views/           # maps that show, without claiming a verdict
│   ├── fusion/          # score fusion and explanation
│   ├── service/         # headless session layer for an interactive client
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
[--robustness default|PATH|none] [--out results.json] [--report report.md]
[--workers N]`
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
0.5 is not scored as calling every image fake. `--workers N` runs the
classical signal detectors (and, if `--baselines` is passed, the cheap
`constant`/`random` baselines) across `N` worker processes instead of one,
so they no longer bottleneck a slower GPU-based detector evaluated in the
same run; results are identical to `--workers 1` (the default), only faster.

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

Extraction can run image decoding, window cutting and augmentation in worker processes with
`features extract --workers N`; augmented views are computed on a 2x crop window aligned to the
image's 16-pixel grid rather than on the whole image, which keeps the JPEG block alignment of
whole-image augmentation while avoiding full-resolution re-encoding of very large photographs.

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

### Attribution: where the head looked

The heatmap says *how generated* each crop looks; the attribution map answers
the other half of the question — *where inside those crops* the classifier
found its evidence. It is Grad-CAM on the backbone's own patch grid (16x16 for
DINOv2 ViT-B/14 at 224 px), with two adaptations to this architecture: the
gradient is taken of the **calibrated** logit, so the picture explains the
number the report prints rather than an intermediate one, and because the head
reads several transformer blocks there is one map per depth, combined with the
head's own learned layer-importance weights.

`analyze` writes it next to the heatmap: `<stem>_<detector>_attribution.png`
under `--save-heatmaps`, an `attribution` entry in `--json`, and a
`<detector>_attribution.png` / `<detector>_attribution_overlay.png` pair plus a
captioned figure under the heatmap overlay in `--report-dir`'s `report.md`.

```bash
IMGFORENSICS_HEAD_ATTRIBUTION=0 imgforensics analyze image.jpg --report-dir out/
```

It is on by default and costs one extra forward and backward pass over the
handful of 224 px crops the score already used — no second model, nothing to
download. Measured on this project's RTX 2060 with the backbone already
loaded: 27 ms to 91 ms per image for a one-crop 256 px image, 59 ms to
149 ms for a four-crop 720x960 photograph. `IMGFORENSICS_HEAD_ATTRIBUTION=0`
(or `false`/`no`/`off`) skips it.
The score, the label and the heatmap come from the untouched no-grad path
either way, so switching attribution on or off changes no number.

**A saliency map is not a manipulation mask.** It is normalized to a maximum
of 1 within its own image, so its values rank pixels against each other and
mean nothing across images — a bright pixel marks evidence the classifier used
for its *score*, not a claim that the pixel was edited. `dinov2_head` is a
whole-image classifier, so on a fully generated image that evidence can sit
anywhere, empty sky included. For "which pixels were manipulated", the
localizers below are the right tool.

## Localization (optional `ml` extra)

Where the detectors answer *how* generated an image looks, the localizers
answer *where*. Phase 4a registers two, both inference-only wrappers around
released weights, both returning a per-pixel probability that a pixel was
manipulated, plus `localizer_ensemble`, which combines their heatmaps. They
are registered as normal detectors, so `analyze` and `benchmark` treat them
like any other, but their `heatmap` is the primary output and the image-level
score is derived from it.

| detector | model | cue | license (code / weights) |
|---|---|---|---|
| `iml_vit` | [IML-ViT](https://github.com/SunnyHaze/IML-ViT) (arXiv:2307.14863), ViT-B/16 with windowed attention, a simple feature pyramid and an edge-supervised decoder head, trained on CASIA v2 | RGB pixels only | MIT / MIT |
| `catnet_v2` | [CAT-Net v2](https://github.com/mjkwon2021/CAT-Net) (WACV 2021 / IJCV 2022), an HRNetV2-W48 RGB stream fused with a DCT stream, trained on CASIAv2 + FantasticReality + IMD2020 + tampCOCO + compRAISE | RGB pixels **and** the JPEG stream's quantized DCT coefficients and quantization table | Apache-2.0 / CC-BY-4.0 |

```bash
imgforensics weights list                              # what is registered, and what is installed
imgforensics weights fetch iml_vit --accept-license    # 350 MB, MIT, sha256-verified
imgforensics weights fetch catnet_v2 --accept-license  # 873 MB, CC-BY-4.0, sha256-verified
imgforensics analyze image.jpg --detector iml_vit --detector catnet_v2 --save-heatmaps out/
```

Using `catnet_v2`'s weights requires **attributing CAT-Net** (CC-BY-4.0); see
THIRD_PARTY_NOTICES.md.

`weights fetch` prints the model's license, `commercial_ok` flag, size and
source and refuses to download without `--accept-license`, exactly as
`datasets fetch` does; the file is verified against a recorded sha256, is
re-verified rather than re-downloaded on a second run, and lands in
`weights/<name>/` (or wherever `IMGFORENSICS_WEIGHTS_DIR` points) — both
gitignored. **Weights are never committed.** The two `weights` commands work
without the `ml` extra installed: fetching a model and being able to run it
are separate problems.

### `iml_vit`: pixels only

**Crop and pad, never resize — and never truncate.** IML-ViT is a 1024×1024
model that reaches that size by zero-padding at the bottom-right, not by
scaling, so a small image keeps its native pixel statistics. Upstream's own
transform then *crops* anything larger than 1024 px to its top-left corner,
which throws away most of a modern photograph; this wrapper instead runs
overlapping 1024 px tiles at stride 768 and averages the overlaps, so every
pixel is seen at native resolution and the returned heatmap covers the whole
image at its original shape. The number of tiles used is reported in
`details["tiles"]`.

**The image-level score.** The paper and the released code report pixel
metrics only, so there is no upstream rule to follow: this detector reports
the mean of the top 1% of heatmap values. A plain mean would shrink with the
manipulated region and call every small edit authentic; a plain max is one
noisy pixel away from calling everything fake. `details` carries `max_prob`,
`mean_prob` and `area_fraction_above_0.5` next to it, so the score can always
be checked against the map it came from. **With no weights installed the
localizer abstains** — score 0.5, label `uncertain`, and a `reason` naming
the directory it looked in and the command that fills it — rather than
failing the run.

Measured on this project's RTX 2060 (6 GB), batch 1 under fp16 autocast:
1.6 GB peak allocated / 2.6 GB peak reserved VRAM, and 0.3–0.5 s per 1024 px
tile (the spread is GPU contention, not image size — a 256 px thumbnail costs
the same forward pass as a 1024 px photograph, and an n-tile image costs n of
them).

**It does not survive diffusion inpainting.** On CocoGlide — the standard
probe for exactly that — IML-ViT scores pixel F1@0.5 0.059, best-F1 0.486,
AP 0.423, IoU 0.037 and an image-level AUC of 0.535, against a
predict-everything trivial baseline of F1 0.355 / AP 0.252. It ranks
manipulated pixels better than chance but almost never crosses 0.5 on a
GLIDE edit, so it is a weak, badly-calibrated signal on this domain and must
not be read as a verdict. Full tables in
[docs/benchmarks/03_cocoglide_iml_vit.md](docs/benchmarks/03_cocoglide_iml_vit.md).

### `catnet_v2`: the JPEG stream, not just the pixels

CAT-Net's second stream reads the *quantized DCT coefficients* and the
*quantization table* of the JPEG the image is stored in, so a region pasted
in with a different compression history stands out even where the pixels
look seamless. Two consequences follow, both of them upstream's design and
both reproduced here:

- **Non-JPEG input is re-encoded to a quality-100, 4:4:4 JPEG first** (in
  memory), because the model has no other way to be fed. `details` records
  `input_was_jpeg`, `jpeg_quality_estimate` (the input's own, from the
  `metadata` signal's estimator) and `dct_source`, so a heatmap can never be
  read without knowing which of the two paths produced it.
- Reading those coefficients needs a JPEG entropy decoder. Upstream uses
  `jpegio`, which publishes no Windows wheel and no wheel past CPython 3.10,
  so this project decodes them in pure numpy/Python instead
  (`imgforensics.localization._jpegcoef`: baseline and extended-sequential
  Huffman JPEGs, chroma subsampling and restart intervals included;
  progressive, arithmetic and 12-bit JPEGs are rejected by name and fall back
  to the re-encode). It is checked against libjpeg two ways — dequantize +
  inverse-DCT the decoded coefficients and compare to Pillow's own decoded
  pixels (max abs error 1 grey level), and re-quantize those pixels and
  compare back to the coefficients. **No new dependency was added.**

Inference runs at full resolution up to 1024 px, padded to whole 8x8 blocks
with 127.5 the way upstream's dataset does; anything larger is covered by
1024 px tiles at stride 768, both multiples of 8 so a tile boundary never
cuts a DCT block in half. The image-level score is the same top-1% rule as
`iml_vit`, deliberately, so the two are comparable.

**On CocoGlide it clearly beats `iml_vit`**: pixel F1@0.5 0.364 vs 0.059,
best-F1 0.605 vs 0.486, AP 0.566 vs 0.423, IoU 0.288 vs 0.037, image-level
AUC 0.666 vs 0.535 — and unlike `iml_vit` it also beats the
predict-everything baseline at the fixed 0.5 threshold, not only on ranking.
Read that as a lower bound rather than a like-for-like number: CocoGlide is
PNG, so every image goes through the quality-100 re-encode and the DCT
stream is reading a compression history this toolkit created. Cost on the
RTX 2060: 1.0 GB peak allocated / 1.2 GB reserved at 1024 px (flat above one
tile), 549 ms per 256 px image — of which roughly 340 ms is the Python JPEG
decoder, not the network. Full tables in
[docs/benchmarks/03_cocoglide_catnet.md](docs/benchmarks/03_cocoglide_catnet.md).

### `localizer_ensemble`: both maps at once

The two localizers read different evidence — one only the pixels, the other
also the JPEG stream — so `localizer_ensemble` runs both and combines their
heatmaps pixelwise. It scores the combined map with the same top-1% rule its
members use, so its number belongs in the same benchmark column as theirs,
and it reports `member_scores` and `member_elapsed_ms` in `details` so the
combination can always be traced back to what went into it.

```bash
imgforensics analyze image.jpg --detector localizer_ensemble --save-heatmaps out/
IMGFORENSICS_LOCALIZER_ENSEMBLE_MODE=max imgforensics analyze image.jpg --detector localizer_ensemble
```

Three combination modes, chosen with `IMGFORENSICS_LOCALIZER_ENSEMBLE_MODE`
(or the `mode=` constructor argument):

- `mean` (default) — the pixelwise mean. Both members emit calibrated
  probabilities, so a fixed 0.5 threshold on their mean still means what it
  means for either member alone.
- `max` — the pixelwise maximum: whichever member is more confident at each
  pixel. Finds a region one member missed entirely, at the cost of inheriting
  the other's false positives.
- `rank_mean` — each map is replaced by its own per-image percentile rank
  first, which equalizes members whose probabilities live on different
  scales. **It discards both members' calibration:** a rank map's values are
  uniform over [0, 1] by construction, so thresholding one at 0.5 marks the
  upper half of the image whatever the image contains. Pixel AP and best-F1
  stay meaningful under it; F1@0.5 and IoU@0.5 do not. Hence not the default.

**Memory.** The ensemble owns its own member instances, so benchmarking
`iml_vit`, `catnet_v2` and `localizer_ensemble` in one run loads each model
twice. On a 6 GB card, give the ensemble its own `benchmark` invocation. A
member with no weights installed is dropped, and with none of them installed
the ensemble abstains the same way its members do.

**Pixel metrics per robustness level.** The benchmark runner now records pixel
metrics at every level whose perturbation keeps the pixel grid — `clean`,
JPEG, WEBP and Gaussian noise — not only at `clean`, so the "## Pixel metrics"
table carries a `level` column and a heatmap's quality can be read across
recompression. Levels that resize, crop or re-share the image are still
skipped: the manifest's mask is stored against the original geometry. The
packaged `localization` suite is exactly the geometry-preserving half of the
default one, with identical params, so every level in it carries pixel
metrics:

```bash
imgforensics benchmark manifest.jsonl --detector localizer_ensemble --robustness localization
```

The CocoGlide comparison against the individual members is recorded in
[docs/benchmarks/](docs/benchmarks/) once the run is done.

## Fusion

`imgforensics.fusion` combines several detectors' scores into one calibrated
probability with an abstain band, using a pure-numpy L2-regularized logistic
stacking model (no scikit-learn, no torch) fitted on saved benchmark records:

```bash
imgforensics benchmark manifest.jsonl --all-signals --out results.json --robustness none
imgforensics fusion fit results.json --out weights/fuser.json
imgforensics analyze image.jpg --fuser weights/fuser.json   # or set IMGFORENSICS_FUSER
imgforensics fusion eval results.json --fuser weights/fuser.json --report eval.md
```

`fusion eval` reports, per robustness level, each detector's own AUC / balanced accuracy /
FPR / TPR alone, the same numbers for the fused verdict over every image, and again over just
the images the fuser is willing to call (outside the abstain band, with the abstain rate) --
the table [docs/benchmarks/04_fusion_wildrf.md](docs/benchmarks/04_fusion_wildrf.md) reports,
now reproducible from a fitted fuser and saved benchmark results instead of an ad-hoc script.

Each detector's score is imputed as abstaining (0.5) when it did not run, so
a fuser degrades gracefully with a subset of its detectors present; a "real
below low / fake above high / uncertain in between" band is fitted on a
held-out split so the fuser can say "not sure" instead of guessing.
`imgforensics fusion info fuser.json` prints its weights, band and metrics
(train/held-out AUC, ECE before/after calibration, abstain rate). The band search only
accepts a band that leaves enough held-out images outside it -- the larger of
`--min-outside-count` (default 20) and `--min-outside-fraction` (default 0.10) of the
held-out split -- because a band supported by a handful of images is a loophole, not a
fit; when no such band meets the target, the best one is kept and `band target met` is
reported as false.

`imgforensics analyze IMAGE --report-dir DIR` writes a self-contained report folder: `report.json`
with every detector's score, label and details plus the fused verdict when a fuser is configured,
`<detector>_heatmap.png` and `<detector>_overlay.png` for every detector that produced a heatmap,
and `report.md` with one plain-language card per detector, the fused verdict first.

## Service layer (Phase 6a)

`imgforensics.service` is the headless layer an interactive client sits on -- the forensic
workbench described in
[docs/design/01_toolbox_architecture.md](docs/design/01_toolbox_architecture.md). It computes
no forensics of its own: it holds the state a one-shot command does not need. `catalogue()`
lists every registered tool with its category, kind, parameters and whether its weights are
installed (answered without importing torch), so a client can draw a tool tree before running
anything. `AnalysisSession` loads one image once and runs tools on it lazily, caching each
result under the parameters it was produced with, and serves each map as a `MapPyramid` --
levels down to 256 px, 256-pixel tiles on request -- so a 12-megapixel heatmap never crosses
the wire whole. `SessionStore` keeps sessions in memory with a TTL and a cap, so a public
demo needs no database.

```python
from imgforensics.core.image import ForensicImage
from imgforensics.service import AnalysisSession, catalogue

print([tool.name for tool in catalogue() if tool.installed])
session = AnalysisSession(ForensicImage.from_path("image.jpg"), name="image.jpg")
result = session.run("ela", {"quality": 80})  # cached per (tool, parameters)
tile = session.maps("ela")["heatmap"].tile(level=0, x=0, y=0)
```

Tools that take a user-facing setting declare it as a `ParameterSpec` (name, type, range,
default, description) and accept it as a constructor keyword argument: `ela.quality`,
`localizer_ensemble.mode` and `dinov2_head.attribution` so far. Deployment settings -- a
device, a weights directory -- are deliberately not parameters.

### Views

A view renders the image under one transform and claims nothing: score 0.5, label
`uncertain`, `details["kind"] = "view"`, and the map as its heatmap. Three of them:
`luminance_gradient` (Sobel gradient magnitude of the luma, with an optional pre-blur
`radius`), `noise_residual` (luma minus its local median, `window` 3 or 5) and `bit_planes`
(one bit of the 8-bit luma, `plane` 0-7). They are pure numpy/Pillow and need no extra.

Because a 0.5 card on every image is noise, `imgforensics analyze` skips the views unless one
is named -- `analyze image.jpg --detector luminance_gradient` -- and `imgforensics benchmark`
refuses them outright, there being no verdict to score. `--save-heatmaps` and `--report-dir`
treat a named view like any other tool.

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
