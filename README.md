# image-forensics-toolkit

[![CI](https://github.com/Onurkutan/image-forensics-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/Onurkutan/image-forensics-toolkit/actions/workflows/ci.yml)

A toolkit that produces *calibrated evidence* about whether an image is fully AI-generated,
locally AI-edited (inpainting, generative fill) or classically manipulated (splicing,
copy-move) -- with heatmaps, plain-language explanations, and a benchmark harness that says
what each piece of evidence is worth. It is for anyone holding one image who needs to say
something defensible about it, and for anyone who wants to see how such a system is built and
measured. There is no universal "AI or not" verdict for any image, and nothing here claims
one: this toolkit measures, calibrates, and says when it is unsure.

Status: v0.1.0, early development. Every number below comes from
[`docs/benchmarks/`](docs/benchmarks/), the unflattering ones included.

## What you get

- **A workbench.** `imgforensics serve`, then one URL: a tool tree grouped by what each tool
  looks at, one shared pan/zoom across the original and every open map (drawn from 256-pixel
  tiles, so a 12-megapixel heatmap costs a few PNGs instead of 48 MB), parameter sliders that
  re-run a tool 400 ms after they stop moving, per-tool and fused verdicts with the abstain
  band drawn on them, and the report download. Plain HTML, CSS and ES modules -- no build step,
  no Node, no CDN, and a Content-Security-Policy that permits requests to this server only.
- **A CLI.** `imgforensics analyze image.jpg` prints one card per tool; `--report-dir out/`
  writes `report.json`, a `report.md` of plain-language cards, and a heatmap/overlay PNG pair
  per tool that produced a map.
- **14 tools in four kinds.** Seven classical signals (`metadata`, `ela`, `c2pa`,
  `sd_watermark`, `copy_move`, `jpeg_ghost`, `double_jpeg`); three views that render the image
  under one transform and claim no verdict (`luminance_gradient`, `noise_residual`,
  `bit_planes`); one learned whole-image detector (`dinov2_head`, with a Grad-CAM map showing
  where it looked); three localizers (`iml_vit`, `catnet_v2`, `localizer_ensemble`).
- **Calibrated fusion with an abstain band** -- a pure-numpy logistic stacking model over the
  tool scores, fitted on saved benchmark records, allowed to answer "uncertain".
- **A benchmark harness** -- manifests with file hashes and licenses, a class-bias audit, a
  deterministic 15-level robustness suite, per-level pixel metrics, trivial baselines in every
  table, Markdown reports.
- **A demo and an HTTP API** -- one Gradio page and a FastAPI service, both over the same
  headless library the CLI uses.

## Try it in two minutes

```bash
git clone https://github.com/Onurkutan/image-forensics-toolkit.git
cd image-forensics-toolkit
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate

# torch first, from the wheel index your driver supports (cu130 shown; use cpu for CPU-only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[ml,api]"
```

Then fetch the localizer weights. Each command prints the model's license, `commercial_ok`
flag, size and source and refuses to download without `--accept-license`; the file is verified
against a recorded sha256 and lands in the gitignored `weights/`. **No third-party weights are
committed to this repository.**

```bash
imgforensics weights list                              # what is registered, and what is installed
imgforensics weights fetch iml_vit --accept-license    # 350 MB, MIT
imgforensics weights fetch catnet_v2 --accept-license  # 873 MB, CC-BY-4.0
imgforensics serve                                     # workbench at http://127.0.0.1:8000/
```

Or stay on the command line:

```bash
imgforensics analyze image.jpg
imgforensics analyze image.jpg --report-dir out/
imgforensics analyze image.jpg --json --save-heatmaps out/
```

**The AI-generation head is not in this repository.** It is trained on datasets licensed for
research use only, so it is research-only itself and is published separately under its own
model card. Point `IMGFORENSICS_HEAD_DIR` at a downloaded checkpoint, or train your own:

```bash
imgforensics train head --config configs/experiments/02_diverse_reals_augmented_only.yaml
IMGFORENSICS_HEAD_DIR=weights/dinov2_head_02 imgforensics analyze image.jpg
```

Without a checkpoint `dinov2_head` **abstains** -- score 0.5, label `uncertain`, and a
`reason` saying where it looked -- rather than failing the run, and every other tool still
works. A localizer whose weights are missing does the same.

## How good is it, honestly

| What is measured | Test set | Result |
|---|---|---|
| Same-family generators (sanity check) | [Community Forensics val](docs/benchmarks/02_experiment_summary.md), 42 unseen generators, 1,000 images | AUC **1.000** |
| Social media, its own train split seen | [WildRF test](docs/benchmarks/02_experiment_summary.md), 1,000 images | AUC **0.980**, FPR 0.137 at 0.5 |
| Social media, genuinely unseen | [the same 1,000, WildRF removed from training](docs/benchmarks/06_experiment_03_summary.md) | AUC **0.804**, FPR **0.547** |
| ... the same, after fusion with the signals | [experiment 03 fuser](docs/benchmarks/06_experiment_03_summary.md) | AUC 0.831, FPR 0.238; on the 20.2% it will call, balanced accuracy 0.887 |
| Local diffusion edits, image level | [CocoGlide](docs/benchmarks/02_cocoglide.md), 1,024 images | head AUC 0.644 |
| Local diffusion edits, pixel level | [CocoGlide masks](docs/benchmarks/03_cocoglide_catnet.md), 512 images | `catnet_v2` best-F1 **0.605**, [`iml_vit`](docs/benchmarks/03_cocoglide_iml_vit.md) 0.486 |
| Recompression and rescaling | [15-level robustness suite](docs/benchmarks/01_val_dinov2.md) | AUC 1.000 through JPEG q50, 0.567 at quarter scale |

**Same-family generators: 1.000, and it means less than it looks.** On 42 generators the head
never saw, drawn from the same corpus as its training generators, it separates real from
generated perfectly -- a sanity check, not a claim about the open world.
[Experiment 01](docs/benchmarks/01_experiment_summary.md) reached the same 1.000 while calling
99.9% of ordinary COCO photographs fake.

**In-distribution social media: 0.980.** On the WildRF test split -- real and generated images
taken from Reddit, Twitter and Facebook -- the shipped default head reaches AUC 0.980 at a
13.7% false-positive rate on real photographs. WildRF's *train* split was in that head's
training data, so this is a held-out set of images, not a held-out distribution
([experiment 02](docs/benchmarks/02_experiment_summary.md)).

**The genuine cross-dataset test: 0.804, and the failure is specific.**
[Experiment 03](docs/benchmarks/06_experiment_03_summary.md) removes the WildRF train split and
changes nothing else. The head still calls generated images generated (TPR 0.893 at 0.5) but
calls **54.7% of platform-laundered real photographs fake**, while held-out COCO photographs
stay at a 2.6% false-positive rate: it has learned to read platform re-encoding as evidence of
generation. Fusing it with the seven signals cuts the false-positive rate to 23.8% and the
calibration error from 0.216 to 0.044, at the cost of recall (89.3% to 73.8%); the abstain band
then calls 202 of the 1,000 images -- 20.2%, at balanced accuracy 0.887 -- and hands the rest
back as `uncertain`. One image in five answered, right about 89% of the time, is the honest
number here. On the distribution it *has* seen, the same machinery is far stronger:
[fusion 01](docs/benchmarks/04_fusion_wildrf.md) drops the false-positive rate from 13.7% to
3.6% and calls 534 of 1,000 images at balanced accuracy 0.996.

**Local edits: a whole-image head does not see them.** On CocoGlide (authentic COCO images
against the same images locally inpainted by GLIDE) the head reaches AUC 0.644; a few percent
of edited pixels do not move a whole-image score. That is the localizers' job, and they are
only partly up to it: `catnet_v2` reaches pixel best-F1 0.605 and F1@0.5 0.364, `iml_vit` 0.486
and 0.059, against a predict-everything baseline of F1 0.355. `catnet_v2` beats that baseline
at the fixed 0.5 threshold as well as on ranking; `iml_vit` ranks manipulated pixels better
than chance but almost never crosses 0.5 -- a calibration failure on top of a domain-transfer
failure. Read `catnet_v2` as a lower bound: CocoGlide is PNG, so every image goes through a
quality-100 re-encode first and its DCT stream reads a compression history this toolkit
created. Full tables, with AP, IoU and a per-mask-size breakdown, are in
[docs/benchmarks/](docs/benchmarks/).

**Robustness: recompression is survivable, rescaling is not.** Across the 15-level suite the
head holds AUC 1.000 from clean through JPEG quality 50, WEBP, noise, an 80% crop and the
social re-share pipeline, and collapses to 0.567 at quarter-scale resize -- crop-never-resize
has nothing left to crop once the image is smaller than the crop
([`01_val_dinov2.md`](docs/benchmarks/01_val_dinov2.md)).

**What this means for you.** This is a triage and evidence tool, not an oracle: the heatmaps,
the attribution map and the per-tool cards are the product, and the fused score is a summary of
them. A C2PA manifest, when present, is the most decisive thing in the report -- but a missing
one proves nothing, since most images, AI-generated ones included, carry none. Laundering is
the great destroyer: once a platform has re-encoded, resized and stripped an image, the
metadata and JPEG-history signals go blind and the head starts reading the laundering itself,
which is exactly what the 0.804 measures. And a fuser is a calibration layer fitted on one
distribution -- the weights behind these numbers are WildRF's, not universal, so a different
deployment needs `imgforensics fusion fit` run again on its own data.

## How it works

```
core/ signals/ detectors/ localization/   one image in, a DetectionResult out
                 |                        (score, label, details, heatmap, attribution)
              fusion/                     calibrated stacking, abstain band, report builder
                 |
              service/                    sessions, tool catalogue, parameters, map tiles
                 |
                api/                      FastAPI over the service (extra: api)
                 |
              client                      the workbench: HTML, CSS, ES modules, no build
```

Alongside these live `data/` (dataset registry, manifests, bias audit), `eval/` (metrics,
robustness suite, benchmark runner), `views/`, `utils/` and `cli.py`. The reasoning behind the
layering is in [docs/design/01_toolbox_architecture.md](docs/design/01_toolbox_architecture.md).

**Crop, never resize; augment always.** Every image reaches the backbone as native-resolution
224 px crops; anything smaller is reflection-padded, never upscaled, because resizing
low-passes exactly the high-frequency artifacts a detector keys on. Training views pass through
random JPEG, WEBP, blur, downscale-upscale, noise and cut-out
([`configs/augment_default.yaml`](configs/augment_default.yaml)), so the head cannot learn the
training set's own JPEG history. Augmentation runs on a 2x crop window aligned to the image's
16-pixel grid, and the policy is part of the cache key, so switching policies never reuses the
wrong features.

**A frozen backbone and a small head.** Features come once from a frozen DINOv2 ViT-B/14 (a
self-supervised space separates real from generated better than a language-aligned one; CLIP
ViT-L/14 is the registered comparison), read at several depths and cached per image, so
changing the head costs no GPU time. Only a 1.06 M-parameter head trains on top. Training keeps
the best validation AUC, early-stops, then fits a temperature and bias on the validation split
-- a calibration that does not reduce the expected calibration error is not applied, and the
checkpoint says so. The checkpoint also records each training manifest's license and the
`commercial_ok` flag ANDed across them, the only place a training set's licensing survives the
move from data to model. Beside its heatmap the head paints a Grad-CAM map taken on the
*calibrated* logit, so the picture explains the number the report prints -- but **a saliency
map is not a manipulation mask:** normalized within its own image, it ranks pixels against each
other and means nothing across images, and on a fully generated image the evidence can sit
anywhere, empty sky included. For "which pixels were edited", use the localizers.

**Localizers, and a JPEG decoder written from scratch.** `iml_vit` reads RGB pixels only;
`catnet_v2` also reads the JPEG stream's quantized DCT coefficients and quantization table, so
a region pasted in with a different compression history stands out where the pixels look
seamless. Both pad rather than scale, tile above 1024 px at stride 768 instead of truncating
the way upstream does, and score an image as the mean of the top 1% of heatmap values -- a
plain mean shrinks with the edited region, a plain max is one noisy pixel from calling
everything fake. Non-JPEG input to `catnet_v2` is re-encoded to a quality-100 JPEG in memory,
which `details` records so a heatmap is never read without knowing which path produced it.
Reading coefficients needs an entropy decoder, and `jpegio` publishes no Windows wheel and none
past CPython 3.10, so this project decodes baseline and extended-sequential JPEGs in pure
numpy/Python and falls back to the re-encode for progressive, arithmetic and 12-bit files.
**No new dependency was added.**

**Stacking fusion that is allowed to say no.** `fusion fit` fits an L2-regularized logistic
model over saved benchmark records -- no scikit-learn, no torch -- imputing a tool that did
not run as abstaining at 0.5, so a fuser degrades gracefully with a subset of its tools
present. A "real below low / fake above high / uncertain in between" band is fitted on a
held-out split, and the search accepts only a band leaving enough held-out images outside it
(the larger of `--min-outside-count` and `--min-outside-fraction`): a band supported by a
handful of images is a loophole, not a fit. When none meets the target, the best is kept and
`band target met` is reported as false.

**A headless service and a thin client.** `imgforensics.service` computes no forensics; it
holds the state a one-shot command does not need. `catalogue()` lists every tool with its
parameters and whether its weights are installed, answered without importing torch.
`AnalysisSession` loads one image once, runs tools lazily, caches each result under the
parameters that produced it, and serves each map as a tiled pyramid, so a 12-megapixel heatmap
never crosses the wire whole. `SessionStore` keeps sessions in memory behind a TTL and a cap,
so a public deployment needs no database. The API and the workbench consume that same
contract, which is why a benchmark table describes what the user sees. The service is importable
directly -- `AnalysisSession(ForensicImage.from_path(path)).run("ela", {"quality": 80})` -- with
no web framework anywhere near it.

## Reference

### Tools and their blind spots

A signal's score is the probability the image is generated or manipulated (0 = confidently
real, 1 = confidently fake); the label is `real`, `fake` or `uncertain`. **No signal alone
should be read as a verdict** -- see `details` for the evidence. Every metadata- and
JPEG-domain signal below shares one blind spot: it decays to uninformative once a platform has
re-encoded, resized and stripped the image, which is why the table lists only what each one
misses *in addition*. The literature behind each is in
[docs/research/03_signals_provenance_evaluation.md](docs/research/03_signals_provenance_evaluation.md).

| Signal | Reads | Blind spots beyond laundering |
|---|---|---|
| `metadata` | EXIF/XMP, PNG text chunks, Photoshop APP13, the embedded EXIF thumbnail, JPEG quantization tables; high on a known AI-generator or editor marker, or a thumbnail/image mismatch | Markers are only as good as the list of known tool names; with nothing usable left it abstains at 0.5 rather than guessing |
| `ela` | Re-encodes at a fixed JPEG quality and diffs against the original | Bounded to [0.3, 0.65] -- an explanation aid, never a standalone verdict; blind on an already-uniformly-recompressed image |
| `c2pa` | Any embedded C2PA manifest (`provenance` extra); high when signed as AI-generated, or when the signed content hash no longer matches | A missing manifest proves nothing -- most images, AI-generated ones included, carry none. An untrusted or self-signed signer scores 0.5, not "fake" |
| `sd_watermark` | Decodes the DWT-DCT ("dwtDct") invisible watermark Stable Diffusion reference pipelines embed, against known payloads | Embeds in chroma only, so it survives neither a JPEG re-encode (even at quality 100, through chroma subsampling) nor a resize; covers the two reference-pipeline payloads, not every SD fork or later generator |
| `copy_move` | Block-matching search (quantized zig-zag DCT features) for a region duplicated inside the same image, with a matched-region heatmap | Blind to a clone rotated or rescaled before pasting; repetitive textures (tiles, fences) are guarded by a minimum-distance rule and a shift-consistency check but are not impossible to fool. Finding no duplicate is not evidence of authenticity |
| `jpeg_ghost` | Re-saves at a range of qualities and finds, per block, the quality whose re-save error is anomalously low compared to the rest (Farid's JPEG ghosts), with a heatmap | Bounded to [0.3, 0.7] like `ela` -- an explanation aid, never a verdict; needs JPEG-derived pixels |
| `double_jpeg` | Whether the JPEG 8x8 blocking grid still starts at the image origin, plus a secondary-phase check for a masked older grid underneath (crop/composite detection); and, on JPEG input, whether the DCT coefficient histogram -- normalized by the file's own quantization step, so it measures a genuine second, coarser compression rather than how lossy the current one is -- shows aligned double-quantization periodicity | Quantization-history fingerprints, not proof of malicious editing. The histogram check catches only a *coarser-then-finer* compression (the other orders leave no detectable comb) and only on JPEG input; double compression is normal for any re-shared image, and a misaligned grid only proves a crop-then-resave happened |

The three views (`luminance_gradient`: Sobel magnitude of the luma, optional pre-blur `radius`;
`noise_residual`: luma minus its local median, `window` 3 or 5; `bit_planes`: one bit of the
8-bit luma, `plane` 0-7) are pure numpy/Pillow and return score 0.5, label `uncertain`,
`details["kind"] = "view"`. Because a 0.5 card on every image is noise, `analyze` skips them
unless one is named with `--detector`, and `benchmark` refuses them outright.

| Localizer | Model, trained on | Cue | License (code / weights) |
|---|---|---|---|
| `iml_vit` | [IML-ViT](https://github.com/SunnyHaze/IML-ViT) (arXiv:2307.14863), CASIA v2 | RGB pixels only | MIT / MIT |
| `catnet_v2` | [CAT-Net v2](https://github.com/mjkwon2021/CAT-Net) (WACV 2021 / IJCV 2022), CASIAv2 + FantasticReality + IMD2020 + tampCOCO + compRAISE | RGB pixels **and** the JPEG stream's DCT coefficients and quantization table | Apache-2.0 / CC-BY-4.0 |
| `localizer_ensemble` | both of the above, heatmaps combined pixelwise | both | as above |

The ensemble drops a member whose weights are absent and abstains when none is left.
`IMGFORENSICS_LOCALIZER_ENSEMBLE_MODE` picks `mean` (default -- both members emit calibrated
probabilities, so a 0.5 threshold on their mean still means what it means for either alone),
`max` (finds a region one member missed, inheriting the other's false positives) or
`rank_mean`, which equalizes members on different scales but **discards both members'
calibration**: a rank map is uniform on [0, 1] by construction, so a 0.5 threshold marks the
upper half of any image, leaving pixel AP and best-F1 meaningful but not F1@0.5 or IoU@0.5.
The ensemble owns its member instances, so benchmarking it beside them loads each model twice
-- on a 6 GB card, give it its own run. Per-model VRAM and timing on this project's RTX 2060 are
recorded in [docs/benchmarks/](docs/benchmarks/) and in
[docs/ROADMAP.md](docs/ROADMAP.md)'s phase 4a result notes.

### Commands

```bash
# analyze
imgforensics analyze IMAGE [--json] [--detector NAME ...] [--save-heatmaps DIR] [--report-dir DIR] [--fuser fuser.json]
imgforensics version

# datasets, manifests, bias audit
imgforensics datasets list | show NAME | recipe NAME
imgforensics datasets fetch NAME --dest DIR --accept-license [--dry-run] [--variant full]
imgforensics datasets prepare NAME --src DIR --out manifest.jsonl
imgforensics datasets materialize "Community Forensics" --src DIR --out TREE [--max-rows N]
imgforensics manifest build ROOT --dataset NAME --out manifest.jsonl
imgforensics manifest sample IN --n N --out OUT
imgforensics manifest merge A B ... --out OUT
imgforensics manifest split IN --out-train TRAIN --out-val VAL [--by FIELD] [--val-fraction F] [--holdout GROUP]
imgforensics manifest crop IN --out-dir DIR --out OUT --size N --mode center|tiles [--label real ...]
imgforensics audit manifest.jsonl [--strict]

# benchmark
imgforensics benchmark manifest.jsonl [--detector NAME ...] [--all-signals] [--baselines] \
    [--robustness default|localization|PATH|none] [--limit N] [--root DIR] \
    [--out results.json] [--report report.md] [--workers N]

# learned detector
imgforensics features extract manifest.jsonl --cache-dir data/features [--backbone NAME] \
    [--crop-mode center|grid|random] [--max-crops 4] [--views 2] \
    [--augment configs/augment_default.yaml] [--workers N] [--device auto]
imgforensics features info --cache-dir data/features
imgforensics train head --config configs/head_dinov2.yaml [--epochs N] [--out DIR]
imgforensics weights list
imgforensics weights fetch NAME --accept-license

# fusion
imgforensics fusion fit results.json [more.json ...] --out weights/fuser.json \
    [--level NAME] [--detector NAME ...] [--target-bacc 0.9] \
    [--min-outside-count 20] [--min-outside-fraction 0.10]
imgforensics fusion info weights/fuser.json
imgforensics fusion eval results.json --fuser weights/fuser.json [--report eval.md]

# serve, demo, tests
imgforensics serve [--host 127.0.0.1] [--port 8000] [--fuser PATH] [--ttl-seconds 1800] [--max-sessions 32]
imgforensics demo [--host 127.0.0.1] [--port 7860] [--fuser PATH] [--tool NAME ...] [--max-side 2048] [--share]
pytest
```

An end-to-end pass -- benchmark, fit a fuser on the records, use it:

```bash
imgforensics benchmark manifest.jsonl --all-signals --out results.json --robustness none
imgforensics fusion fit results.json --out weights/fuser.json
imgforensics analyze image.jpg --fuser weights/fuser.json   # or set IMGFORENSICS_FUSER
imgforensics fusion eval results.json --fuser weights/fuser.json --report eval.md
```

A manifest is JSON Lines of labeled images (path, label, source, generator, split, mask path,
sha256, resolution, format, JPEG quality) plus a `*.meta.json` sidecar (dataset name, license,
`commercial_ok`). `datasets fetch` prints the license, refuses to proceed without
`--accept-license`, and never asks for or stores Kaggle or Hugging Face credentials -- a
dataset needing one prints instructions and stops. `manifest crop` writes native-resolution
crops, never a resize, closing a resolution gap between classes that would otherwise let a
detector learn scene scale instead of generation artifacts. `benchmark` tunes each threshold on
the manifest's val split, or marks the report "threshold tuned in-sample"; a score must be
*strictly above* a threshold to count as a fake call, so a tool abstaining at 0.5 is not scored
as calling every image fake. Pixel metrics are recorded only at levels that preserve the pixel
grid (`clean`, JPEG, WEBP, noise), since a mask is stored against the original geometry, and
`--robustness localization` selects exactly that half of the default suite
([`robustness_default.yaml`](src/imgforensics/eval/robustness_default.yaml);
metrics in [`imgforensics.eval.metrics`](src/imgforensics/eval/metrics.py)).

### Environment variables

| Variable | Effect |
|---|---|
| `IMGFORENSICS_HEAD_DIR` | Where `dinov2_head` looks for its checkpoint (default `weights/dinov2_head/`, gitignored) |
| `IMGFORENSICS_WEIGHTS_DIR` | Where fetched weights and the Hub backbone cache live (default `weights/<name>/`; the DINOv2 backbone is about 350 MB, fetched on first use) |
| `IMGFORENSICS_FUSER` | Default fuser for `analyze`, `serve` and `demo`, with `weights/fuser.json` as the last fallback |
| `IMGFORENSICS_HEAD_ATTRIBUTION` | `0`/`false`/`no`/`off` skips the Grad-CAM map. On by default, costing one extra forward and backward pass over the crops the score already used -- 27 ms to 149 ms per image on this project's RTX 2060, no second model, nothing to download. Score, label and heatmap come from the untouched no-grad path either way, so switching it changes no number |
| `IMGFORENSICS_LOCALIZER_ENSEMBLE_MODE` | `mean` (default), `max` or `rank_mean` for `localizer_ensemble` |

### Extras

| Extra | Pulls in | Needed for |
|---|---|---|
| `ml` | `torch`, `torchvision`, `timm`, `safetensors` | `dinov2_head`, `iml_vit`, `catnet_v2`, feature extraction, training |
| `api` | `fastapi`, `uvicorn`, `python-multipart` | `imgforensics serve` and the workbench |
| `demo` | `gradio` | `imgforensics demo` |
| `provenance` | `c2pa-python` | the `c2pa` signal |
| `data` | `huggingface_hub`, `gdown`, `pyarrow` | `datasets fetch` and `datasets materialize` |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `mypy`, `httpx`, `cryptography` | running the test suite |

The base install is torch-free and web-framework-free, and everything else imports and runs
without any extra. The `weights` commands work without `ml`: fetching a model and being able to
run it are separate problems. The `features` commands print an install hint and exit 1 when
`ml` is missing.

### API routes

`imgforensics serve` puts the workbench client at `/` and the JSON API under it, schema at
`/docs`.

| Method | Path | What it does |
|---|---|---|
| GET | `/` | the workbench client; its files are served from `/static` |
| GET | `/health` | `status` and the version whose contract this server speaks |
| GET | `/tools` | the catalogue: every tool with its parameters and its `installed` flag |
| POST | `/sessions` | upload one image (multipart field `file`), get a session id |
| GET | `/sessions/{id}` | the image's size and format, and the tools run so far |
| POST | `/sessions/{id}/tools/{name}` | run a tool with `{"parameters": {...}}`, get its result and its map URLs |
| GET | `/sessions/{id}/maps/{name}/{map}/{z}/{x}/{y}.png` | one 256-pixel tile of a map, as 8-bit grayscale PNG |
| GET | `/sessions/{id}/fusion` | the fused verdict, when a fuser is configured |
| GET | `/sessions/{id}/report.zip` | the explanation report folder, zipped |

```bash
BASE=http://127.0.0.1:8000
SESSION=$(curl -sF file=@image.jpg $BASE/sessions | python -c "import json,sys; print(json.load(sys.stdin)['id'])")
curl -s -X POST $BASE/sessions/$SESSION/tools/ela \
  -H 'Content-Type: application/json' -d '{"parameters": {"quality": 80}}'
curl -s -o tile.png $BASE/sessions/$SESSION/maps/ela/heatmap/0/0/0.png
```

A tile URL names a tool and a map, never the parameters: it serves the map of that tool's last
run, so changing a setting leaves a viewer's tile URLs valid. An unknown or expired session and
an unregistered tool are 404; a refused parameter or an out-of-range tile is 422, carrying the
message the service wrote. Without a fuser the fused-verdict route is a 404 and everything else
works. Two caveats worth stating plainly: **there is no authentication**, and **sessions are
in-memory** -- one process, a TTL and a population cap, so a restart loses them and two
replicas do not share them. Hence the loopback default; anything network-reachable wants a
reverse proxy in front of it.

### Demo

`imgforensics demo` is one Gradio page over the same service layer -- a link anybody can open,
deliberately thin, retired once the workbench is deployed. Upload, tick tools, press Analyze:
a summary (fused verdict with its band, one score bar per tool, then the caveats -- every tool
that abstained with its reason, every tool that failed with its exception), a gallery of the
same overlays `--report-dir` writes, and the JSON behind both. Views start unchecked; a tool
whose weights are missing is greyed out rather than hidden; a tool that raises becomes an error
card instead of taking the page down. One caveat: **uploads longer than 2048 px on a side are
downscaled** (`--max-side`), and the page says so, because a public CPU deployment cannot run
CAT-Net over a 12-megapixel photograph. The library itself never resizes, so `analyze` reads
the original pixels; below the cap the upload's own encoded file is analyzed, so `metadata`,
`c2pa` and the JPEG-history half of `double_jpeg` behave as they do from the CLI.
[`spaces/`](spaces/) holds what a Hugging Face Space needs to run this page.

## Data and licensing

This is a personal, non-commercial research project. That widens what it may *use* but not what
it may *redistribute*: the repository is MIT-licensed, so anything committed must be
MIT-compatible. **No third-party weights, images or datasets are committed here.** They are
downloaded from their original sources through a gate that prints the license and requires
explicit acceptance, and they keep their own terms.

- Every dataset and model entry carries a `commercial_ok` flag, and a checkpoint inherits that
  flag ANDed across its training manifests, so the licensing provenance of any trained model
  can be traced back to its data. The flag is bookkeeping, not a plan: the project is and
  stays non-commercial.
- Research-only components stay isolated in a clearly marked optional extra used for
  benchmarking only; the default install holds only permissively licensed pieces. The
  `dinov2_head` checkpoint trained here is research-only for that reason, and is published
  separately under its own model card rather than shipped.
- TGIF/TGIF2 is CC BY-SA 4.0, and that ShareAlike clause is why no image set derived from it is
  redistributed. The in-house inpainting set is built on COCO images with mixed Flickr
  licenses, so only masks, prompts, image IDs and scripts are released -- never the images.
- Using `catnet_v2`'s weights requires **attributing CAT-Net** (CC-BY-4.0). The exact wording,
  with every external model, dataset and library and its license, is in
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The full policy table, the disk plan and the list of what is deliberately never downloaded are
in [docs/ROADMAP.md](docs/ROADMAP.md), section 7.

## Roadmap and research notes

[docs/ROADMAP.md](docs/ROADMAP.md) is the build plan and the running results log, phase by
phase. Runbooks for individual experiments are in [docs/experiments/](docs/experiments/), the
benchmark tables in [docs/benchmarks/](docs/benchmarks/), the workbench design decision in
[docs/design/01_toolbox_architecture.md](docs/design/01_toolbox_architecture.md), and the three
literature surveys the plan derives from in [docs/research/](docs/research/).

## License

The code in this repository is released under the MIT license, see [LICENSE](LICENSE).
Third-party models, weights and datasets are not included in the repository; they are
downloaded from their original sources and keep their own licenses, some of which permit
research use only. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the current library
list; models and datasets are added there as they are integrated.
