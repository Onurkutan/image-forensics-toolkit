# Third-party notices

This file lists every third-party library this repository depends on, its
license, and whether that license permits commercial use
(`commercial_ok`). Third-party models, weights and datasets are not
included in this repository; the ones already integrated are listed under
"Model weights" below, with their own license and `commercial_ok` flag, and
further ones are added as later phases integrate them. They are downloaded
by the user from their original source and are never redistributed by this
repository (see `docs/ROADMAP.md`, section 7).

## Runtime dependencies

| Package | License | commercial_ok |
|---|---|---|
| numpy | BSD-3-Clause | yes |
| pillow | MIT-CMU (PIL Software License) | yes |
| opencv-python-headless | Apache-2.0 | yes |
| typer | MIT | yes |
| rich | MIT | yes |
| pydantic | MIT | yes |
| piexif | MIT | yes |
| PyWavelets | MIT | yes |
| PyYAML | MIT | yes |

## Optional runtime dependencies

Installed with the named extra; the corresponding signal degrades to an
"uncertain" abstain with a `details["reason"]` explanation when the extra is
not installed, rather than failing. The `ml` extra is the exception to that
pattern: it backs the learned detectors rather than a signal, so the
`imgforensics features` commands print an install hint and exit 1 when it is
missing (`imgforensics.detectors.is_ml_available()` reports the same thing in
code).

| Package | Extra | License | commercial_ok |
|---|---|---|---|
| c2pa-python | `provenance` | MIT OR Apache-2.0 | yes |
| huggingface_hub | `data` | Apache-2.0 | yes |
| gdown | `data` | MIT | yes |
| torch | `ml` | BSD-3-Clause | yes |
| torchvision | `ml` | BSD-3-Clause | yes |
| timm | `ml` | Apache-2.0 | yes |
| safetensors | `ml` | Apache-2.0 | yes |

## Model weights

Weights are **never committed to this repository**: each is downloaded from
the Hugging Face Hub on first use and cached outside the working tree (the
Hub's default cache, or `IMGFORENSICS_WEIGHTS_DIR` when set -- point it at
the gitignored `weights/` directory to keep everything inside the project).
Both are loaded frozen, through `timm`, and only ever read.

| Model | timm id | License | commercial_ok | Source |
|---|---|---|---|---|
| DINOv2 ViT-B/14 | `vit_base_patch14_dinov2.lvd142m` | Apache-2.0 | yes | `facebookresearch/dinov2`; downloaded from the Hugging Face Hub on first use, never committed |
| OpenAI CLIP ViT-L/14 | `vit_large_patch14_clip_224.openai` | MIT | yes | `openai/CLIP`; downloaded from the Hugging Face Hub on first use, never committed |

## Development dependencies

| Package | License | commercial_ok |
|---|---|---|
| pytest | MIT | yes |
| pytest-cov | MIT | yes |
| ruff | MIT | yes |
| mypy | MIT | yes |
| cryptography | `dev` (tests only) | Apache-2.0 OR BSD-3-Clause | yes |

## Vendored source files

Vendored (not installed as a dependency) because the upstream package's
default backend pulls in `torch`, `onnxruntime` and `opencv-python`, which
would conflict with this project's `opencv-python-headless` dependency and
its CPU-only, small-dependency design (see `docs/ROADMAP.md`, section 3).
Only the pure numpy/opencv/PyWavelets codec was vendored; the upstream
project's GAN-based backend and its heavier dependencies were not.

| File | Vendored from | License | commercial_ok |
|---|---|---|---|
| `src/imgforensics/signals/_vendor/dwtdct.py` | `ShieldMnt/invisible-watermark`, `imwatermark/maxDct.py` (`EmbedMaxDct`, the `dwtDct` method) | MIT (see the file's header for the full upstream notice) | yes |
