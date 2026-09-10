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
| pyarrow | `data` | Apache-2.0 | yes |
| torch | `ml` | BSD-3-Clause | yes |
| torchvision | `ml` | BSD-3-Clause | yes |
| timm | `ml` | Apache-2.0 | yes |
| safetensors | `ml` | Apache-2.0 | yes |

## Model weights

Weights are **never committed to this repository**: each is downloaded from
its original source on first use and cached outside the working tree, and
each is loaded in eval mode with gradients disabled and only ever read.

The two frozen backbones come from the Hugging Face Hub, through `timm`, and
land in the Hub's default cache or in `IMGFORENSICS_WEIGHTS_DIR` when it is
set (point it at the gitignored `weights/` directory to keep everything
inside the project).

| Model | timm id | License | commercial_ok | Source |
|---|---|---|---|---|
| DINOv2 ViT-B/14 | `vit_base_patch14_dinov2.lvd142m` | Apache-2.0 | yes | `facebookresearch/dinov2`; downloaded from the Hugging Face Hub on first use, never committed |
| OpenAI CLIP ViT-L/14 | `vit_large_patch14_clip_224.openai` | MIT | yes | `openai/CLIP`; downloaded from the Hugging Face Hub on first use, never committed |

The localizer weights are fetched by `imgforensics weights fetch NAME
--accept-license`, which prints the license below and refuses to download
without that flag, verifies the sha256 recorded in
`imgforensics.localization.weights`, and writes into
`$IMGFORENSICS_WEIGHTS_DIR/<name>/` or `weights/<name>/` (both gitignored).

| Model | File | License | commercial_ok | Source |
|---|---|---|---|---|
| IML-ViT (CASIAv2-trained release) | `iml-vit_checkpoint.pth`, 350.2 MB, sha256 `7631fe85...26cce4` | MIT (Copyright (c) 2023 Xiaochen Ma) | yes | `SunnyHaze/IML-ViT`, `checkpoints/ckpt_download_page.md` -> Google Drive file id `1xXJGJPW1i5j9Pc1JKd7fJmIAQkvt9jY7`; downloaded on first use, never committed |

## Development dependencies

| Package | License | commercial_ok |
|---|---|---|
| pytest | MIT | yes |
| pytest-cov | MIT | yes |
| ruff | MIT | yes |
| mypy | MIT | yes |
| cryptography | `dev` (tests only) | Apache-2.0 OR BSD-3-Clause | yes |

## Vendored source files

Source vendored (rather than installed as a dependency) always keeps its
original license header, and every file's header lists exactly what was
changed. Each entry below says why the code was copied instead of depended
on.

**`invisible-watermark`**: the upstream package's default backend pulls in
`torch`, `onnxruntime` and `opencv-python`, which would conflict with this
project's `opencv-python-headless` dependency and its CPU-only,
small-dependency design (see `docs/ROADMAP.md`, section 3). Only the pure
numpy/opencv/PyWavelets codec was vendored; the upstream project's GAN-based
backend and its heavier dependencies were not.

**`IML-ViT`**: the upstream project is a research repository, not a
published package, so there is nothing to depend on. Its model definition
imports `fvcore` (for distributed-training batch norm and a weight
initializer) and its data pipeline imports `albumentations`; neither is
needed for inference, and both were removed rather than added as
dependencies. Training code -- the BCE and edge losses, the MAE
initialization hook, the augmentation transforms -- was not vendored either.
Module and parameter names are unchanged, so the released checkpoint loads
with `strict=True`. Parts of the vendored backbone are themselves derived
from `facebookresearch/detectron2` (Apache-2.0), as the upstream file
states; both licenses are permissive and both are recorded in the file
header.

| File | Vendored from | License | commercial_ok |
|---|---|---|---|
| `src/imgforensics/signals/_vendor/dwtdct.py` | `ShieldMnt/invisible-watermark`, `imwatermark/maxDct.py` (`EmbedMaxDct`, the `dwtDct` method) | MIT (see the file's header for the full upstream notice) | yes |
| `src/imgforensics/localization/_vendor/iml_vit/vit.py` | `SunnyHaze/IML-ViT` @ `07dd2be0f4ea27a5c97c9fa5ffbe236733833eac`, `modules/window_attention_ViT.py` | MIT, with parts from `facebookresearch/detectron2` (Apache-2.0) | yes |
| `src/imgforensics/localization/_vendor/iml_vit/decoder.py` | `SunnyHaze/IML-ViT` @ `07dd2be0f4ea27a5c97c9fa5ffbe236733833eac`, `modules/decoderhead.py` | MIT (head design credited upstream to NVlabs/SegFormer) | yes |
| `src/imgforensics/localization/_vendor/iml_vit/model.py` | `SunnyHaze/IML-ViT` @ `07dd2be0f4ea27a5c97c9fa5ffbe236733833eac`, `iml_vit_model.py` | MIT | yes |
