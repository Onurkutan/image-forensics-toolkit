# Third-party notices

This file lists every third-party library this repository depends on, its
license, and whether that license permits commercial use
(`commercial_ok`). Third-party models, weights and datasets are not
included in this repository; they will be listed here, with their own
license and `commercial_ok` flag, as they are added in later phases. They
are downloaded by the user from their original source and are never
redistributed by this repository (see `docs/ROADMAP.md`, section 7).

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
not installed, rather than failing.

| Package | Extra | License | commercial_ok |
|---|---|---|---|
| c2pa-python | `provenance` | MIT OR Apache-2.0 | yes |
| huggingface_hub | `data` | Apache-2.0 | yes |
| gdown | `data` | MIT | yes |

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
