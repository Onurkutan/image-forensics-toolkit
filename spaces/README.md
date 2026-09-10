---
title: imgforensics demo
emoji: "\U0001F50E"
colorFrom: gray
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# imgforensics on a Hugging Face Space

The public stopgap demo of
[image-forensics-toolkit](https://github.com/Onurkutan/image-forensics-toolkit) --
milestone 6c of [`docs/design/01_toolbox_architecture.md`](../docs/design/01_toolbox_architecture.md).
This directory holds everything the Space needs and nothing else; the Space itself is not
created from here, and no image, weight file or model card is published by these files.

## What it runs

`app.py` is a handful of lines over `imgforensics.demo`: it builds the Gradio page with
`build_demo(DemoConfig())` and launches it. The page takes one upload, runs the tools you
tick, and shows a summary (the fused verdict with its abstain band, one score bar per tool,
then the caveats), a gallery of colour-mapped overlays, and the raw JSON behind both. It is
the same service layer the CLI and the API sit on, so a score here is the score
`imgforensics analyze` prints.

Gradio is the right tool for a link anybody can open and the wrong one for the workbench
that replaces this page (milestone 6d): no synchronized pan and zoom across several maps,
no per-tool parameter panels, no tiles. The demo is retired once the workbench is deployed.

## Build

`requirements.txt` installs the project from git with the `ml`, `demo` and `provenance`
extras. The pretrained weights are not part of that install -- nothing third-party is
committed to this repository, and every weight file carries its own license, which the
toolkit prints before a byte moves. Fetch them in the Space's build step:

```bash
imgforensics weights fetch catnet_v2 --accept-license
imgforensics weights fetch iml_vit --accept-license
```

`--accept-license` is the acknowledgement that the printed license has been read: CAT-Net
v2's weights are CC-BY-4.0 (so using them requires attributing CAT-Net) and IML-ViT's are
MIT. Both land under `$IMGFORENSICS_WEIGHTS_DIR`, which the Space should point at a
writable directory that survives the build.

## The AI-generation head is research-only

`dinov2_head` is trained on datasets whose licenses permit research use only, so it is
**not** in this repository and **not** fetched by the command above. It is published
separately, under its own research-only model card, and the Space loads it by pointing
`IMGFORENSICS_HEAD_DIR` at the downloaded checkpoint directory. Without that variable the
detector abstains with a reason instead of failing, and every other tool still runs -- so a
Space that cannot accept those terms is a working Space with one fewer tool.

Set `IMGFORENSICS_FUSER` to a fitted fuser (`imgforensics fusion fit`) for the fused
verdict. Without one, the page simply has no fused line.

## Data

No image is stored. An upload is decoded in memory, analyzed by one
`imgforensics.service.AnalysisSession` that exists for the length of that run, and dropped
when the run ends; nothing is written to disk, and there is no database. (The API next door
keeps its sessions longer, but bounded the same way -- in one process's memory behind a TTL
and a population cap, `imgforensics.service.SessionStore`.) Nothing is logged beyond what
the Space's own runtime records.

## Hardware

Sized for **CPU basic**. Uploads longer than 2048 px on a side are downscaled before
analysis -- a demo-only concession, stated on the page, because CAT-Net's DCT stream over a
12-megapixel photograph is minutes of CPU. The library never resizes: a local run reads the
original pixels. A CPU Space also decodes JPEG coefficients in pure Python, which is the
slowest step in the page and the subject of milestone 6e.

This is a research demo. The toolkit is MIT-licensed, its numbers are calibrated on the
benchmarks in [`docs/benchmarks/`](../docs/benchmarks/), and none of them is evidence about
any particular image.
