# Design note 01: the interactive forensic toolbox

Status: accepted, 2026-09-10. The project owner chose the **web client** (section 5);
a desktop shell stays a possible later wrapper around the same API.

## 1. What the end product is

The roadmap's Phase 6 currently lists a FastAPI service and a Gradio demo. The product
this project is actually aiming at is an **interactive forensic workbench** in the spirit
of [Sherloq](https://github.com/GuidoBartoli/sherloq): one image loaded once, a tree of
tools grouped by what they look at, each tool opening a view with its own parameters,
every view sharing one pan/zoom with the original so a region can be compared across
tools, and a report exported at the end. The difference from Sherloq is what sits
underneath: calibrated scores with an abstain band, learned detectors and localizers next
to the classical signals, and a benchmark harness that says how much each tool is worth.

Sherloq is GPL-3.0 (PySide2) and vendors non-commercial Noiseprint weights. **No code and
no weights from it can enter this MIT repository.** Its tool catalogue is used here only
as a checklist of ideas; anything adopted is reimplemented from the underlying papers and
cited (section 6).

## 2. Constraints that decide the architecture

1. `imgforensics` stays a headless, pip-installable library. Nothing in it may import a
   GUI or web framework, and the base install stays torch-free (the `ml` extra is optional
   today and remains so).
2. The GUI is a **thin client over one JSON contract**. Every number and every map it shows
   comes from a `DetectionResult` produced through the registry; the client never
   computes forensics itself. This keeps the CLI, the benchmark runner, the report
   builder and the GUI on the same code path, so a benchmark table describes what the
   user sees.
3. Large images are the norm (WildRF medians above 2,000 px; 12-megapixel photographs
   are the stated target). Maps are computed at native resolution but *served*
   downscaled, with zoomed tiles fetched on demand; the report builder's 1,024 px overlay
   cap is the precedent.
4. Weights are never bundled. A deployed instance fetches them through the existing
   license gate (`imgforensics weights fetch --accept-license`) at build time; the UI
   shows a tool as "not installed" rather than failing when its weights are absent, which
   the detectors' abstention contract already provides.

## 3. Proposed layering

```
imgforensics.core / signals / detectors / localization / fusion   (unchanged, headless)
        │
imgforensics.service   pure Python: sessions, tool catalogue, parameters, map tiles
        │
imgforensics.api       FastAPI wrapper over the service (optional extra `api`)
        │
client                 web SPA (recommended) or desktop app, talking JSON + PNG only
```

### 3.1 `imgforensics.service` (no web framework, fully unit-testable)

- `ToolSpec`: registry name, display name, category (section 6), `kind`
  (`signal` | `detector` | `localizer` | `view`), whether it needs the `ml` extra, whether
  its weights are installed, and a list of `ParameterSpec` (name, type, range, default,
  description).
- `ParameterSpec` is the one addition the detector contract needs: a classmethod
  `BaseDetector.parameters() -> list[ParameterSpec]` (default: none) and constructor
  keyword arguments matching it. Tools that already take arguments (`LocalizerEnsemble`'s
  `mode`, `LearnedDetector`'s `attribution`) become the first declared parameters; the
  ELA quality, copy-move block size and JPEG-ghost quality range follow.
- `AnalysisSession`: holds one `ForensicImage`, runs tools lazily on request, caches each
  `DetectionResult` keyed by (tool, parameters), exposes `maps(tool)` as image pyramids
  for tiling, and builds the existing explanation report (`fusion.explain_report`) on
  demand. Sessions are in-memory with a TTL, so a public demo needs no database.
- `views`: tools of kind `view` return a map and no score (score 0.5, label `uncertain`,
  `details["kind"] = "view"`), are excluded from fusion by construction (the fuser lists
  its detectors explicitly), skipped by the default `analyze` run and refused by
  `benchmark` (there is no score to benchmark). They
  are what makes the workbench a workbench: luminance gradient, noise residual, bit
  planes and the like show *something* on every image without claiming a verdict.

### 3.2 `imgforensics.api` (extra `api`: FastAPI + uvicorn)

| Method | Path | Purpose |
|---|---|---|
| POST | `/sessions` | upload an image, get a session id and the image info |
| GET | `/tools` | the tool catalogue with parameters and installed/abstaining state |
| POST | `/sessions/{id}/tools/{name}` | run a tool with parameters, return the `DetectionResult` as JSON plus map URLs |
| GET | `/sessions/{id}/maps/{name}/{z}/{x}/{y}.png` | a tile of a map at pyramid level `z` |
| GET | `/sessions/{id}/fusion` | the fused verdict and contributions from the configured fuser |
| GET | `/sessions/{id}/report.zip` | the explanation report folder |

OpenAPI comes for free and doubles as the client contract. The CLI's `--json` output and
the API's JSON share one serializer (`utils.jsonsafe`), so the two never drift.

### 3.3 Client

A single-page web application: one canvas component that renders the original and up to
N tool views side by side (or as toggled layers) under **one shared transform**, a tool
tree on the left, parameter sliders that re-run a tool with a debounce, score cards with
the abstain band drawn on them, and a report download. Plain TypeScript with a small
framework (Svelte or Preact) is enough; no chart library is needed beyond `<canvas>`.

## 4. Why not just Gradio

Gradio gives a demo in a day and stays the right tool for the Hugging Face Space
*stopgap*. It cannot do synchronized pan/zoom across several views, per-tool parameter
panels that re-run one tool without re-running all, or tiled maps for a 12-megapixel
image. The toolbox needs all three. Plan: Gradio demo first (cheap, public link), the
workbench on the API afterwards; the demo is retired once the workbench is deployed.

## 5. Web or desktop (decided: web)

| | Web client (recommended) | Desktop client (PySide6, LGPL) |
|---|---|---|
| Distribution | one URL; Hugging Face Space or any container | installer per OS; PyInstaller bundles with torch weigh about 2 GB |
| Offline use | no (unless self-hosted) | yes |
| Synced multi-view canvas | `<canvas>` + one transform; well trodden | QGraphicsView; native and fast, more code |
| Large images | tiles from the server; browser memory bounded | native memory; simplest |
| Portfolio value | live demo link, screenshots, anyone can try it | screenshots only |
| Licensing | none beyond MIT | PySide6 is LGPL (fine when not statically linked); PyQt is GPL/commercial and is out |
| Duplication | none: the API is needed for the Space anyway | viewer code that the web demo would not share |

Recommendation: **web first.** It reuses the API the roadmap already promises, gives the
project a public demo, and keeps the client stateless. A desktop shell can wrap the same
API later (a local server plus a webview) if offline field use becomes a real
requirement; that path costs little because the client would not change.

Decision (2026-09-10): web client. Sections 3.1 and 3.2 are where the work starts.

## 6. Tool catalogue

Sherloq's categories, mapped to what exists and what would be added. "Own" means
reimplemented from the literature under MIT; nothing is ported from Sherloq.

| Category | Exists in `imgforensics` | Candidates to add (own implementation) |
|---|---|---|
| General / File | sha256 digest, format, size (`ForensicImage`) | hex header view (`view`) |
| Metadata | EXIF, XMP, editor and AI-generator markers, thumbnail consistency, quantization tables (`metadata`); C2PA manifest verification (`c2pa`) | geolocation card from the EXIF already parsed |
| Inspection | -- | magnifier with contrast stretch, channel histograms (`view`) |
| Detail | -- | luminance gradient, echo-edge filter, frequency split (`view`) |
| Noise | Stable Diffusion watermark decode (`sd_watermark`) | noise residual and min/max deviation maps, bit-plane view (`view`); PRNU camera fingerprint (needs a reference set; later) |
| JPEG | ELA (`ela`), JPEG ghost (`jpeg_ghost`), double compression and blocking grid (`double_jpeg`), quality estimate (`metadata`) | quantization-table matching against a camera/software table database |
| Tampering | block copy-move (`copy_move`); IML-ViT, CAT-Net v2 and their ensemble (`iml_vit`, `catnet_v2`, `localizer_ensemble`) | keypoint copy-move (rotation and scale invariant), resampling detection, contrast-enhancement detection |
| AI generation | DINOv2 head with Grad-CAM attribution (`dinov2_head`), calibrated fusion with abstain band | per-generator-year decay chart in the report |
| Various | -- | median-filtering detection, dead/hot pixel map (`view`) |

Every added `signal` goes through the same gate the existing seven did: unit tests on
self-made fixtures, documented failure modes on laundered images, and a benchmark row. A
`view` needs tests only for determinism and shape.

## 7. Milestones

- **6a. Service layer.** `ParameterSpec`/`parameters()`, `ToolSpec` catalogue,
  `AnalysisSession` with result caching and map pyramids, `view` kind, first three views
  (luminance gradient, noise residual, bit planes). Pure Python, tested without any GUI.
- **6b. API.** FastAPI wrapper with the six routes above, `api` extra, tests through
  `TestClient`, one Dockerfile.
- **6c. Gradio stopgap on a Hugging Face Space.** CPU-only torch, weights fetched at build
  through the license gate, the experiment 02 head published to the Hub under a
  research-only card first.
- **6d. Workbench client.** Upload, tool tree, synced views, sliders, score cards with the
  band, report download. Deployed next to the API; the Gradio demo retired.
- **6e. CPU inference.** ONNX export of the head; profile and vectorize the pure-Python JPEG
  coefficient decoder, which at about 340 ms per 256 px image is the first thing a CPU
  deployment will feel.
- **6f. Desktop shell**, only if offline field use becomes a requirement later.

## 8. Risks

- A CPU Space is slow for CAT-Net's DCT stream on large images; 6e is scheduled before
  the workbench is opened to the public.
- Browser memory on 20-megapixel inputs; tiles and a hard cap on simultaneous views
  (four) bound it.
- Scope creep through the catalogue. Each `view` is cheap, each `signal` is not; the
  benchmark row is the admission ticket for a signal.
- The head's own numbers: experiment 03 shows a 54.7% false-positive rate on unseen
  social-media photographs when WildRF is not in training. The UI must show the abstain
  band and the per-tool caveats, never a bare verdict; that is a product requirement, not
  a footnote.
