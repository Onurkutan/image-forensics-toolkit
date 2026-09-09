# Experiment 01: DINOv2 head on Community Forensics Small

The first real-data run of the learned detector (`dinov2_head`, Phase 3): train the
multi-layer head on a slice of Community Forensics Small, evaluate on a
generator-disjoint held-out slice, and check it against an in-the-wild test set. Every
command below already exists in `imgforensics`; nothing here needs new code.

**License note.** Community Forensics is CC-BY-4.0 *for research purposes only*
(`commercial_ok: false` in the registry). A head trained on it is research-only --
`head.json` records this automatically (see README, "The checkpoint records its
training data's licenses").

## Steps

### 1. Fetch Community Forensics (bounded to 8 shards)

```bash
imgforensics datasets fetch "Community Forensics" --dest data/raw --accept-license
```

Uses the default recipe step (`max_files: 8`), not `--variant full` -- the packaged
note already bounds this to the first 8 sorted Parquet shards instead of the full
~260 GB `-Small` repository. **Disk:** about 23 GB for the 8 shards (1 GB to 4 GB each; about 24,000 images, with
512x512 fakes and 1024x1024 reals, see step 3a); read `datasets recipe "Community Forensics"`
for the exact plan before running.

### 2. Materialize the Parquet shards into an image tree

```bash
imgforensics datasets materialize "Community Forensics" \
    --src data/raw/"Community Forensics"/data --out data/materialized/cf
```

Writes `<real|fake>/<generator>/<split>/<name>.<ext>` plus `attributes.jsonl` and
`materialize.json` (rows read/written/skipped, per-label and per-generator counts,
formats seen). **Disk:** roughly the same order as step 1's download -- the row
bytes are written out unchanged, so budget for the total to roughly double.

### 3. Build and audit a manifest

```bash
imgforensics datasets prepare "Community Forensics" \
    --src data/materialized/cf --out data/manifests/cf.jsonl --strict-audit
```

`prepare` picks up `generator`/`split` from `attributes.jsonl` (falling back to the
folder-derived values), then runs the bias audit automatically. Read the printed
report before continuing -- a format or resolution skew between real and fake here
is the classic self-deception this project's audit exists to catch (README, "Data
and evaluation"). **Disk:** negligible (a JSON Lines file plus a report).

### 4. Split into generator-disjoint train/val

```bash
imgforensics manifest split data/manifests/cf.jsonl \
    --out-train data/manifests/train.jsonl --out-val data/manifests/val.jsonl \
    --by generator --val-fraction 0.2 --seed 0
```

No generator appears in both halves; real images (no generator) are split
proportionally so both halves have reals too. The printed group lists are what "seen
generators" (val) vs. the training set mean below -- keep them for the report. To
additionally hold out specific generators regardless of size, add
`--holdout NAME [NAME ...]`. **Disk:** negligible.

### 5. Optional: extra real images from COCO

```bash
imgforensics datasets fetch COCO --dest data/raw --variant val_only --accept-license
imgforensics datasets prepare COCO --src data/raw/COCO --out data/manifests/coco.jsonl
imgforensics manifest merge data/manifests/train.jsonl data/manifests/coco.jsonl \
    --out data/manifests/train.jsonl
```

Worth doing if step 3's audit flagged a real/fake imbalance or a real-image
diversity gap. **Disk:** about 1 GB (`val2017.zip`, per the packaged recipe note).

### 6. Extract cached features

```bash
imgforensics features extract data/manifests/train.jsonl \
    --cache-dir data/features --views 2 --augment configs/augment_default.yaml
imgforensics features extract data/manifests/val.jsonl \
    --cache-dir data/features
```

Val is cached un-augmented (view 0 only) on purpose -- `train head` would extract it
that way anyway, but pre-extracting here separates the backbone-time cost from the
head-training step. **Disk:** cannot be estimated ahead of time -- it depends on
image count x views x crops x layers x dim x 2 bytes (float16). Run with `--limit
200` first and read `features info --cache-dir data/features` to extrapolate a
per-image byte cost, then scale to the full manifest.

### 7. Train the head

```bash
imgforensics train head --config configs/head_dinov2.yaml
```

Edit `configs/head_dinov2.yaml`'s `train_manifest`/`val_manifest` paths first (they
are placeholders pointing at `data/manifests/train.jsonl` / `val.jsonl`, which step 4
already produced under those exact names). **Disk:** small -- the head is about
1.06M parameters, so `head.safetensors` plus `head.json` and `training_log.jsonl`
land well under 10 MB in `weights/dinov2_head/`.

### 8. Benchmark

On the val manifest. Because step 4 split by generator, every generator in val is
unseen by the head: this is the held-out-generator number, with the caveat that the
real images on both sides come from the same sources and the fake images from the
same dataset family (the same 512x512 pipeline). ITW-SM below is the cross-dataset
check that removes that caveat:

```bash
imgforensics benchmark data/manifests/val.jsonl \
    --detector dinov2_head --all-signals --baselines --robustness default \
    --report docs/benchmarks/01_val.md --out docs/benchmarks/01_val.json
```

Then fetch and prepare ITW-SM as an in-the-wild test:

```bash
imgforensics datasets fetch ITW-SM --dest data/raw --accept-license
imgforensics datasets prepare ITW-SM --src data/raw/ITW-SM --out data/manifests/itw_sm.jsonl
imgforensics benchmark data/manifests/itw_sm.jsonl \
    --detector dinov2_head --all-signals --baselines --robustness default \
    --report docs/benchmarks/01_itw_sm.md --out docs/benchmarks/01_itw_sm.json
```

ITW-SM has no train/val split of its own, so both reports carry the "threshold tuned
in-sample" caveat unless val's tuned threshold is reused manually. **Disk:** ITW-SM
is under 5 GB (registry estimate); the two JSON/Markdown report pairs are a few MB
each at most.

## What to report

Once both `benchmark` runs and `train head` have completed, `docs/benchmarks/`
should carry enough to fill in:

- **AUC and balanced accuracy at 0.5** on val (held-out generators, same dataset
  family; `01_val.md`'s image-metrics table, `fixed` row) and on ITW-SM (cross-dataset,
  in the wild; `01_itw_sm.md`). Both reports carry the in-sample-threshold caveat for
  the `tuned` row; quote the `fixed` row.
- **The robustness table** from each report (`## Robustness`).
- **Per-generator AUC** from val's `## Per-generator AUC` table.
- **The head's layer weights**, printed by `train head` and stored in
  `head.json`'s `head` section.
- **Calibration ECE before -> after**, printed by `train head` (`ECE before ->
  after`) and stored in `head.json`'s `val.ece_before`/`val.ece_after`.
