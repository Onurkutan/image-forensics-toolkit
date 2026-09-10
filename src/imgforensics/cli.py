"""Command-line interface for imgforensics."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import numpy as np
import typer
from PIL import Image
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from imgforensics import __version__, detectors
from imgforensics.core import registry
from imgforensics.core.image import ForensicImage
from imgforensics.core.types import DetectionResult
from imgforensics.data import (
    LicenseNotAcceptedError,
    Manifest,
    audit_manifest,
    build_manifest,
    crop_entries,
    fetch,
    get_dataset,
    get_recipe,
    label_from_parent_folder,
    load_registry,
    materialize_parquet,
    merge,
    prepare,
    sample,
    split_by_group,
)
from imgforensics.detectors import BACKBONES, CropPolicy, FeatureCache, FeatureExtractor
from imgforensics.detectors.features import extract_to_cache
from imgforensics.eval.preprocess import AugmentationConfig
from imgforensics.eval.records import ScoreRecord
from imgforensics.eval.robustness import RobustnessSuite
from imgforensics.eval.runner import BenchmarkConfig, BenchmarkResult, run_benchmark
from imgforensics.fusion.explain_report import ReportPaths, build_report
from imgforensics.fusion.report import explain
from imgforensics.fusion.stacking import Fuser, fit_fuser
from imgforensics.localization import (  # also registers the iml_vit localizer
    WEIGHTS,
    fetch_weights,
    weights_file,
)
from imgforensics.signals import SIGNAL_NAMES  # also registers all seven signal detectors
from imgforensics.utils.image_io import image_hash
from imgforensics.utils.jsonsafe import to_jsonable

app = typer.Typer(help="Detect AI-generated images, AI-inpainted regions, and manipulations.")
datasets_app = typer.Typer(help="Browse the external dataset registry.")
manifest_app = typer.Typer(help="Build dataset manifests.")
features_app = typer.Typer(help="Extract and cache frozen-backbone features.")
train_app = typer.Typer(help="Train the learned detectors' heads.")
weights_app = typer.Typer(help="Fetch pretrained model weights, after a license gate.")
fusion_app = typer.Typer(help="Fit and inspect calibrated stacking fusers.")
app.add_typer(datasets_app, name="datasets")
app.add_typer(manifest_app, name="manifest")
app.add_typer(features_app, name="features")
app.add_typer(train_app, name="train")
app.add_typer(weights_app, name="weights")
app.add_typer(fusion_app, name="fusion")
console = Console()
# A wider, fixed-width console for the dataset-registry tables: names and
# license strings are long enough that the default (terminal-detected, often
# 80-column) width truncates them illegibly, especially when stdout is not a
# real terminal (e.g. under CliRunner in tests, or piped output).
_registry_console = Console(width=160)

_DETAIL_STRING_LIMIT = 80
_DEFAULT_FEATURE_CACHE = Path("data/features")
_BYTES_PER_MIB = 1024 * 1024
_FUSER_PATH_ENV = "IMGFORENSICS_FUSER"
_DEFAULT_FUSER_PATH = Path("weights/fuser.json")


def _resolve_fuser_path(explicit: Path | None) -> Path | None:
    """Resolve the fuser to use for ``analyze``: ``--fuser``, then the env var, then the default.

    The default path is only used when the file actually exists (so
    ``analyze`` behaves identically to today when no fuser has ever been
    fitted); an explicit ``--fuser`` or ``$IMGFORENSICS_FUSER`` is expected
    to point at a real file and is not silently ignored if it does not.
    """
    if explicit is not None:
        return explicit
    env_value = os.environ.get(_FUSER_PATH_ENV)
    if env_value:
        return Path(env_value)
    if _DEFAULT_FUSER_PATH.is_file():
        return _DEFAULT_FUSER_PATH
    return None


_TOP_CONTRIBUTIONS_SHOWN = 5


def _fusion_payload(fuser_model: Fuser, results: list[DetectionResult]) -> dict[str, Any]:
    """The fused verdict for one ``analyze`` run: probability, label, band, and contributions."""
    scores = {result.detector: result.score for result in results}
    probability = fuser_model.predict(scores)
    label = fuser_model.predict_label(scores)
    contributions = explain(fuser_model, scores)
    return {
        "probability": probability,
        "label": label,
        "band": {"low": fuser_model.band.low, "high": fuser_model.band.high},
        "contributions": [
            {
                "detector": contribution.detector,
                "score": contribution.score,
                "weight": contribution.weight,
                "contribution": contribution.contribution,
                "present": contribution.present,
                "note": contribution.note,
            }
            for contribution in contributions
        ],
    }


def _print_fusion_panel(fusion: dict[str, Any]) -> None:
    contributions_table = Table(show_header=True, box=None)
    contributions_table.add_column("detector", style="bold")
    contributions_table.add_column("score", justify="right")
    contributions_table.add_column("contribution", justify="right")
    contributions_table.add_column("note")
    for item in fusion["contributions"][:_TOP_CONTRIBUTIONS_SHOWN]:
        score_text = f"{item['score']:.2f}" + ("" if item["present"] else " (missing)")
        contributions_table.add_row(
            item["detector"], score_text, f"{item['contribution']:+.3f}", escape(item["note"])
        )

    band = fusion["band"]
    panel = Panel(
        contributions_table,
        title=(
            f"Fused verdict  probability={fusion['probability']:.2f}  {fusion['label']}  "
            f"(band [{band['low']:.2f}, {band['high']:.2f}])"
        ),
        title_align="left",
    )
    console.print(panel)


def _truncate(text: str, limit: int = _DETAIL_STRING_LIMIT) -> str:
    if len(text) > limit:
        return text[: limit - 1] + "\N{HORIZONTAL ELLIPSIS}"
    return text


def _save_heatmap(heatmap: np.ndarray, out_dir: Path, stem: str, detector_name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_{detector_name}.png"
    array_8bit = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(array_8bit, mode="L").save(out_path)
    return out_path


def _resolve_detector_names(selected: list[str] | None) -> list[str]:
    available_names = registry.available()
    if not selected:
        return available_names
    unknown = [name for name in selected if name not in available_names]
    if unknown:
        raise typer.BadParameter(
            f"Unknown detector(s): {', '.join(unknown)}. "
            f"Available: {', '.join(available_names) or '<none>'}"
        )
    return selected


@app.command()
def analyze(
    path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Image file to analyze",
        ),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print a single JSON document instead of tables/panels."),
    ] = False,
    save_heatmaps: Annotated[
        Path | None,
        typer.Option(
            "--save-heatmaps",
            help="Directory to write each detector's heatmap as an 8-bit grayscale PNG to.",
        ),
    ] = None,
    detector: Annotated[
        list[str] | None,
        typer.Option("--detector", help="Run only this detector (repeatable). Default: all."),
    ] = None,
    fuser: Annotated[
        Path | None,
        typer.Option(
            "--fuser",
            help=(
                "fuser.json to fuse detector scores into one verdict with. "
                f"Default: ${_FUSER_PATH_ENV}, else {_DEFAULT_FUSER_PATH} if it exists, else none."
            ),
        ),
    ] = None,
    report_dir: Annotated[
        Path | None,
        typer.Option(
            "--report-dir",
            help=(
                "Directory to build an explanation report in: report.json, report.md, and a "
                "heatmap/overlay PNG pair per detector with a heatmap. Works with or without "
                "--fuser."
            ),
        ),
    ] = None,
) -> None:
    """Analyze a single image and print per-detector results."""
    forensic_image = ForensicImage.from_path(path)
    digest = image_hash(forensic_image.rgb)
    names = _resolve_detector_names(detector)

    fuser_path = _resolve_fuser_path(fuser)
    loaded_fuser: Fuser | None = None
    if fuser_path is not None:
        if not fuser_path.is_file():
            raise typer.BadParameter(f"Fuser file not found: {fuser_path}")
        loaded_fuser = Fuser.load(fuser_path)

    results: list[DetectionResult] = []
    heatmap_paths: dict[str, Path | None] = {}
    for name in names:
        detector_cls = registry.get(name)
        instance = detector_cls()
        instance.load()
        result = instance.run(forensic_image)
        results.append(result)

        saved_path: Path | None = None
        if save_heatmaps is not None and result.heatmap is not None:
            saved_path = _save_heatmap(result.heatmap, save_heatmaps, path.stem, name)
        heatmap_paths[name] = saved_path

    fusion_payload = _fusion_payload(loaded_fuser, results) if loaded_fuser is not None else None

    report_paths: ReportPaths | None = None
    if report_dir is not None:
        report_paths = build_report(
            forensic_image,
            results,
            fuser=loaded_fuser,
            out_dir=report_dir,
            source_name=path.name,
        )

    if json_output:
        document = {
            "file": str(path),
            "width": forensic_image.width,
            "height": forensic_image.height,
            "results": [
                {
                    "detector": result.detector,
                    "score": result.score,
                    "label": result.label,
                    "elapsed_ms": result.elapsed_ms,
                    "details": to_jsonable(result.details),
                    "heatmap": (
                        str(heatmap_paths[result.detector])
                        if heatmap_paths[result.detector] is not None
                        else None
                    ),
                }
                for result in results
            ],
        }
        if fusion_payload is not None:
            document["fusion"] = to_jsonable(fusion_payload)
        if report_paths is not None:
            document["report_dir"] = str(report_paths.out_dir)
        typer.echo(json.dumps(document, indent=2))
        return

    info_table = Table(title="Image info")
    info_table.add_column("Field")
    info_table.add_column("Value")
    info_table.add_row("file", path.name)
    info_table.add_row("size", f"{forensic_image.width}x{forensic_image.height}")
    info_table.add_row("format", forensic_image.format or "unknown")
    info_table.add_row("sha256", digest[:12])
    console.print(info_table)

    if not names:
        console.print("No detectors registered yet.")
        raise typer.Exit(code=0)

    for result in results:
        details_table = Table(show_header=False, box=None)
        details_table.add_column("Field", style="bold")
        details_table.add_column("Value")
        for key, value in result.details.items():
            if value is None:
                continue
            details_table.add_row(key, escape(_truncate(str(to_jsonable(value)))))
        heatmap_path = heatmap_paths[result.detector]
        if heatmap_path is not None:
            details_table.add_row("heatmap", str(heatmap_path))
        elapsed = f"{result.elapsed_ms:.1f}" if result.elapsed_ms is not None else "-"
        details_table.add_row("elapsed_ms", elapsed)

        panel = Panel(
            details_table,
            title=f"{result.detector}  score={result.score:.2f}  {result.label}",
            title_align="left",
        )
        console.print(panel)

    if fusion_payload is not None:
        _print_fusion_panel(fusion_payload)

    if report_paths is not None:
        console.print(f"Report written to {report_paths.out_dir}")


@app.command()
def version() -> None:
    """Print the installed imgforensics version."""
    console.print(__version__)


@app.command()
def audit(
    manifest_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Manifest .jsonl file."),
    ],
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Exit with code 1 if the audit finds any problems."),
    ] = False,
) -> None:
    """Audit a manifest's real/fake halves for format, resolution, quality, and duplicate bias."""
    manifest = Manifest.load(manifest_path)
    report = audit_manifest(manifest, strict=False)
    console.print(Markdown(report.to_markdown()))
    if strict and not report.ok:
        raise typer.Exit(code=1)


@app.command()
def benchmark(
    manifest_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Manifest .jsonl file."),
    ],
    detector: Annotated[
        list[str] | None,
        typer.Option("--detector", help="Registered detector name to include (repeatable)."),
    ] = None,
    all_signals: Annotated[
        bool,
        typer.Option("--all-signals", help="Include every classical signal detector."),
    ] = False,
    baselines: Annotated[
        bool,
        typer.Option(
            "--baselines", help="Include the trivial baselines (constant, random, signals_mean)."
        ),
    ] = False,
    robustness: Annotated[
        str,
        typer.Option(
            "--robustness",
            help="'default' (packaged suite), a YAML suite path, or 'none' (clean level only).",
        ),
    ] = "default",
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Evaluate only the first N entries after a seeded shuffle."),
    ] = None,
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Dataset root; defaults to the manifest's own recorded root."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Path to write the full results (score records) as JSON to."),
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Path to write the Markdown report to; stdout when omitted."),
    ] = None,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            help=(
                "Run the classical signal detectors in this many worker processes "
                "(default 1: sequential). Learned detectors and signals_mean always "
                "run in the main process."
            ),
        ),
    ] = 1,
) -> None:
    """Evaluate registered detectors over a manifest and report Markdown tables."""
    manifest = Manifest.load(manifest_path)

    names = set(detector or [])
    if all_signals:
        # The signals only: the registry can also hold the learned detector,
        # which --detector dinov2_head selects explicitly.
        names |= set(SIGNAL_NAMES)
    unknown = sorted(names - set(registry.available()))
    if unknown:
        raise typer.BadParameter(
            f"Unknown detector(s): {', '.join(unknown)}. "
            f"Available: {', '.join(registry.available()) or '<none>'}"
        )
    if not names and not baselines:
        raise typer.BadParameter(
            "No detectors selected: pass --detector, --all-signals, and/or --baselines."
        )

    if robustness == "default":
        suite = RobustnessSuite.default()
    elif robustness == "none":
        suite = None
    else:
        suite = RobustnessSuite.from_yaml(Path(robustness))

    config = BenchmarkConfig(
        detectors=sorted(names),
        include_baselines=baselines,
        robustness=suite,
        limit=limit,
        workers=workers,
    )
    result = run_benchmark(manifest, config, root=root, progress=False)

    if out is not None:
        result.save_json(out)
        console.print(f"Wrote {len(result.records)} score record(s) to {out}")

    markdown_report = result.to_markdown()
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(markdown_report, encoding="utf-8")
        console.print(f"Wrote report to {report}")
    else:
        console.print(Markdown(markdown_report))


@datasets_app.command("list")
def datasets_list() -> None:
    """Print every registered dataset as a table."""
    table = Table(title="Dataset registry")
    table.add_column("name", no_wrap=True)
    table.add_column("task")
    table.add_column("access")
    table.add_column("license")
    table.add_column("commercial_ok")
    table.add_column("approx_gb", justify="right")
    table.add_column("verified")
    for info in load_registry():
        table.add_row(
            info.name,
            info.task,
            info.access,
            _truncate(info.license, 40),
            "-" if info.commercial_ok is None else str(info.commercial_ok),
            "-" if info.approx_size_gb is None else f"{info.approx_size_gb:g}",
            str(info.verified),
        )
    _registry_console.print(table)


@datasets_app.command("show")
def datasets_show(
    name: Annotated[str, typer.Argument(help="Exact dataset name, as printed by 'datasets list'.")],
) -> None:
    """Print every field of one dataset registry entry."""
    try:
        info = get_dataset(name)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(show_header=False, box=None, title=info.name, title_justify="left")
    table.add_column("field", style="bold")
    table.add_column("value")
    for field_name, value in info.model_dump().items():
        table.add_row(field_name, "-" if value is None else escape(str(value)))
    _registry_console.print(table)


@datasets_app.command("recipe")
def datasets_recipe(
    name: Annotated[str, typer.Argument(help="Exact dataset name, as printed by 'datasets list'.")],
) -> None:
    """Print the acquisition recipe (download steps) for one registered dataset."""
    try:
        recipe = get_recipe(name)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc

    console.print(f"[bold]Recipe: {recipe.dataset}[/bold]")
    if recipe.subset_note:
        console.print(recipe.subset_note)
    console.print("")
    console.print("[bold]steps:[/bold]")
    for step in recipe.steps:
        console.print(step.model_dump_json(indent=2, exclude_none=True))
    if recipe.variants:
        for variant_name, steps in recipe.variants.items():
            console.print(f"[bold]variant '{variant_name}':[/bold]")
            for step in steps:
                console.print(step.model_dump_json(indent=2, exclude_none=True))


@datasets_app.command("fetch")
def datasets_fetch(
    name: Annotated[str, typer.Argument(help="Exact dataset name, as printed by 'datasets list'.")],
    dest: Annotated[Path, typer.Option("--dest", help="Destination directory.")],
    variant: Annotated[
        str | None, typer.Option("--variant", help="Named alternative step list from the recipe.")
    ] = None,
    accept_license: Annotated[
        bool,
        typer.Option(
            "--accept-license", help="Accept the printed license and proceed with the download."
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Print the license and download plan only; download nothing."
        ),
    ] = False,
) -> None:
    """Download a registered dataset per its acquisition recipe, after a license gate."""
    try:
        report = fetch(
            name,
            dest,
            variant=variant,
            accept_license=accept_license,
            dry_run=dry_run,
            progress=True,
        )
    except LicenseNotAcceptedError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title=f"Fetch report: {name}")
    table.add_column("step")
    table.add_column("status")
    table.add_column("bytes", justify="right")
    for step in report.steps:
        table.add_row(escape(step.description), step.status, str(step.bytes_downloaded))
    console.print(table)


@datasets_app.command("prepare")
def datasets_prepare(
    name: Annotated[
        str,
        typer.Argument(
            help="Dataset name; used for the manifest source and, if registered, layout."
        ),
    ],
    src: Annotated[
        Path,
        typer.Option(
            "--src", exists=True, file_okay=False, readable=True, help="Downloaded dataset root."
        ),
    ],
    out: Annotated[Path, typer.Option("--out", help="Path to write the manifest (.jsonl) to.")],
    strict_audit: Annotated[
        bool,
        typer.Option("--strict-audit", help="Exit with code 1 if the bias audit finds problems."),
    ] = False,
) -> None:
    """Build a manifest for a registered (or ad-hoc) dataset using its layout, and audit it."""
    manifest, skipped, report = prepare(name, src, out)
    console.print(f"Wrote {len(manifest.entries)} entries to {out}")
    if skipped:
        console.print(f"[yellow]{len(skipped)} file(s) skipped (unreadable):[/yellow]")
        for line in skipped[:20]:
            console.print(f"  {line}")
        if len(skipped) > 20:
            console.print(f"  ... and {len(skipped) - 20} more")
    console.print(Markdown(report.to_markdown()))
    if strict_audit and not report.ok:
        raise typer.Exit(code=1)


@datasets_app.command("materialize")
def datasets_materialize(
    name: Annotated[
        str,
        typer.Argument(
            help=(
                "Dataset name (currently only 'Community Forensics' ships a Parquet materializer)."
            )
        ),
    ],
    src: Annotated[
        Path,
        typer.Option(
            "--src",
            exists=True,
            file_okay=False,
            readable=True,
            help="Directory containing the downloaded *.parquet shards.",
        ),
    ],
    out: Annotated[
        Path, typer.Option("--out", help="Destination folder tree to materialize into.")
    ],
    max_rows: Annotated[
        int | None,
        typer.Option("--max-rows", help="Stop after this many rows total, across all shards."),
    ] = None,
) -> None:
    """Materialize Community Forensics' Parquet shards into a real/fake image tree."""
    if name != "Community Forensics":
        console.print(
            f"[yellow]No Parquet materializer is registered for {name!r}; "
            "only 'Community Forensics' is supported today.[/yellow]"
        )
    report = materialize_parquet(src, out, max_rows=max_rows, progress=True)

    table = Table(title=f"Materialize report: {name}")
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("rows read", str(report.rows_read))
    table.add_row("written", str(report.written))
    table.add_row("skipped", json.dumps(report.skipped))
    table.add_row("by_label", json.dumps(report.by_label))
    table.add_row("by_generator", json.dumps(report.by_generator))
    table.add_row("formats", json.dumps(report.formats))
    console.print(table)
    console.print(f"Wrote {out / 'materialize.json'} and {out / 'attributes.jsonl'}")


@weights_app.command("list")
def weights_list() -> None:
    """Print every registered set of pretrained weights, and whether it is installed."""
    table = Table(title="Model weights")
    table.add_column("name", style="bold")
    table.add_column("model")
    table.add_column("license")
    table.add_column("commercial", justify="center")
    table.add_column("size (MB)", justify="right")
    table.add_column("installed")
    for name, spec in sorted(WEIGHTS.items()):
        path = weights_file(name)
        table.add_row(
            name,
            spec.model or "-",
            spec.license,
            {True: "yes", False: "no", None: "?"}[spec.commercial_ok],
            f"{spec.size_mb:.1f}" if spec.size_mb is not None else "-",
            str(path) if path.is_file() else "[dim]no[/dim]",
        )
    _registry_console.print(table)


@weights_app.command("fetch")
def weights_fetch(
    name: Annotated[str, typer.Argument(help="Weights name, as printed by 'weights list'.")],
    dest: Annotated[
        Path | None,
        typer.Option(
            "--dest",
            help="Base directory; defaults to IMGFORENSICS_WEIGHTS_DIR, then weights/.",
        ),
    ] = None,
    accept_license: Annotated[
        bool,
        typer.Option(
            "--accept-license", help="Accept the printed license and proceed with the download."
        ),
    ] = False,
) -> None:
    """Download one set of pretrained weights into the weights directory."""
    try:
        report = fetch_weights(name, dest, accept_license=accept_license, progress=True)
    except LicenseNotAcceptedError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title=f"Weights: {name}")
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("status", report.status)
    table.add_row("path", str(report.path))
    table.add_row("bytes", str(report.bytes_downloaded))
    table.add_row("sha256", report.sha256 or "-")
    console.print(table)
    if report.status == "failed":
        raise typer.Exit(code=1)


def _require_ml() -> None:
    """Exit with a clear message when the optional ``ml`` extra is missing."""
    if not detectors.is_ml_available():
        console.print(
            "[red]The 'ml' extra (torch, timm) is not installed, so features cannot be "
            "extracted.[/red]\n"
            'Install it with: pip install -e ".[ml]"  '
            "(add --index-url https://download.pytorch.org/whl/cu130 for a CUDA build; "
            'see the README section "Learned detectors").'
        )
        raise typer.Exit(code=1)


@features_app.command("extract")
def features_extract(
    manifest_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Manifest .jsonl file."),
    ],
    cache_dir: Annotated[
        Path,
        typer.Option("--cache-dir", help="Directory to read and write cached features in."),
    ] = _DEFAULT_FEATURE_CACHE,
    backbone: Annotated[
        str,
        typer.Option("--backbone", help="Frozen backbone name (see 'features info')."),
    ] = "dinov2_vitb14",
    crop_mode: Annotated[
        str, typer.Option("--crop-mode", help="Crop policy mode: center, grid, or random.")
    ] = "grid",
    max_crops: Annotated[int, typer.Option("--max-crops", help="Maximum crops per image.")] = 4,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Process only the first N manifest entries.")
    ] = None,
    device: Annotated[
        str, typer.Option("--device", help="auto, cpu, cuda, or an explicit torch device.")
    ] = "auto",
    augment: Annotated[
        Path | None,
        typer.Option(
            "--augment",
            exists=True,
            dir_okay=False,
            readable=True,
            help=(
                "AugmentationConfig YAML applied to views above 0 "
                "(e.g. configs/augment_default.yaml)."
            ),
        ),
    ] = None,
    views: Annotated[
        int,
        typer.Option("--views", help="Views per image; view 0 is un-augmented, 1..K-1 augmented."),
    ] = 1,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            help=(
                "Decode, EXIF-transpose, window-cut and augment images across this many "
                "worker processes; the backbone forward pass always stays in one process. "
                "1 (the default) decodes serially, as before this option existed."
            ),
        ),
    ] = 1,
) -> None:
    """Extract frozen-backbone features for every manifest image, caching them on disk."""
    _require_ml()

    if backbone not in BACKBONES:
        raise typer.BadParameter(
            f"Unknown backbone {backbone!r}. Available: {', '.join(sorted(BACKBONES))}"
        )
    if crop_mode not in ("center", "grid", "random"):
        raise typer.BadParameter(f"Unknown crop mode {crop_mode!r}. Use center, grid, or random.")
    if views < 1:
        raise typer.BadParameter(f"--views must be at least 1, got {views}.")
    if views > 1 and augment is None:
        raise typer.BadParameter(
            "--views above 1 without --augment would cache identical copies of view 0; "
            "pass --augment configs/augment_default.yaml."
        )

    manifest = Manifest.load(manifest_path)
    entries = manifest.entries[:limit] if limit is not None else manifest.entries
    root = Path(manifest.meta.root)
    paths = [root / entry.path for entry in entries]

    spec = BACKBONES[backbone]
    policy = CropPolicy.model_validate(
        {"size": spec.input_size, "mode": crop_mode, "max_crops": max_crops}
    )
    augmentation = AugmentationConfig.from_yaml(augment) if augment is not None else None
    extractor = FeatureExtractor(
        backbone=backbone,
        device=device,
        crop_policy=policy,
        augment=augmentation,
        views=views,
    )
    cache = FeatureCache(cache_dir)

    console.print(
        f"Extracting {backbone} features for {len(paths)} image(s) x {views} view(s) "
        f"into {cache_dir} (crop mode {crop_mode}, up to {max_crops} crops of {spec.input_size}px)"
    )
    summary = extract_to_cache(paths, cache, extractor, progress=True)
    cache.write_index()

    table = Table(title="Feature extraction")
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("device", str(extractor.device))
    table.add_row("images", str(summary["images"]))
    table.add_row("views", str(summary["views"]))
    table.add_row("feature arrays", str(summary["arrays"]))
    table.add_row("cache hits (skipped)", str(summary["cache_hits"]))
    table.add_row("feature shape", str(summary["feature_shape"]))
    table.add_row("elapsed", f"{float(summary['elapsed_s']):.2f} s")
    table.add_row("throughput", f"{float(summary['images_per_s']):.2f} images/s")
    table.add_row("cache size", f"{float(summary['cache_bytes']) / _BYTES_PER_MIB:.2f} MiB")
    console.print(table)


@features_app.command("info")
def features_info(
    cache_dir: Annotated[
        Path, typer.Option("--cache-dir", help="Feature cache directory to summarize.")
    ] = _DEFAULT_FEATURE_CACHE,
) -> None:
    """Print how many cached feature files the cache holds, per backbone."""
    _require_ml()

    cache = FeatureCache(cache_dir)
    counts = cache.stats()
    table = Table(title=f"Feature cache: {cache_dir}")
    table.add_column("backbone")
    table.add_column("images", justify="right")
    for name in sorted(counts):
        table.add_row(name, str(counts[name]))
    table.add_row("total", str(sum(counts.values())))
    console.print(table)
    console.print(f"Cache size: {cache.size_bytes() / _BYTES_PER_MIB:.2f} MiB")


@train_app.command("head")
def train_head_command(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Training config YAML (see configs/head_dinov2.yaml).",
        ),
    ],
    epochs: Annotated[
        int | None, typer.Option("--epochs", help="Override the config's epoch count.")
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Override the config's output directory.")
    ] = None,
) -> None:
    """Train a head on cached frozen-backbone features and write a calibrated checkpoint."""
    _require_ml()

    # Imported here, not at module scope: this is the one command that needs
    # torch, and paying its import on every other command would be wasteful.
    from imgforensics.detectors.learned import HEAD_DIR_ENV
    from imgforensics.detectors.train import TrainConfig, train_head

    config = TrainConfig.from_yaml(config_path)
    overrides: dict[str, Any] = {}
    if epochs is not None:
        overrides["epochs"] = epochs
    if out is not None:
        overrides["out_dir"] = out
    if overrides:
        config = config.model_copy(update=overrides)

    for manifest_field in (config.train_manifest, config.val_manifest):
        if not Path(manifest_field).is_file():
            raise typer.BadParameter(
                f"Manifest {manifest_field} does not exist. Build one with "
                "'imgforensics manifest build' or 'imgforensics datasets prepare'."
            )

    console.print(
        f"Training a {config.backbone} head on {config.train_manifest} "
        f"({config.views} view(s), up to {config.epochs} epochs) -> {config.out_dir}"
    )
    report = train_head(config, progress=True)
    meta = report.meta

    table = Table(title="Training report")
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("device", config.device)
    table.add_row("head parameters", f"{report.head_parameters:,}")
    table.add_row("train crops / images", f"{report.train_crops} / {report.train_images}")
    table.add_row("train views", meta.train_views)
    table.add_row("val crops / images", f"{report.val_crops} / {report.val_images}")
    table.add_row("epochs run (best)", f"{meta.epochs_run} ({meta.best_epoch})")
    table.add_row("val AUC", f"{meta.val.auc:.4f}")
    table.add_row("val balanced acc @0.5", f"{meta.val.balanced_accuracy:.4f}")
    table.add_row(
        "val balanced acc @tuned",
        f"{meta.val.balanced_accuracy_tuned:.4f} (t={meta.val.threshold:.3f})",
    )
    table.add_row("ECE before -> after", f"{meta.val.ece_before:.4f} -> {meta.val.ece_after:.4f}")
    table.add_row(
        "calibration", f"T={meta.calibration.temperature:.3f}, b={meta.calibration.bias:.3f}"
    )
    table.add_row("layer weights", ", ".join(f"{value:.3f}" for value in report.layer_weights))
    table.add_row("commercial_ok", "-" if meta.commercial_ok is None else str(meta.commercial_ok))
    table.add_row("licenses", ", ".join(meta.licenses) or "-")
    table.add_row("elapsed", f"{report.elapsed_s:.2f} s")
    console.print(table)
    console.print(f"Wrote {report.weights_path}, {report.metadata_path} and {report.log_path}")
    console.print(
        f"Use it with: set {HEAD_DIR_ENV}={report.out_dir} "
        "(or copy it to weights/dinov2_head), then run "
        "'imgforensics analyze IMAGE --detector dinov2_head'"
    )


@manifest_app.command("build")
def manifest_build(
    root: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True, help="Dataset root directory."),
    ],
    dataset: Annotated[
        str, typer.Option("--dataset", help="Dataset name recorded as each entry's source.")
    ],
    out: Annotated[Path, typer.Option("--out", help="Path to write the manifest (.jsonl) to.")],
    license_: Annotated[
        str | None,
        typer.Option("--license", help="License string recorded in the manifest metadata."),
    ] = None,
    commercial_ok: Annotated[
        bool | None,
        typer.Option(
            "--commercial-ok/--no-commercial-ok",
            help="Whether this dataset slice may be used commercially (default: unset/unknown).",
        ),
    ] = None,
) -> None:
    """Build a manifest from ROOT, labeling each image by its parent folder name."""
    manifest, skipped = build_manifest(
        root,
        dataset=dataset,
        label_of=label_from_parent_folder,
        license=license_,
        commercial_ok=commercial_ok,
        progress=False,
    )
    manifest.save(out)
    console.print(f"Wrote {len(manifest.entries)} entries to {out}")
    if skipped:
        console.print(f"[yellow]{len(skipped)} file(s) skipped (unreadable):[/yellow]")
        for line in skipped[:20]:
            console.print(f"  {line}")
        if len(skipped) > 20:
            console.print(f"  ... and {len(skipped) - 20} more")


@manifest_app.command("sample")
def manifest_sample(
    in_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Input manifest .jsonl."),
    ],
    n: Annotated[int, typer.Option("--n", help="Target sample size.")],
    out: Annotated[Path, typer.Option("--out", help="Path to write the sampled manifest to.")],
    seed: Annotated[int, typer.Option("--seed", help="Seed for the deterministic sample.")] = 0,
    stratify: Annotated[
        str,
        typer.Option(
            "--stratify", help="Comma-separated ManifestEntry field names to stratify by."
        ),
    ] = "label,generator",
) -> None:
    """Draw a deterministic, proportional stratified sample of N entries from a manifest."""
    manifest = Manifest.load(in_path)
    fields = tuple(field.strip() for field in stratify.split(",") if field.strip())
    sampled = sample(manifest, n, seed=seed, stratify_by=fields)
    sampled.save(out)
    console.print(f"Wrote {len(sampled.entries)} of {len(manifest.entries)} entries to {out}")


@manifest_app.command("merge")
def manifest_merge(
    inputs: Annotated[
        list[Path],
        typer.Argument(
            exists=True, dir_okay=False, readable=True, help="Manifest .jsonl files to merge."
        ),
    ],
    out: Annotated[Path, typer.Option("--out", help="Path to write the merged manifest to.")],
) -> None:
    """Concatenate several manifests into one (see imgforensics.data.manifest.merge)."""
    manifests = [Manifest.load(path) for path in inputs]
    merged = merge(manifests)
    merged.save(out)
    console.print(f"Wrote {len(merged.entries)} entries (from {len(manifests)} manifests) to {out}")


@manifest_app.command("split")
def manifest_split(
    in_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Input manifest .jsonl."),
    ],
    out_train: Annotated[
        Path, typer.Option("--out-train", help="Path to write the train-half manifest to.")
    ],
    out_val: Annotated[
        Path, typer.Option("--out-val", help="Path to write the val-half manifest to.")
    ],
    by: Annotated[
        str,
        typer.Option("--by", help="ManifestEntry field to split by, group-disjoint."),
    ] = "generator",
    val_fraction: Annotated[
        float,
        typer.Option(
            "--val-fraction", help="Target fraction of grouped (and of ungrouped) entries in val."
        ),
    ] = 0.2,
    holdout: Annotated[
        list[str] | None,
        typer.Option("--holdout", help="Group name to force entirely into val (repeatable)."),
    ] = None,
    seed: Annotated[
        int, typer.Option("--seed", help="Seed for the deterministic group shuffle/assignment.")
    ] = 0,
) -> None:
    """Split a manifest into group-disjoint train/val halves (see manifest.split_by_group)."""
    manifest = Manifest.load(in_path)
    train, val = split_by_group(
        manifest, group_field=by, val_fraction=val_fraction, seed=seed, holdout=holdout
    )
    train.save(out_train)
    val.save(out_val)

    def _groups(side: Manifest) -> list[str]:
        values = (getattr(entry, by) for entry in side.entries)
        return sorted({value for value in values if value is not None})

    console.print(f"Wrote {len(train.entries)} train entries to {out_train}")
    console.print(f"  {by} groups: {', '.join(_groups(train)) or '<none>'}")
    console.print(f"Wrote {len(val.entries)} val entries to {out_val}")
    console.print(f"  {by} groups: {', '.join(_groups(val)) or '<none>'}")


@manifest_app.command("crop")
def manifest_crop(
    in_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Input manifest .jsonl."),
    ],
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Directory to write cropped/copied images into.")
    ],
    out: Annotated[Path, typer.Option("--out", help="Path to write the new manifest to.")],
    size: Annotated[int, typer.Option("--size", help="Crop edge length in pixels.")],
    mode: Annotated[
        str,
        typer.Option(
            "--mode", help="'center' (one crop per image) or 'tiles' (every non-overlapping tile)."
        ),
    ] = "center",
    label: Annotated[
        list[str] | None,
        typer.Option(
            "--label", help="Only crop entries with this label (repeatable); default: every label."
        ),
    ] = None,
    min_side: Annotated[
        int | None,
        typer.Option(
            "--min-side",
            help="Stricter floor than --size for crop eligibility; smaller images pass through.",
        ),
    ] = None,
) -> None:
    """Crop a manifest to a native resolution, no resampling (see manifest.crop_entries)."""
    if mode not in ("center", "tiles"):
        raise typer.BadParameter(f"--mode must be 'center' or 'tiles', got {mode!r}.")
    unknown_labels = sorted({name for name in (label or []) if name not in ("real", "fake")})
    if unknown_labels:
        raise typer.BadParameter(f"Unknown label(s): {', '.join(unknown_labels)}. Use real/fake.")

    manifest = Manifest.load(in_path)
    labels = cast("list[Literal['real', 'fake']] | None", label) if label else None
    cropped_manifest, report = crop_entries(
        manifest,
        out_dir,
        size=size,
        mode=cast('Literal["center", "tiles"]', mode),
        labels=labels,
        min_side=min_side,
        progress=False,
    )
    cropped_manifest.save(out)

    table = Table(title="Crop report")
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("cropped images", str(report.cropped))
    table.add_row("tiles written", str(report.tiles_written))
    table.add_row("passthrough", str(report.passthrough))
    table.add_row("copied", str(report.copied))
    table.add_row("skipped (unreadable)", str(report.skipped_unreadable))
    table.add_row("bytes written", f"{report.bytes_written:,}")
    console.print(table)
    console.print(f"Wrote {len(cropped_manifest.entries)} entries to {out}")
    console.print(json.dumps(cropped_manifest.summary(), indent=2))


@fusion_app.command("fit")
def fusion_fit(
    records_paths: Annotated[
        list[Path],
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="BenchmarkResult JSON file(s), as written by 'imgforensics benchmark --out'.",
        ),
    ],
    out: Annotated[Path, typer.Option("--out", help="Path to write the fitted fuser.json to.")],
    level: Annotated[
        list[str] | None,
        typer.Option(
            "--level",
            help="Robustness level to include (repeatable). Default: every level present.",
        ),
    ] = None,
    detector: Annotated[
        list[str] | None,
        typer.Option(
            "--detector",
            help="Detector to fuse (repeatable). Default: every non-baseline detector present.",
        ),
    ] = None,
    target_bacc: Annotated[
        float,
        typer.Option(
            "--target-bacc",
            help="Target balanced accuracy outside the abstain band.",
        ),
    ] = 0.9,
) -> None:
    """Fit a calibrated stacking fuser on one or more saved benchmark results."""
    records: list[ScoreRecord] = []
    source_hashes: dict[str, str] = {}
    for records_path in records_paths:
        result = BenchmarkResult.load_json(records_path)
        records.extend(result.records)
        source_hashes[str(records_path)] = hashlib.sha256(records_path.read_bytes()).hexdigest()

    try:
        fitted = fit_fuser(
            records,
            detectors=detector or None,
            levels=level or None,
            target_balanced_accuracy=target_bacc,
            records_sha256=source_hashes,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    fitted.save(out)

    table = Table(title="Fuser fit report")
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("detectors", ", ".join(fitted.detectors))
    table.add_row(
        "images (fake / real)",
        f"{fitted.fit_info.n_images} ({fitted.fit_info.n_fake} / {fitted.fit_info.n_real})",
    )
    table.add_row("sources", ", ".join(fitted.fit_info.sources) or "-")
    table.add_row("levels", ", ".join(fitted.fit_info.levels) or "-")
    table.add_row("train AUC", f"{fitted.metrics.train_auc:.4f}")
    table.add_row("held-out AUC", f"{fitted.metrics.holdout_auc:.4f}")
    table.add_row(
        "ECE before -> after", f"{fitted.metrics.ece_before:.4f} -> {fitted.metrics.ece_after:.4f}"
    )
    table.add_row("temperature", f"{fitted.temperature:.3f}")
    table.add_row("band [low, high]", f"[{fitted.band.low:.3f}, {fitted.band.high:.3f}]")
    table.add_row("abstain rate", f"{fitted.metrics.abstain_rate:.3f}")
    table.add_row(
        "outside-band balanced accuracy", f"{fitted.metrics.outside_band_balanced_accuracy:.4f}"
    )
    console.print(table)
    console.print(f"Wrote {out}")


@fusion_app.command("info")
def fusion_info(
    fuser_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="fuser.json to inspect."),
    ],
) -> None:
    """Print a fitted fuser's detectors, weights, abstain band, and fit metrics."""
    fitted = Fuser.load(fuser_path)

    table = Table(title=f"Fuser: {fuser_path}")
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("format_version", str(fitted.format_version))
    table.add_row("created", fitted.created)
    table.add_row("package_version", fitted.package_version)
    table.add_row("bias", f"{fitted.bias:.4f}")
    table.add_row("temperature", f"{fitted.temperature:.4f}")
    table.add_row("band [low, high]", f"[{fitted.band.low:.3f}, {fitted.band.high:.3f}]")
    table.add_row(
        "images (fake / real)",
        f"{fitted.fit_info.n_images} ({fitted.fit_info.n_fake} / {fitted.fit_info.n_real})",
    )
    table.add_row("sources", ", ".join(fitted.fit_info.sources) or "-")
    table.add_row("levels", ", ".join(fitted.fit_info.levels) or "-")
    table.add_row("train AUC", f"{fitted.metrics.train_auc:.4f}")
    table.add_row("held-out AUC", f"{fitted.metrics.holdout_auc:.4f}")
    table.add_row(
        "ECE before -> after", f"{fitted.metrics.ece_before:.4f} -> {fitted.metrics.ece_after:.4f}"
    )
    table.add_row("abstain rate", f"{fitted.metrics.abstain_rate:.3f}")
    table.add_row(
        "outside-band balanced accuracy", f"{fitted.metrics.outside_band_balanced_accuracy:.4f}"
    )
    table.add_row("records_sha256", json.dumps(fitted.records_sha256))
    console.print(table)

    weights_table = Table(title="Per-detector weights")
    weights_table.add_column("detector")
    weights_table.add_column("logit weight", justify="right")
    weights_table.add_column("presence weight", justify="right")
    for index, name in enumerate(fitted.detectors):
        weights_table.add_row(
            name, f"{fitted.logit_weights[index]:.4f}", f"{fitted.presence_weights[index]:.4f}"
        )
    console.print(weights_table)


if __name__ == "__main__":
    app()
