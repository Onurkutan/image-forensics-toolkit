"""Command-line interface for imgforensics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from PIL import Image
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

import imgforensics.signals  # noqa: F401  (side effect: registers all seven signals)
from imgforensics import __version__
from imgforensics.core import registry
from imgforensics.core.image import ForensicImage
from imgforensics.core.types import DetectionResult
from imgforensics.data import (
    LicenseNotAcceptedError,
    Manifest,
    audit_manifest,
    build_manifest,
    fetch,
    get_dataset,
    get_recipe,
    label_from_parent_folder,
    load_registry,
    merge,
    prepare,
    sample,
)
from imgforensics.eval.robustness import RobustnessSuite
from imgforensics.eval.runner import BenchmarkConfig, run_benchmark
from imgforensics.utils.image_io import image_hash

app = typer.Typer(help="Detect AI-generated images, AI-inpainted regions, and manipulations.")
datasets_app = typer.Typer(help="Browse the external dataset registry.")
manifest_app = typer.Typer(help="Build dataset manifests.")
app.add_typer(datasets_app, name="datasets")
app.add_typer(manifest_app, name="manifest")
console = Console()
# A wider, fixed-width console for the dataset-registry tables: names and
# license strings are long enough that the default (terminal-detected, often
# 80-column) width truncates them illegibly, especially when stdout is not a
# real terminal (e.g. under CliRunner in tests, or piped output).
_registry_console = Console(width=160)

_DETAIL_STRING_LIMIT = 80


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays and bytes into JSON-serialisable values."""
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


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
) -> None:
    """Analyze a single image and print per-detector results."""
    forensic_image = ForensicImage.from_path(path)
    digest = image_hash(forensic_image.rgb)
    names = _resolve_detector_names(detector)

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
                    "details": _jsonable(result.details),
                    "heatmap": (
                        str(heatmap_paths[result.detector])
                        if heatmap_paths[result.detector] is not None
                        else None
                    ),
                }
                for result in results
            ],
        }
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
            details_table.add_row(key, escape(_truncate(str(_jsonable(value)))))
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
        typer.Option("--all-signals", help="Include every registered signal detector."),
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
) -> None:
    """Evaluate registered detectors over a manifest and report Markdown tables."""
    manifest = Manifest.load(manifest_path)

    names = set(detector or [])
    if all_signals:
        names |= set(registry.available())
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


if __name__ == "__main__":
    app()
