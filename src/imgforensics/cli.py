"""Command-line interface for imgforensics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import imgforensics.signals  # noqa: F401  (side effect: registers metadata/ela/c2pa/sd_watermark)
from imgforensics import __version__
from imgforensics.core import registry
from imgforensics.core.image import ForensicImage
from imgforensics.core.types import DetectionResult
from imgforensics.utils.image_io import image_hash

app = typer.Typer(help="Detect AI-generated images, AI-inpainted regions, and manipulations.")
console = Console()

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
            details_table.add_row(key, _truncate(str(_jsonable(value))))
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


if __name__ == "__main__":
    app()
