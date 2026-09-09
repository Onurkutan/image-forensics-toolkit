"""Command-line interface for imgforensics."""

from __future__ import annotations

import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from imgforensics import __version__
from imgforensics.core import registry
from imgforensics.utils.image_io import image_hash, load_image

app = typer.Typer(help="Detect AI-generated images, AI-inpainted regions, and manipulations.")
console = Console()


@app.command()
def analyze(path: Path) -> None:
    """Analyze a single image and print detector results."""
    image = load_image(path)
    digest = image_hash(image)

    info_table = Table(title="Image info")
    info_table.add_column("Field")
    info_table.add_column("Value")
    info_table.add_row("file", path.name)
    info_table.add_row("size", f"{image.width}x{image.height}")
    info_table.add_row("mode", image.mode)
    info_table.add_row("sha256", digest[:12])
    console.print(info_table)

    detector_names = registry.available()
    if not detector_names:
        console.print("No detectors registered yet.")
        raise typer.Exit(code=0)

    results_table = Table(title="Detector results")
    results_table.add_column("detector")
    results_table.add_column("score")
    results_table.add_column("label")
    results_table.add_column("elapsed_ms")
    for name in detector_names:
        detector_cls = registry.get(name)
        detector = detector_cls()
        detector.load()
        start = time.perf_counter()
        result = detector.predict(image)
        elapsed_ms = (time.perf_counter() - start) * 1000
        results_table.add_row(
            result.detector,
            f"{result.score:.3f}",
            result.label,
            f"{elapsed_ms:.1f}",
        )
    console.print(results_table)


@app.command()
def version() -> None:
    """Print the installed imgforensics version."""
    console.print(__version__)


if __name__ == "__main__":
    app()
