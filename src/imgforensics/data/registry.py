"""Dataset registry: catalogue of external datasets this project can use.

Third-party datasets are never redistributed by this repository -- this
module only records where each one comes from, its license, and a
``commercial_ok`` flag so a downstream training or evaluation config can
filter on it with one predicate. The data lives in ``datasets.yaml``,
shipped as package data.

Every entry carries a ``verified`` flag and a ``verified_on`` date.
``verified`` is ``True`` only when someone actually opened a primary
source for that dataset (its project page, GitHub/Hugging Face/Zenodo
listing, or official download page) and checked the entry's key facts
(license, access, and at least one of image count/size) against it --
never a search-result snippet or background knowledge alone.
``verified_on`` records the date of that check (ISO ``YYYY-MM-DD``) so
stale entries can be re-checked over time. When ``verified`` is ``False``,
treat every field -- especially ``license`` and ``commercial_ok`` -- as a
lead to re-check before relying on it, not a confirmed fact.
"""

from __future__ import annotations

from importlib import resources
from typing import Literal

import yaml
from pydantic import BaseModel

Task = Literal["detection", "localization", "real-source"]
Access = Literal["open", "form", "gated", "unknown"]


class DatasetInfo(BaseModel):
    """A single external dataset entry in the registry."""

    name: str
    task: Task
    homepage: str | None = None
    download: str | None = None
    access: Access
    license: str
    commercial_ok: bool | None = None
    approx_size_gb: float | None = None
    images: str | None = None
    generators: str | None = None
    masks: bool | None = None
    citation: str | None = None
    notes: str | None = None
    verified: bool = False
    verified_on: str | None = None


def load_registry() -> list[DatasetInfo]:
    """Load and validate every dataset entry from the packaged ``datasets.yaml``."""
    text = (
        resources.files("imgforensics.data").joinpath("datasets.yaml").read_text(encoding="utf-8")
    )
    raw = yaml.safe_load(text) or {}
    return [DatasetInfo.model_validate(item) for item in raw.get("datasets", [])]


def get_dataset(name: str) -> DatasetInfo:
    """Return the registry entry named ``name`` (case-sensitive, exact match).

    Raises:
        KeyError: if no dataset is registered under ``name``, listing the
            available names in the error message.
    """
    for info in load_registry():
        if info.name == name:
            return info
    known = ", ".join(sorted(entry.name for entry in load_registry()))
    raise KeyError(f"No dataset registered as {name!r}. Available: {known or '<none>'}")


def registry_table_markdown() -> str:
    """Render the full registry as a Markdown table."""
    header = (
        "| Name | Task | Access | License | Commercial OK | Approx GB | Verified |\n"
        "|---|---|---|---|---|---|---|"
    )
    rows = []
    for info in load_registry():
        commercial = "unverified" if info.commercial_ok is None else str(info.commercial_ok)
        size = "-" if info.approx_size_gb is None else f"{info.approx_size_gb:g}"
        rows.append(
            f"| {info.name} | {info.task} | {info.access} | {info.license} | "
            f"{commercial} | {size} | {info.verified} |"
        )
    return "\n".join([header, *rows])
