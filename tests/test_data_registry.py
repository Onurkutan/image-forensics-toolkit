"""Tests for imgforensics.data.registry: the packaged dataset registry."""

from __future__ import annotations

import pytest

from imgforensics.data.registry import (
    DatasetInfo,
    get_dataset,
    load_registry,
    registry_table_markdown,
)


def test_load_registry_parses_and_validates() -> None:
    entries = load_registry()
    assert all(isinstance(entry, DatasetInfo) for entry in entries)


def test_load_registry_has_at_least_twenty_six_entries() -> None:
    entries = load_registry()
    assert len(entries) >= 26


def test_load_registry_names_are_unique() -> None:
    entries = load_registry()
    names = [entry.name for entry in entries]
    assert len(names) == len(set(names))


def test_load_registry_includes_expected_datasets() -> None:
    names = {entry.name for entry in load_registry()}
    expected = {
        "Community Forensics",
        "Synthbuster",
        "ITW-SM",
        "WildRF",
        "Chameleon",
        "GenImage",
        "CASIA v1.0",
        "CASIA v2.0",
        "Columbia",
        "COVERAGE",
        "IMD2020",
        "DEFACTO",
        "CocoGlide",
        "AutoSplice",
        "tampCOCO",
        "TGIF",
        "TGIF2",
        "MIML",
        "DEAL-300K",
        "COCO",
        "RAISE",
        "UnbiasedGenImage",
        "DiffusionForensics",
        "ImageNet",
        "LAION-400M/5B",
    }
    missing = expected - names
    assert not missing, f"missing dataset entries: {missing}"


def test_get_dataset_returns_matching_entry() -> None:
    info = get_dataset("TGIF")
    assert info.name == "TGIF"
    assert info.task == "localization"


def test_get_dataset_unknown_name_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_dataset("does-not-exist")


def test_registry_table_markdown_lists_every_dataset() -> None:
    markdown = registry_table_markdown()
    entries = load_registry()
    assert markdown.startswith("| Name |")
    for entry in entries:
        assert entry.name in markdown


@pytest.mark.parametrize("entry", load_registry(), ids=lambda entry: entry.name)
def test_unverified_entries_leave_commercial_ok_or_are_explicit(entry: DatasetInfo) -> None:
    # Every entry is a validated DatasetInfo already (see test_load_registry_parses_and_validates);
    # this just spot-checks the field types the CLI depends on directly.
    assert isinstance(entry.verified, bool)
    assert entry.commercial_ok in (True, False, None)
    assert entry.access in ("open", "form", "gated", "unknown")
    assert entry.task in ("detection", "localization", "real-source")


@pytest.mark.parametrize("entry", load_registry(), ids=lambda entry: entry.name)
def test_homepage_and_download_are_urls_or_null(entry: DatasetInfo) -> None:
    if entry.homepage is not None:
        assert entry.homepage.startswith("http"), f"{entry.name}: homepage is not a URL"
    if entry.download is not None:
        assert entry.download.startswith("http"), f"{entry.name}: download is not a URL"


@pytest.mark.parametrize("entry", load_registry(), ids=lambda entry: entry.name)
def test_verified_entries_record_a_verified_on_date(entry: DatasetInfo) -> None:
    if entry.verified:
        assert entry.verified_on is not None, f"{entry.name}: verified but missing verified_on"
