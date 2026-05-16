"""Tests for ``synthpix.hf.card`` README generation."""

from __future__ import annotations

import io
import subprocess

import pytest
from ruamel.yaml import YAML

from synthpix.hf.card import DatasetCardMeta, make_dataset_card
from synthpix.hf.layout import LayoutSummary


def _fake_layout() -> LayoutSummary:
    return LayoutSummary(
        splits={"train": 10, "val": 2, "test": 4},
        reynolds_by_split={
            "train": [],
            "val": [],
            "test": ["DNS_turbulence", "cylinder"],
        },
        total_bytes=1024,
        mat_files=16,
        extra_files=0,
    )


def _parse_frontmatter(card: str) -> dict:
    assert card.startswith("---\n")
    end = card.index("\n---\n", 4)
    body = card[4:end]
    yaml = YAML(typ="safe")
    return yaml.load(io.StringIO(body))


def test_card_starts_with_yaml_frontmatter():
    meta = DatasetCardMeta(
        name="example-piv",
        source_url="https://example.org/data",
        citation="cite me",
        pretty_name="Example PIV",
        synthpix_version="9.9.9",
        synthpix_commit="abc1234",
    )

    card = make_dataset_card(meta, _fake_layout())
    fm = _parse_frontmatter(card)

    assert fm["license"] == "other"
    assert fm["license_name"] == "research-only-arr"
    assert fm["pretty_name"] == "Example PIV"
    assert "PIV" in fm["tags"]
    assert "synthetic" in fm["tags"]


def test_card_body_contains_source_citation_and_counts():
    meta = DatasetCardMeta(
        name="example-piv",
        source_url="https://example.org/data",
        citation="@article{foo}",
        pretty_name="Example PIV",
        synthpix_version="0.1.2",
        synthpix_commit="abc1234",
    )

    card = make_dataset_card(meta, _fake_layout())

    assert "Example PIV" in card
    assert "https://example.org/data" in card
    assert "@article{foo}" in card
    assert "train" in card
    assert "10" in card
    assert "DNS_turbulence" in card
    assert "0.1.2" in card
    assert "abc1234" in card


def test_card_license_note_is_present():
    meta = DatasetCardMeta(
        name="example-piv",
        source_url="https://example.org/data",
        citation="cite",
        pretty_name="Example PIV",
        synthpix_version="0.1.2",
        synthpix_commit=None,
    )

    card = make_dataset_card(meta, _fake_layout())

    assert "hosted under terms set by the original authors" in card


def test_card_meta_auto_fills_synthpix_version(monkeypatch):
    monkeypatch.setattr(
        "synthpix.hf.card._lookup_synthpix_version", lambda: "1.2.3"
    )

    meta = DatasetCardMeta(
        name="example-piv",
        source_url="https://example.org/data",
        citation="cite",
        pretty_name="Example PIV",
        synthpix_commit="deadbee",
    )

    card = make_dataset_card(meta, _fake_layout())

    assert "1.2.3" in card


def test_card_meta_auto_fills_synthpix_commit(monkeypatch):
    def fake_check_output(*args, **kwargs):
        return b"feedface1234\n"

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    meta = DatasetCardMeta(
        name="example-piv",
        source_url="https://example.org/data",
        citation="cite",
        pretty_name="Example PIV",
        synthpix_version="0.1.2",
    )

    card = make_dataset_card(meta, _fake_layout())

    assert "feedface1234" in card


def test_card_meta_handles_commit_failure(monkeypatch):
    def raise_error(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["git"])

    monkeypatch.setattr(subprocess, "check_output", raise_error)

    meta = DatasetCardMeta(
        name="example-piv",
        source_url="https://example.org/data",
        citation="cite",
        pretty_name="Example PIV",
        synthpix_version="0.1.2",
    )

    # Should not raise.
    card = make_dataset_card(meta, _fake_layout())

    assert "Example PIV" in card


def test_card_handles_empty_reynolds_map():
    layout = LayoutSummary(
        splits={"train": 1},
        reynolds_by_split={"train": []},
        total_bytes=64,
        mat_files=1,
        extra_files=0,
    )
    meta = DatasetCardMeta(
        name="example-piv",
        source_url="https://example.org/data",
        citation="cite",
        pretty_name="Example PIV",
        synthpix_version="0.1.2",
        synthpix_commit="abc",
    )

    card = make_dataset_card(meta, layout)

    assert "train" in card
