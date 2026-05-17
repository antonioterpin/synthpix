"""Tests for ``synthpix.hf.layout`` directory introspection."""

from __future__ import annotations

from pathlib import Path

import pytest

from synthpix.hf.layout import LayoutSummary, inspect_local_layout


def _make_mat(path: Path, payload: bytes = b"x" * 16) -> None:
    """Create a fake ``.mat`` file with a known byte payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


@pytest.fixture
def synthetic_tree(tmp_path: Path) -> Path:
    """Build a representative dataset layout under ``tmp_path``."""
    for i in range(4):
        _make_mat(tmp_path / "train" / f"flow_{i:04d}.mat")
    for i in range(2):
        _make_mat(tmp_path / "val" / f"flow_{i:04d}.mat")
    for i in range(2):
        _make_mat(
            tmp_path / "test" / "DNS_turbulence" / "Re100" / f"flow_{i}.mat"
        )
    for i in range(2):
        _make_mat(tmp_path / "test" / "cylinder" / f"flow_{i}.mat")

    # Empty tune split — should still be reported with count 0.
    (tmp_path / "tune").mkdir()

    # Extra non-``.mat`` file should be counted under ``extra_files``.
    (tmp_path / "train" / "notes.txt").write_text("ignore me")

    # Dotfiles must be skipped entirely.
    (tmp_path / "train" / ".hidden").write_text("nope")

    return tmp_path


def test_inspect_returns_layout_summary(synthetic_tree: Path):
    summary = inspect_local_layout(synthetic_tree)

    assert isinstance(summary, LayoutSummary)


def test_inspect_counts_mat_files_per_split(synthetic_tree: Path):
    summary = inspect_local_layout(synthetic_tree)

    assert summary.splits["train"] == 4
    assert summary.splits["val"] == 2
    assert summary.splits["test"] == 4
    assert summary.splits["tune"] == 0


def test_inspect_maps_reynolds_subdirs(synthetic_tree: Path):
    summary = inspect_local_layout(synthetic_tree)

    assert summary.subdirs_by_split["test"] == ["DNS_turbulence", "cylinder"]
    # Splits without ``Re*``-style subdirs get an empty list.
    assert summary.subdirs_by_split["train"] == []


def test_inspect_counts_extras_and_skips_dotfiles(synthetic_tree: Path):
    summary = inspect_local_layout(synthetic_tree)

    assert summary.mat_files == 4 + 2 + 4
    assert summary.extra_files == 1
    assert summary.total_bytes > 0


def test_inspect_tolerates_missing_split(tmp_path: Path):
    _make_mat(tmp_path / "train" / "flow.mat")

    summary = inspect_local_layout(tmp_path)

    assert "tune" not in summary.splits
    assert summary.splits == {"train": 1}


def test_inspect_skips_files_in_hidden_directories(tmp_path: Path):
    # Files nested under any hidden ancestor must not be counted.
    _make_mat(tmp_path / "train" / "real.mat")
    _make_mat(tmp_path / "train" / ".cache" / "leaked.mat")
    _make_mat(tmp_path / "train" / "scene" / ".tmp" / "still_leaked.mat")
    (tmp_path / "train" / "scene" / ".tmp" / "junk.txt").write_text("nope")

    summary = inspect_local_layout(tmp_path)

    assert summary.splits["train"] == 1
    assert summary.mat_files == 1
    assert summary.extra_files == 0
    # ``.cache`` (hidden dir) must not surface as a subdir name.
    assert ".cache" not in summary.subdirs_by_split["train"]


def test_layout_summary_is_frozen(synthetic_tree: Path):
    summary = inspect_local_layout(synthetic_tree)

    with pytest.raises(Exception):
        summary.mat_files = 0  # type: ignore[misc]
