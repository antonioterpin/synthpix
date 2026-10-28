"""Tests for the FileDataSource base class.

FileDataSource provides common functionality for data sources that
load data from a list of files, including file discovery,
initialization validation, and basic indexing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from synthpix.data_sources.base import FileDataSource


class ConcreteDataSource(FileDataSource):
    """A concrete implementation of FileDataSource for testing purposes.

    It simply returns the file path when a file is loaded.
    """

    def load_file(self, file_path: Path) -> dict[str, Any]:
        """Load file.

        Args:
            file_path: File path used by this test.

        Returns:
            The result used by the test assertions.
        """
        return {"file": file_path}


def test_base_init_str(tmp_path: Path) -> None:
    """Test that FileDataSource correctly handles a single string path as
    input.

    It should automatically wrap the string in a list and proceed with
    file discovery.

    Args:
        tmp_path: Temporary directory for this test.
    """
    f = tmp_path / "test.txt"
    f.touch()
    ds = ConcreteDataSource(str(f))
    assert len(ds) == 1, f"Expected 1 file, got {len(ds)}"
    assert ds.file_list == [str(f)], (
        f"Expected file_list {[str(f)]}, got {ds.file_list}"
    )


def test_base_init_invalid_type() -> None:
    """Test ValueError for invalid input types."""
    with pytest.raises(ValueError, match="dataset_path must be a list"):
        ConcreteDataSource(123)  # pyright: ignore[reportArgumentType]

    with pytest.raises(ValueError, match="dataset_path must be a list"):
        ConcreteDataSource(None)  # pyright: ignore[reportArgumentType]

    with pytest.raises(ValueError, match="dataset_path must be a list"):
        ConcreteDataSource([123, "valid"])  # pyright: ignore[reportArgumentType]


def test_base_no_files_found(tmp_path: Path) -> None:
    """Test that ValueError is raised during initialization if no files are
    found.

    This ensures that the data source doesn't enter an invalid state
    with an empty file list.

    Args:
        tmp_path: Temporary directory for this test.
    """
    # Pattern defaults to "*"
    # Empty dir
    with pytest.raises(ValueError, match="No files found"):
        ConcreteDataSource(str(tmp_path))


def test_base_direct_file_path(tmp_path: Path) -> None:
    """Test passing a file path directly (no discovery needed).

    Args:
        tmp_path: Temporary directory for this test.
    """
    f = tmp_path / "test.txt"
    f.touch()
    ds = ConcreteDataSource([str(f)])
    assert len(ds) == 1, (
        f"Expected 1 file when passing direct path, got {len(ds)}"
    )


def test_base_include_images_default() -> None:
    """Test default include_images property."""
    ds = ConcreteDataSource(["dummy"])
    assert ds.include_images is False, "include_images should default to False"


def test_base_getitem(tmp_path: Path) -> None:
    """Test __getitem__ delegation.

    Args:
        tmp_path: Temporary directory for this test.
    """
    f = tmp_path / "test.txt"
    f.touch()
    ds = ConcreteDataSource(str(f))
    item = ds[0]
    assert item["file"] == str(f), (
        f"Expected 'file' to be {f!s}, got {item['file']}"
    )
