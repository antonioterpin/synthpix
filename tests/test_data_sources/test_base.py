"""Tests for the FileDataSource base class.

FileDataSource provides common functionality for data sources that load data 
from a list of files, including file discovery, initialization validation, 
and basic indexing.
"""

import pytest

from synthpix.data_sources.base import FileDataSource


class ConcreteDataSource(FileDataSource):
    """A concrete implementation of FileDataSource for testing purposes.

    It simply returns the file path when a file is loaded.
    """

    def load_file(self, file_path):
        return {"file": file_path}


def test_base_init_str(tmp_path):
    """Test that FileDataSource correctly handles a single string path as input.

    It should automatically wrap the string in a list and proceed with 
    file discovery.
    """
    f = tmp_path / "test.txt"
    f.touch()
    ds = ConcreteDataSource(str(f))
    assert len(ds) == 1, f"Expected 1 file, got {len(ds)}"
    assert ds.file_list == [str(f)], f"Expected file_list {[str(f)]}, got {ds.file_list}"


def test_base_init_invalid_type():
    """Test ValueError for invalid input types."""
    with pytest.raises(ValueError, match="dataset_path must be a list"):
        ConcreteDataSource(123)  # type: ignore

    with pytest.raises(ValueError, match="dataset_path must be a list"):
        ConcreteDataSource(None)  # type: ignore

    with pytest.raises(ValueError, match="dataset_path must be a list"):
        ConcreteDataSource([123, "valid"])  # type: ignore


def test_base_no_files_found(tmp_path):
    """Test that ValueError is raised during initialization if no files are found.

    This ensures that the data source doesn't enter an invalid state with an 
    empty file list.
    """
    # Pattern defaults to "*"
    # Empty dir
    with pytest.raises(ValueError, match="No files found"):
        ConcreteDataSource(str(tmp_path))


def test_base_direct_file_path(tmp_path):
    """Test passing a file path directly (no discovery needed)."""
    f = tmp_path / "test.txt"
    f.touch()
    ds = ConcreteDataSource([str(f)])
    assert len(ds) == 1, f"Expected 1 file when passing direct path, got {len(ds)}"


def test_base_include_images_default():
    """Test default include_images property."""
    ds = ConcreteDataSource(["dummy"])
    assert ds.include_images is False, "include_images should default to False"


def test_base_getitem(tmp_path):
    """Test __getitem__ delegation."""
    f = tmp_path / "test.txt"
    f.touch()
    ds = ConcreteDataSource(str(f))
    item = ds[0]
    assert item["file"] == str(f), f"Expected 'file' to be {str(f)}, got {item['file']}"
