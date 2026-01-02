"""Tests for FileDataSource base class."""

import pytest
from synthpix.data_sources.base import FileDataSource


class ConcreteDataSource(FileDataSource):
    """Concrete implementation for testing."""

    def load_file(self, file_path):
        return {"file": file_path}


def test_base_init_str(tmp_path):
    """Test initialization with a single string path."""
    f = tmp_path / "test.txt"
    f.touch()
    ds = ConcreteDataSource(str(f))
    assert len(ds) == 1
    assert ds.file_list == [str(f)]


def test_base_init_invalid_type():
    """Test ValueError for invalid input types."""
    with pytest.raises(ValueError, match="dataset_path must be a list"):
        ConcreteDataSource(123)

    with pytest.raises(ValueError, match="dataset_path must be a list"):
        ConcreteDataSource(None)

    with pytest.raises(ValueError, match="dataset_path must be a list"):
        ConcreteDataSource([123, "valid"])


def test_base_no_files_found(tmp_path):
    """Test ValueError when no files match the pattern."""
    # Pattern defaults to "*"
    # Empty dir
    with pytest.raises(ValueError, match="No files found"):
        ConcreteDataSource(str(tmp_path))


def test_base_direct_file_path(tmp_path):
    """Test passing a file path directly (no discovery needed)."""
    f = tmp_path / "test.txt"
    f.touch()
    ds = ConcreteDataSource([str(f)])
    assert len(ds) == 1


def test_base_include_images_default():
    """Test default include_images property."""
    ds = ConcreteDataSource(["dummy"])
    assert ds.include_images is False


def test_base_getitem(tmp_path):
    """Test __getitem__ delegation."""
    f = tmp_path / "test.txt"
    f.touch()
    ds = ConcreteDataSource(str(f))
    item = ds[0]
    assert item["file"] == str(f)
