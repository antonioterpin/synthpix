"""Tests for HDF5DataSource."""

import os
import h5py
import numpy as np
import pytest

from synthpix.data_sources.hdf5 import HDF5DataSource


def test_hdf5_loading(mock_hdf5_files):
    """Test loading HDF5 files using the mock_hdf5_files fixture."""
    file_paths, dims = mock_hdf5_files
    root = os.path.dirname(file_paths[0])

    ds = HDF5DataSource([root])
    assert len(ds) == len(file_paths)
    item = ds[0]

    # fixture creates (X, Y, Z, C) -> (1536, 100, 2048, 2) by default (non-CI)
    # or (128, 10, 128, 2) in CI.
    assert "flow_fields" in item
    assert item["flow_fields"].shape[-1] == 2
    assert item["file"] in file_paths


def test_hdf5_group_error(tmp_path):
    """Test ValueError if the first key is not a Dataset (e.g. it is a Group)."""
    p = tmp_path / "group.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("some_group")
        g.create_dataset("d", data=np.random.rand(5, 5))

    ds = HDF5DataSource(
        [str(tmp_path)]
    )  # Changed to list for consistency with other tests
    # It will find 'group.h5'
    # load_file should raise ValueError because `list(file)[0]` is "some_group" which is a Group.

    with pytest.raises(ValueError, match="Expected Dataset but got"):
        _ = ds[
            0
        ]  # Changed from `ds[0]== 2` to `_ = ds[0]` to correctly trigger the load and error


def test_hdf5_invalid_ext(tmp_path):
    """Test ValueError when no HDF5 files are found."""
    p1 = tmp_path / "test.txt"
    p1.write_text("fail")

    # Should raise ValueError because no files match *.h5 pattern
    with pytest.raises(ValueError, match="No files found"):
        HDF5DataSource([str(tmp_path)])
