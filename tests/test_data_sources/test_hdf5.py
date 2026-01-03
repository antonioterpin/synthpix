"""Tests for HDF5DataSource.

HDF5DataSource is designed to load flow fields from HDF5 files. Each file is 
expected to contain at least one Dataset. The first key found in the HDF5 file 
is assumed to be the Dataset containing the flow field data.
"""

import os

import h5py
import numpy as np
import pytest

from synthpix.data_sources.hdf5 import HDF5DataSource


def test_hdf5_loading(mock_hdf5_files):
    """Test standard loading of HDF5 files using the mock_hdf5_files fixture.

    This test verifies that:
    1. The data source correctly identifies the number of files.
    2. Items can be retrieved by index.
    3. The retrieved items contain the expected 'flow_fields' key.
    4. The shape of the flow fields is correct (last dimension is 2 for UV).
    5. The 'file' metadata matches the expected file path.
    """
    file_paths, dims = mock_hdf5_files
    root = os.path.dirname(file_paths[0])

    ds = HDF5DataSource([root])
    assert len(ds) == len(file_paths), f"Expected {len(file_paths)} files, got {len(ds)}"
    item = ds[0]

    # fixture creates (X, Y, Z, C) -> (1536, 100, 2048, 2) by default (non-CI)
    # or (128, 10, 128, 2) in CI.
    assert "flow_fields" in item, "Item missing 'flow_fields' key"
    assert item["flow_fields"].shape[-1] == 2, f"Expected last dimension to be 2 (UV components), got {item['flow_fields'].shape}"
    assert item["file"] in file_paths, "Item file path not in original file list"


def test_hdf5_group_error(tmp_path):
    """Test that ValueError is raised if the first key in HDF5 is a Group, not a Dataset.

    HDF5DataSource expects the first item in the HDF5 file to be a Dataset. 
    If a file is structured with a Group at the top level instead, it should 
    raise a descriptive ValueError during item access.
    """
    p = tmp_path / "group.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("some_group")
        g.create_dataset("d", data=np.random.rand(5, 5))

    ds = HDF5DataSource(
        [str(tmp_path)]
    )  # Changed to list for consistency with other tests
    # It will find 'group.h5'
    # load_file should raise ValueError because `list(file)[0]` is
    # "some_group" which is a Group.

    with pytest.raises(ValueError, match="Expected Dataset but got"):
        # Changed from `ds[0]== 2` to `_ = ds[0]` to correctly trigger the load
        # and error
        _ = ds[0]


def test_hdf5_invalid_ext(tmp_path):
    """Test that ValueError is raised when no HDF5 files (*.h5) are found in the directory.

    The data source should filter for HDF5 files and raise an error during 
    initialization if the resulting file list is empty.
    """
    p1 = tmp_path / "test.txt"
    p1.write_text("fail")

    # Should raise ValueError because no files match *.h5 pattern
    with pytest.raises(ValueError, match="No files found"):
        HDF5DataSource([str(tmp_path)])
