r"""Tests for NumpyDataSource.

NumpyDataSource is used to load flow fields (and optionally images)
stored as NumPy (.npy) files. It expects a specific filename pattern
r"flow_(\d+).npy" when paired images are requested, to correctly
match flow fields with their corresponding images.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from synthpix.data_sources import NumpyDataSource


def test_numpy_simple(
    mock_numpy_files: tuple[list[str], dict[str, int]],
) -> None:
    """Test standard loading of NumPy flow files without images.

    Verifies that the data source correctly identifies the number of
    files and that retrieved items contain 'flow_fields' but not
    'images1'.

    Args:
        mock_numpy_files: Temporary NumPy file paths and dimensions.
    """
    file_paths, _dims = mock_numpy_files
    root = os.path.dirname(file_paths[0])

    ds = NumpyDataSource([root], include_images=False)
    assert len(ds) == len(file_paths), (
        f"Expected {len(file_paths)} files, got {len(ds)}"
    )
    item = ds[0]
    assert "flow_fields" in item, "Item missing 'flow_fields' key"
    assert "images1" not in item, (
        "Item should not contain 'images1' when include_images=False"
    )


def test_numpy_with_images(
    mock_numpy_files: tuple[list[str], dict[str, int]],
) -> None:
    """Test loading Numpy flow with paired images using fixture.

    Args:
        mock_numpy_files: Temporary NumPy file paths and dimensions.
    """
    file_paths, dims = mock_numpy_files
    root = os.path.dirname(file_paths[0])

    ds = NumpyDataSource([root], include_images=True)
    item = ds[0]

    assert "images1" in item, "Item missing 'images1' key"
    assert "images2" in item, "Item missing 'images2' key"
    assert item["images1"].shape == (
        dims["height"],
        dims["width"],
        3,
    ), (
        f"Image shape mismatch. Expected "
        f"{(dims['height'], dims['width'], 3)}, got "
        f"{item['images1'].shape}"
    )


def test_numpy_missing_images(tmp_path: Path) -> None:
    """Test FileNotFoundError when paired images are missing.

    Args:
        tmp_path: Temporary directory for this test.
    """
    p_flow = tmp_path / "flow_1.npy"
    np.save(p_flow, np.random.rand(64, 64, 2))
    # Missing images img_0.jpg and img_1.jpg

    with pytest.raises(FileNotFoundError, match="Missing images"):
        NumpyDataSource([str(tmp_path)], include_images=True)


def test_numpy_invalid_extension(tmp_path: Path) -> None:
    """Test that ValueError is raised if files do not have the .npy
    extension.

    NumpyDataSource explicitly validates that all files in its file list
    are NumPy files. This test verifies that providing a non-npy file
    triggers the appropriate ValueError.

    Args:
        tmp_path: Temporary directory for this test.
    """
    p_txt = tmp_path / "test.txt"
    p_txt.write_text("dummy")

    # Pass the text file directly to bypass directory glob filtering.
    # The base FileDataSource does not filter individual files passed in the
    # list,
    # so the subclass validation logic will catch this.
    with pytest.raises(ValueError, match="must be numpy files"):
        NumpyDataSource([str(p_txt)])


def test_numpy_bad_filename_format(tmp_path: Path) -> None:
    r"""Test ValueError for filenames that don't match the `flow_(\d+).npy`
    pattern.

    When `include_images=True`, the data source requires a specific
    naming convention to pair flows with images (e.g., flow_1.npy ->
    img_1.jpg). Filenames that don't match this pattern should cause a
    ValueError during initialization.

    Args:
        tmp_path: Temporary directory for this test.
    """
    # This check `mb = re.match` happens in __init__ if include_images=True.
    p = tmp_path / "flow_abc.npy"
    np.save(p, np.random.rand(32, 32, 2))

    with pytest.raises(ValueError, match="Bad filename"):
        NumpyDataSource([str(tmp_path)], include_images=True)


def test_numpy_load_file_bad_filename(tmp_path: Path) -> None:
    """Test ValueError in load_file if filename is bad (defensive check).

    This test dynamically sets `include_images=True` on a data source
    that was initialized without images, to verify that `load_file` also
    performs the filename pattern check.

    Args:
        tmp_path: Temporary directory for this test.
    """
    p = tmp_path / "flow_abc.npy"
    np.save(p, np.random.rand(32, 32, 2))

    ds = NumpyDataSource([str(tmp_path)], include_images=False)
    # Force include_images to True dynamically to trigger the check in load_file
    ds._include_images = True

    with pytest.raises(ValueError, match="Bad filename"):
        ds.load_file(str(p))


def test_numpy_manual_list_validation() -> None:
    """Test validation if file list contains non-npy file (manual
    injection).
    """
    with pytest.raises(ValueError, match="must be numpy files"):
        NumpyDataSource(["file.txt"])
