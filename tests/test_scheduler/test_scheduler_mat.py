"""Tests for the MATLAB (.mat) file flow field scheduler.

These tests verify the loading of flow fields and images from .mat files (both 
v5 and v7.3/HDF5 formats), including resizing, scaling, and handling of 
various interleaved data layouts.
"""
import os

import h5py
import jax
import numpy as np
import pytest
import scipy.io

from synthpix.scheduler import EpisodicFlowFieldScheduler, MATFlowFieldScheduler
from synthpix.types import SchedulerData
from synthpix.utils import load_configuration

config = load_configuration("config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_SCHEDULER"]


@pytest.mark.parametrize("bad_include_images", ["string", 123, None, []])
@pytest.mark.parametrize("mock_mat_files", [1], indirect=True)
def test_mat_scheduler_invalid_include_images(
    bad_include_images, mock_mat_files
):
    """Test that `MATFlowFieldScheduler` rejects non-boolean `include_images` values."""
    files, _ = mock_mat_files
    with pytest.raises(
        ValueError, match="include_images must be a boolean value."
    ):
        MATFlowFieldScheduler.from_config(
            {
                "file_list": files,
                "include_images": bad_include_images,
            }
        )


@pytest.mark.parametrize(
    "bad_output_shape", ["bad_value", None, 123, (256,), (1, 2, 1)]
)
def test_mat_scheduler_invalid_output_shape(bad_output_shape, mock_mat_files):
    """Test that invalid `output_shape` values raise a ValueError."""
    files, _ = mock_mat_files
    with pytest.raises((ValueError, TypeError)):
        MATFlowFieldScheduler.from_config(
            {
                "file_list": files,
                "output_shape": bad_output_shape,
            }
        )


@pytest.mark.parametrize(
    "bad_output_shape", [(256, -1), (0, 256), (256, 0), (-1, -1)]
)
def test_mat_scheduler_invalid_output_shape_values(
    bad_output_shape, mock_mat_files
):
    """Test that invalid `output_shape` values raise a ValueError."""
    files, _ = mock_mat_files
    with pytest.raises(
        ValueError, match="output_shape must contain positive integers."
    ):
        MATFlowFieldScheduler.from_config(
            {
                "file_list": files,
                "output_shape": bad_output_shape,
            }
        )


@pytest.mark.parametrize("mock_mat_files", [2], indirect=True)
def test_mat_scheduler_iteration(mock_mat_files):
    """Verify that `MATFlowFieldScheduler` correctly iterates through multiple .mat files.

    Confirms that `get_batch` yields the expected number of batches with 
    accurate flow field shapes.
    """
    files, _ = mock_mat_files
    scheduler = MATFlowFieldScheduler.from_config(
        {
            "file_list": files,
            "randomize": False,
            "loop": False,
        }
    )

    count = 0
    while True:
        try:
            data = scheduler.get_batch(batch_size=1)
            assert isinstance(data, SchedulerData), f"Expected SchedulerData, got {type(data)}"
            assert data.flow_fields.shape == (1, 256, 256, 2), f"Expected flow field shape (1, 256, 256, 2), got {data.flow_fields.shape}"
            count += 1
        except StopIteration:
            break

    assert count == 2, f"Expected 2 files to be processed, but got {count}"


@pytest.mark.parametrize("mock_mat_files", [1], indirect=True)
def test_mat_scheduler_shape(mock_mat_files):
    files, _ = mock_mat_files
    scheduler = MATFlowFieldScheduler.from_config(
        {
            "file_list": files,
        }
    )
    shape = scheduler.get_flow_fields_shape()
    assert shape == (256, 256, 2), f"Expected flow field shape (256, 256, 2), got {shape}"


@pytest.mark.parametrize("mock_mat_files", [2], indirect=True)
def test_mat_scheduler_init_flags(mock_mat_files):
    files, _ = mock_mat_files
    scheduler = MATFlowFieldScheduler.from_config(
        {
            "file_list": files,
            "randomize": True,
            "loop": True,
        }
    )

    assert scheduler.randomize is True, "Expected randomize flag to be True"
    assert scheduler.loop is True, "Expected loop flag to be True"
    assert scheduler.index == 0, f"Expected initial index to be 0, got {scheduler.index}"


def test_mat_scheduler_invalid_ext(tmp_path):
    bad_file = tmp_path / "invalid.txt"
    bad_file.write_text("invalid content")

    with pytest.raises(
        ValueError, match="All files must be MATLAB .mat files with HDF5 format"
    ):
        MATFlowFieldScheduler([str(bad_file)])


def test_mat_scheduler_file_dir(tmp_path):
    """Test that the scheduler looks for files in the correct directory.

    This test creates a temporary directory and multiple mock .mat files in it,
    then checks if the scheduler correctly identifies the files.
    """
    mat_files = []
    for i in range(2):
        mat_file = tmp_path / f"test_{i}.mat"
        with h5py.File(mat_file, "w") as f:
            f.create_dataset(f"flow_{i}", data=np.random.rand(64, 64, 2))
        mat_files.append(str(mat_file))

    scheduler = MATFlowFieldScheduler(mat_files)
    assert scheduler.file_list == mat_files, "File list in scheduler does not match input files"


@pytest.mark.parametrize("mock_mat_files", [1], indirect=True)
def test_mat_scheduler_get_batch(mock_mat_files):
    files, _ = mock_mat_files
    scheduler = MATFlowFieldScheduler(files)

    batch_size = len(files)
    batch = scheduler.get_batch(batch_size)
    assert isinstance(batch, SchedulerData), f"Expected SchedulerData, got {type(batch)}"
    assert batch.flow_fields.shape == (batch_size, 256, 256, 2), f"Expected flow field shape {(batch_size, 256, 256, 2)}, got {batch.flow_fields.shape}"


@pytest.mark.parametrize("mock_mat_files", [2], indirect=True)
def test_mat_scheduler_with_images(mock_mat_files):
    """Verify that the scheduler correctly loads both flow fields and images.

    Checks that images are present in the output and match the expected 
    dimensions when `include_images=True`.
    """
    files, _ = mock_mat_files
    scheduler = MATFlowFieldScheduler(
        files, include_images=True, output_shape=(256, 256)
    )

    while True:
        try:
            batch = scheduler.get_batch(batch_size=1)
        except StopIteration:
            break

        assert isinstance(batch, SchedulerData), f"Expected SchedulerData, got {type(batch)}"
        assert batch.flow_fields.shape == (1, 256, 256, 2), f"Expected flow field shape (1, 256, 256, 2), got {batch.flow_fields.shape}"
        assert batch.images1 is not None, "Expected images1 to be present in batch"
        assert batch.images2 is not None, "Expected images2 to be present in batch"
        assert batch.images1.shape == (1, 256, 256), f"Expected image1 shape (1, 256, 256), got {batch.images1.shape}"
        assert batch.images2.shape == (1, 256, 256), f"Expected image2 shape (1, 256, 256), got {batch.images2.shape}"


# ============================
# Episodic Scheduler Tests
# ============================


@pytest.mark.parametrize("mock_mat_files", [16], indirect=True)
def test_episode_iteration(mock_mat_files):
    """Verify that `EpisodicFlowFieldScheduler` correctly groups frames into episodes.

    Tests full iteration through an episode and confirms that 
    `get_batch` yields consistent flow field shapes.
    """
    files, dims = mock_mat_files
    H, W = dims["height"], dims["width"]

    batch_size = 4
    episode_length = 8

    base = MATFlowFieldScheduler(files, loop=False, output_shape=(H, W))
    epi = EpisodicFlowFieldScheduler(
        scheduler=base,
        batch_size=batch_size,
        episode_length=episode_length,
        key=jax.random.PRNGKey(42),
    )
    steps = []
    t = 0
    while True:
        try:
            batch = epi.get_batch(batch_size)
        except StopIteration:
            break
        # every batch is (batch_size, H, W, 2)
        assert isinstance(batch, SchedulerData), f"Expected SchedulerData, got {type(batch)}"
        assert batch.flow_fields.shape == (batch_size, H, W, 2), f"Expected flow field shape {(batch_size, H, W, 2)}, got {batch.flow_fields.shape}"

        steps.append(t)
        t += 1
        if t >= episode_length:
            break  # will check below steps_remaining() == 0
    assert steps == list(range(8)), f"Expected steps {list(range(8))}, but got {steps}"
    assert epi.steps_remaining() == 0, f"Expected 0 steps remaining, but got {epi.steps_remaining()}"


@pytest.mark.parametrize("mock_mat_files", [16], indirect=True)
def test_reset_episode_resamples(mock_mat_files):
    """Verify that `reset_episode` correctly reshuffles data for a new episode.

    Ensures that calling `reset_episode` leads to a different sequence of 
    frames, confirming that the internal base scheduler is reset.
    """
    files, dims = mock_mat_files
    H, W = dims["height"], dims["width"]

    batch_size = 4
    episode_length = 8

    base = MATFlowFieldScheduler(files, loop=False, output_shape=(H, W))
    epi = EpisodicFlowFieldScheduler(
        scheduler=base,
        batch_size=batch_size,
        episode_length=episode_length,
        key=jax.random.PRNGKey(42),
    )

    reordered_files = base.file_list
    first0 = epi.get_batch(batch_size)  # t = 0
    epi.reset_episode()
    post_reset_files = base.file_list
    second0 = epi.get_batch(batch_size)  # new t = 0
    assert epi.steps_remaining() == episode_length - 1, f"Expected {episode_length - 1} steps remaining, but got {epi.steps_remaining()}"
    # ensure we didn’t get the exact same files twice
    assert not np.array_equal(first0.flow_fields, second0.flow_fields), "Flow fields before and after reset should be different"

    # Check which files were used by opening the files
    # with a regular scheduler
    base.file_list = reordered_files[:batch_size]
    base.reset()
    rightfirst0 = base.get_batch(batch_size)
    assert np.array_equal(first0.flow_fields, rightfirst0.flow_fields), "Flow fields do not match expected reordered values"

    base.file_list = post_reset_files[:batch_size]
    base.reset()
    rightsecond0 = base.get_batch(batch_size)
    assert np.array_equal(second0.flow_fields, rightsecond0.flow_fields), "Flow fields do not match expected reordered values after reset"


@pytest.mark.parametrize("mock_mat_files", [64], indirect=True)  # 64 frames
def test_steps_remaining(mock_mat_files):
    files, dims = mock_mat_files
    H, W = dims["height"], dims["width"]

    batch_size = 4
    episode_length = 8

    base = MATFlowFieldScheduler(files, loop=False, output_shape=(H, W))
    epi = EpisodicFlowFieldScheduler(
        scheduler=base,
        batch_size=batch_size,
        episode_length=episode_length,
        key=jax.random.PRNGKey(42),
    )

    assert epi.steps_remaining() == episode_length, f"Expected {episode_length} steps remaining, got {epi.steps_remaining()}"
    _ = epi.get_batch(batch_size)
    _ = epi.get_batch(batch_size)
    assert epi.steps_remaining() == episode_length - 2, f"Expected {episode_length - 2} steps remaining after 2 batches, got {epi.steps_remaining()}"


def test_path_is_hdf5_nonexistent():
    """_path_is_hdf5 should gracefully handle a missing file."""
    assert (
        MATFlowFieldScheduler._path_is_hdf5("does_not_exist_123.mat") is False
    ), "Expected False for non-existent .mat file"
    assert (
        MATFlowFieldScheduler._path_is_hdf5("does_not_exist_123.hdf5") is False
    ), "Expected False for non-existent .hdf5 file"
    assert (
        MATFlowFieldScheduler._path_is_hdf5("does_not_exist_123.npy") is False
    ), "Expected False for non-existent .npy file"


def _rand_flow(shape):
    """Generate a optical-flow field."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(shape).astype(np.float32)


def _rand_image(shape):
    """Generate a random image."""
    rng = np.random.default_rng(123)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def make_mat(tmp_path, name, **variables):
    """Create a v5 .mat file via SciPy and return its path."""
    p = tmp_path / f"{name}.mat"
    scipy.io.savemat(p, variables)
    return str(p)


def make_hdf5(tmp_path, name, **variables):
    """Create a simple HDF5 file (mimicking MATLAB v7.3)."""
    p = tmp_path / f"{name}.mat"
    with h5py.File(p, "w") as f:
        for k, v in variables.items():
            f.create_dataset(k, data=v)
    return str(p)


def test_mat_v5_basic(tmp_path):
    """SciPy branch, include_images=False, sized exactly."""
    V = _rand_flow((4, 4, 2))
    fpath = make_mat(tmp_path, "basic", V=V)
    loader = MATFlowFieldScheduler(
        [fpath], include_images=False, output_shape=(4, 4)
    )

    data = loader.load_file(fpath)
    assert isinstance(data, SchedulerData), f"Expected SchedulerData, got {type(data)}"
    np.testing.assert_array_equal(data.flow_fields, V)


def test_mat_with_images_resize(tmp_path):
    """Test automatic resizing and scaling of images and flow fields.

    Verifies that when the input dimensions differ from the target 
    `output_shape`, both images and flow vectors are correctly 
    interpolated and scaled (in the case of flow).
    """
    I0 = _rand_image((2, 2))
    I1 = _rand_image((2, 2))
    V = np.ones((2, 2, 2), dtype=np.float32)  # all-ones flow
    fpath = make_mat(tmp_path, "img_resize", V=V, I0=I0, I1=I1)

    loader = MATFlowFieldScheduler(
        [fpath], include_images=True, output_shape=(4, 4)
    )
    data = loader.load_file(fpath)

    # images resized to 4×4
    assert data.images1 is not None, "Expected images1 to be present"
    assert data.images2 is not None, "Expected images2 to be present"
    assert data.images1.shape == data.images2.shape == (4, 4), f"Expected 4x4 images, got {data.images1.shape} and {data.images2.shape}"
    # flow resized and *scaled* by factor 2 on both axes
    assert data.flow_fields.shape == (4, 4, 2), f"Expected (4, 4, 2) flow field, got {data.flow_fields.shape}"
    assert np.allclose(data.flow_fields[..., 0], 2.0), "Flow fields X component should be scaled to 2.0"
    assert np.allclose(data.flow_fields[..., 1], 2.0), "Flow fields Y component should be scaled to 2.0"


def test_flow_transposed(tmp_path):
    """Test handling of transposed flow layouts (2, H, W).

    Verifies that the loader correctly permutes the axes to the 
    standard (H, W, 2) format.
    """
    V = _rand_flow((2, 4, 4))
    fpath = make_mat(tmp_path, "transpose", V=V)

    loader = MATFlowFieldScheduler([fpath], output_shape=(4, 4))
    data = loader.load_file(fpath)

    assert data.flow_fields.shape == (4, 4, 2), f"Expected (4, 4, 2) flow field, got {data.flow_fields.shape}"
    # spot-check one value to prove correct permutation
    assert np.isclose(data.flow_fields[1, 2, 0], V[0, 1, 2]), f"Flow X component at [1, 2] does not match V[0, 1, 2]"
    assert np.isclose(data.flow_fields[1, 2, 1], V[1, 1, 2]), f"Flow Y component at [1, 2] does not match V[1, 1, 2]"


def test_hdf5_fallback(monkeypatch, tmp_path):
    """SciPy raises NotImplementedError -> h5py branch loads data."""
    # Patch scipy.io.loadmat to always raise NotImplementedError
    monkeypatch.setattr(
        scipy.io,
        "loadmat",
        lambda *a, **kw: (_ for _ in ()).throw(NotImplementedError),
    )
    V = _rand_flow((4, 4, 2))
    fpath = make_hdf5(tmp_path, "v73", V=V)

    loader = MATFlowFieldScheduler([fpath], output_shape=(4, 4))
    data = loader.load_file(fpath)

    assert data.flow_fields.shape == (4, 4, 2), f"Expected (4, 4, 2) flow field, got {data.flow_fields.shape}"
    np.testing.assert_array_equal(data.flow_fields, V)


def test_missing_V_raises(tmp_path):
    """No 'V' present should raise."""
    fpath = make_mat(
        tmp_path, "noV", I0=_rand_image((4, 4)), I1=_rand_image((4, 4))
    )
    loader = MATFlowFieldScheduler([fpath], include_images=False)
    with pytest.raises(ValueError, match="missing 'V'"):
        loader.load_file(fpath)


def test_missing_images_raises(tmp_path):
    """include_images=True but only one image in file."""
    V = _rand_flow((4, 4, 2))
    fpath = make_mat(tmp_path, "noI1", V=V, I0=_rand_image((4, 4)))
    loader = MATFlowFieldScheduler([fpath], include_images=True)
    with pytest.raises(ValueError, match="missing required keys"):
        loader.load_file(fpath)


def test_load_fails_non_hdf5(monkeypatch, tmp_path):
    """SciPy ValueError & path is NOT HDF5."""
    # Force scipy.io.loadmat to raise ValueError
    monkeypatch.setattr(
        scipy.io,
        "loadmat",
        lambda *a, **kw: (_ for _ in ()).throw(ValueError("boom")),
    )
    # create an *empty* .mat file so _path_is_hdf5 is False
    empty_path = tmp_path / "bad.mat"
    empty_path.write_bytes(b"not a mat file")
    path = str(empty_path)

    loader = MATFlowFieldScheduler([path])
    with pytest.raises(ValueError, match="Failed to load .*legacy MATLAB"):
        loader.load_file(path)


@pytest.mark.parametrize("mock_mat_files", [3], indirect=True)
def test_mat_scheduler_get_batch_with_images(mock_mat_files):
    """Cover get_batch include_images==True."""
    files, _ = mock_mat_files
    scheduler = MATFlowFieldScheduler(
        files,
        include_images=True,
        output_shape=(256, 256),
        loop=False,
    )

    batch_size = len(files)
    batch = scheduler.get_batch(batch_size)

    img_prevs = batch.images1
    img_nexts = batch.images2
    flows = batch.flow_fields

    assert img_prevs is not None, "Expected images1 to be present in batch"
    assert img_nexts is not None, "Expected images2 to be present in batch"

    assert img_prevs.shape == (batch_size, 256, 256), f"Expected images1 shape {(batch_size, 256, 256)}, got {img_prevs.shape}"
    assert img_nexts.shape == (batch_size, 256, 256), f"Expected images2 shape {(batch_size, 256, 256)}, got {img_nexts.shape}"
    assert flows.shape == (batch_size, 256, 256, 2), f"Expected flows shape {(batch_size, 256, 256, 2)}, got {flows.shape}"
    assert flows.dtype == np.float32, f"Expected flows dtype float32, got {flows.dtype}"


def test_next_loop_reset(tmp_path):
    flow = _rand_flow((4, 4, 2))
    fpath = make_mat(tmp_path, "one", V=flow)

    sched = MATFlowFieldScheduler(
        [fpath],
        loop=True,
        randomize=False,
        output_shape=(4, 4),
    )

    first = sched.get_batch(batch_size=1)  # consumes file, index -> 1
    second = sched.get_batch(batch_size=1)  # triggers reset() branch

    # Both iterations must yield the same (resized) flow field
    np.testing.assert_array_equal(first.flow_fields[0, ...], flow)
    np.testing.assert_array_equal(second.flow_fields[0, ...], flow)

    # After two successful returns we are back at "end of list"
    assert sched.index == 0, f"Expected index 0 after loop reset, got {sched.index}"
    assert sched._slice_idx == 1, f"Expected _slice_idx 1, got {sched._slice_idx}"
    assert sched.loop is True, "Sanity check: expected loop=True"


def test_next_skip_on_error(tmp_path):
    # 1) Corrupted file that `load_file` cannot interpret
    bad_path = tmp_path / "broken.mat"
    bad_path.write_bytes(b"definitely not a MATLAB file")

    # 2) Valid file that should ultimately be returned
    good_flow = _rand_flow((4, 4, 2))
    good_path = make_mat(tmp_path, "good", V=good_flow)

    sched = MATFlowFieldScheduler(
        [str(bad_path), good_path],
        loop=False,
        output_shape=(4, 4),
    )

    sample = sched.get_batch(1)  # bad file skipped, good file returned

    # The scheduler must now point to the good file it cached
    assert sched._cached_file == good_path, f"Expected cached file to be {good_path}, got {sched._cached_file}"
    np.testing.assert_array_equal(sample.flow_fields[0, ...], good_flow)

    # Both list entries have been consumed (bad skipped, good returned)
    assert sched.index == 1, f"Expected index 1 after skipping bad file and returning good file, got {sched.index}"


@pytest.mark.parametrize("mock_mat_files", [2], indirect=True)
def test_mat_scheduler_get_batch_too_large_pads_correctly(
    mock_mat_files, caplog
):
    """Ask for a batch that is larger than the number of remaining slices.

    Because loop=False, the scheduler should raise StopIteration.
    """
    files, _ = mock_mat_files
    scheduler = MATFlowFieldScheduler(
        files,
        include_images=True,
        output_shape=(256, 256),
        loop=False,
    )

    batch_size = len(files) + 3  # deliberately larger than the dataset

    batch = scheduler.get_batch(batch_size)
    assert batch.mask is not None, "Expected mask to be present in padded batch"
    assert np.sum(batch.mask) == len(files), f"Expected mask sum {len(files)}, got {np.sum(batch.mask)}"
    assert batch.flow_fields.shape == (batch_size, 256, 256, 2), f"Expected flow field shape {(batch_size, 256, 256, 2)}, got {batch.flow_fields.shape}"
    assert batch.images1 is not None, "Expected images1 to be present"
    assert batch.images2 is not None, "Expected images2 to be present"
    assert batch.images1.shape == (batch_size, 256, 256), f"Expected images1 shape {(batch_size, 256, 256)}, got {batch.images1.shape}"
    assert batch.images2.shape == (batch_size, 256, 256), f"Expected images2 shape {(batch_size, 256, 256)}, got {batch.images2.shape}"
    assert (
        batch.flow_fields[len(files):, ...].sum() == 0.0
    ), "Padded entries in flow fields should be zeroed out"


def test_hdf5_recursive_group(monkeypatch, tmp_path):
    # 1) Make sure SciPy cannot handle the file so the loader switches to h5py
    monkeypatch.setattr(
        scipy.io,
        "loadmat",
        lambda *a, **kw: (_ for _ in ()).throw(NotImplementedError),
    )

    # 2) Create an HDF5 .mat file with a nested group structure
    flow = _rand_flow((4, 4, 2))
    fpath = tmp_path / "nested.mat"
    with h5py.File(fpath, "w") as f:
        g1 = f.create_group("lvl1")
        g2 = g1.create_group("lvl2")
        g2.create_dataset("extra", data=np.arange(3))  # any dummy dataset
        f.create_dataset("V", data=flow)  # the required flow field

    # 3) Load the file through the scheduler
    loader = MATFlowFieldScheduler([str(fpath)], output_shape=(4, 4))
    data = loader.load_file(str(fpath))

    # 4) Assertions: top-level and nested keys must be present and correct
    np.testing.assert_array_equal(data.flow_fields, flow)


def test_mat_scheduler_outputs_files(mock_mat_files):
    """Test that the scheduler returns correct file paths in the batch."""
    files, dims = mock_mat_files
    scheduler = MATFlowFieldScheduler.from_config(
        {
            "file_list": files,
            "include_images": True,
            "loop": False,
        }
    )

    batch_size = 1
    # when including images, iteration returns dicts with flow and images
    while True:
        try:
            output = scheduler.get_batch(batch_size)
        except StopIteration:
            break
        assert isinstance(output, SchedulerData), f"Expected SchedulerData, got {type(output)}"
        assert output.files is not None, "Expected file paths to be present in output"
        assert len(output.files) == batch_size, f"Expected {batch_size} files, got {len(output.files)}"

        for file_path in output.files:
            assert os.path.basename(file_path) in [
                os.path.basename(f) for f in files
            ], f"File {os.path.basename(file_path)} does not match any of the input files"
