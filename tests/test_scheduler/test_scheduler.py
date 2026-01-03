"""Comprehensive tests for the flow field scheduler hierarchy.

These tests verify the core scheduling logic, including input validation, 
dataset iteration, randomized shuffling, looping behavior, and prefetching 
capabilities across HDF5 and NumPy-based schedulers.
"""
import os
import timeit
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from synthpix.scheduler import (BaseFlowFieldScheduler, HDF5FlowFieldScheduler,
                                NumpyFlowFieldScheduler,
                                PrefetchingFlowFieldScheduler)
from synthpix.scheduler.base import FileEndedError
from synthpix.types import SchedulerData
from synthpix.utils import load_configuration

config = load_configuration("config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_SCHEDULER"]

# ============================
# Base Class Input Validation Tests
# ============================


@pytest.mark.parametrize(
    "file_list", [[None], [123, "invalid"], [123, "invalid"]]
)
def test_invalid_file_list_type(file_list):
    """Test that `HDF5FlowFieldScheduler` rejects non-list `file_list` inputs.

    The scheduler expects a list of file paths (strings).
    """


@pytest.mark.parametrize("file_list", [["nonexistent.h5"]])
def test_invalid_file_paths(file_list):
    with pytest.raises(
        ValueError, match=f"File {file_list[0]} does not exist."
    ):
        HDF5FlowFieldScheduler.from_config({"file_list": file_list})


@pytest.mark.parametrize("randomize", [None, 123, "invalid"])
def test_invalid_randomize(randomize, temp_file):
    with pytest.raises(ValueError, match="randomize must be a boolean value."):
        HDF5FlowFieldScheduler.from_config(
            {
                "file_list": [temp_file],
                "randomize": randomize,
            }
        )


@pytest.mark.parametrize("loop", [None, 123, "invalid"])
def test_invalid_loop(loop, temp_file):
    with pytest.raises(ValueError, match="loop must be a boolean value."):
        HDF5FlowFieldScheduler.from_config(
            {
                "file_list": [temp_file],
                "loop": loop,
            }
        )


@pytest.mark.parametrize(
    "file_list, randomize, loop",
    [
        ([], True, True),
        ([], False, True),
        ([], True, False),
        ([], False, False),
    ],
)
def test_empty_file_list(file_list, randomize, loop):
    with pytest.raises(ValueError, match="The file_list must not be empty."):
        HDF5FlowFieldScheduler.from_config(
            {
                "file_list": file_list,
                "randomize": randomize,
                "loop": loop,
            }
        )


# ============================
# Subclass-Specific Validation
# ============================


def test_non_hdf5_file(temp_txt_file):
    """Test that non-HDF5 files raise a ValueError."""
    with pytest.raises(
        ValueError, match="All files must be HDF5 files with .h5 extension."
    ):
        HDF5FlowFieldScheduler(file_list=temp_txt_file)


@pytest.mark.parametrize("mock_numpy_files", [2], indirect=True)
def test_numpy_scheduler_iteration(mock_numpy_files):
    """Verify that `NumpyFlowFieldScheduler` correctly iterates through all files.

    Confirms that `get_batch` yields the expected number of batches with 
    the correct flow field dimensions.
    """
    files, dims = mock_numpy_files

    scheduler = NumpyFlowFieldScheduler.from_config(
        {
            "file_list": files,
            "randomize": False,
            "loop": False,
        }
    )

    count = 0
    while True:
        try:
            batch = scheduler.get_batch(batch_size=1)
            flow = batch.flow_fields
            assert isinstance(flow, np.ndarray), f"Expected flow to be np.ndarray, got {type(flow)}"
            assert flow.shape == (1, dims["height"], dims["width"], 2), f"Expected flow shape {(1, dims['height'], dims['width'], 2)}, got {flow.shape}"
            count += 1
        except StopIteration:
            break

    assert count == 2, f"Expected 2 files to be processed, but got {count}"


def test_numpy_scheduler_shape(mock_numpy_files):
    files, dims = mock_numpy_files
    scheduler = NumpyFlowFieldScheduler.from_config(
        {
            "file_list": files,
        }
    )
    shape = scheduler.get_flow_fields_shape()
    assert shape == (dims["height"], dims["width"], 2), f"Expected flow field shape {(dims['height'], dims['width'], 2)}, got {shape}"


@pytest.mark.parametrize("mock_numpy_files", [2], indirect=True)
def test_numpy_scheduler_init_flags(mock_numpy_files):
    files, _ = mock_numpy_files
    scheduler = NumpyFlowFieldScheduler.from_config(
        {
            "file_list": files,
            "randomize": True,
            "loop": True,
        }
    )

    assert scheduler.randomize is True, "Expected randomize flag to be True"
    assert scheduler.loop is True, "Expected loop flag to be True"
    assert scheduler.index == 0, f"Expected initial index to be 0, got {scheduler.index}"


def test_numpy_scheduler_invalid_ext(tmp_path):
    bad_file = tmp_path / "invalid.txt"
    bad_file.write_text("invalid content")

    with pytest.raises(
        ValueError, match="All files must be numpy files with '.npy' extension"
    ):
        NumpyFlowFieldScheduler([str(bad_file)])


def test_numpy_scheduler_invalid_npy(tmp_path):
    bad_file = tmp_path / "floww_0.npy"
    np.zeros((1, 64, 64, 2)).astype(np.float32).tofile(bad_file)

    with pytest.raises(ValueError, match=f"Bad filename: {bad_file}"):
        NumpyFlowFieldScheduler([str(bad_file)], include_images=True)


def test_numpy_scheduler_missing_images(tmp_path):
    file = tmp_path / "flow_1.npy"
    np.zeros((1, 64, 64, 2)).astype(np.float32).tofile(file)

    pattern = f"Missing images for frame {1}: {tmp_path}/img_0.jpg, {tmp_path}/img_1.jpg"
    with pytest.raises(FileNotFoundError, match=pattern):
        NumpyFlowFieldScheduler([str(file)], include_images=True)


@pytest.mark.parametrize("mock_numpy_files", [2], indirect=True)
def test_numpy_scheduler_get_batch(mock_numpy_files):
    files, dims = mock_numpy_files
    scheduler = NumpyFlowFieldScheduler.from_config(
        {
            "file_list": files,
        }
    )

    batch_size = len(files)
    batch = scheduler.get_batch(batch_size)
    assert isinstance(batch, SchedulerData), f"Expected SchedulerData, got {type(batch)}"
    assert batch.flow_fields.shape == (
        batch_size,
        dims["height"],
        dims["width"],
        2,
    ), f"Expected flow field shape {(batch_size, dims['height'], dims['width'], 2)}, got {batch.flow_fields.shape}"


@pytest.mark.parametrize("mock_numpy_files", [2], indirect=True)
def test_numpy_scheduler_with_images(mock_numpy_files):
    files, dims = mock_numpy_files
    scheduler = NumpyFlowFieldScheduler.from_config(
        {"file_list": files, "include_images": True, "loop": False}
    )

    batch_size = 1
    # when including images, iteration returns dicts with flow and images
    while True:
        try:
            output = scheduler.get_batch(batch_size)
        except StopIteration:
            break
        assert isinstance(output, SchedulerData), f"Expected SchedulerData, got {type(output)}"
        assert output.images1 is not None, "Expected images1 to be present in output"
        assert output.images2 is not None, "Expected images2 to be present in output"

        flow = output.flow_fields
        img_prev = output.images1
        img_next = output.images2

        assert isinstance(flow, np.ndarray), f"Expected flow to be np.ndarray, got {type(flow)}"
        assert flow.shape == (batch_size, dims["height"], dims["width"], 2), f"Expected flow shape {(batch_size, dims['height'], dims['width'], 2)}, got {flow.shape}"
        assert isinstance(img_prev, np.ndarray), f"Expected images1 to be np.ndarray, got {type(img_prev)}"
        assert img_prev.shape == (batch_size, dims["height"], dims["width"], 3), f"Expected images1 shape {(batch_size, dims['height'], dims['width'], 3)}, got {img_prev.shape}"
        assert isinstance(img_next, np.ndarray), f"Expected images2 to be np.ndarray, got {type(img_next)}"
        assert img_next.shape == (batch_size, dims["height"], dims["width"], 3), f"Expected images2 shape {(batch_size, dims['height'], dims['width'], 3)}, got {img_next.shape}"


@pytest.mark.parametrize("mock_numpy_files", [2], indirect=True)
def test_numpy_scheduler_loop_reset(mock_numpy_files):
    """Test that the NumPy scheduler automatically restarts when `loop=True`.

    Verifies that after the last file is yielded, the scheduler resets its 
    internal pointer and returns to the first file instead of raising 
    `StopIteration`.
    """
    files, dims = mock_numpy_files
    scheduler = NumpyFlowFieldScheduler(
        file_list=files,
        loop=True,
        randomize=False,
    )

    expected_total = len(files) * 2
    out_shapes = [
        scheduler.get_batch(1).flow_fields.shape for _ in range(expected_total)
    ]

    # We should have received the right number of samples
    assert len(out_shapes) == expected_total, f"Expected {expected_total} samples due to looping, but got {len(out_shapes)}"

    # Every returned flow must have the correct shape
    assert set(out_shapes) == {(1, dims["height"], dims["width"], 2)}, f"Found unexpected flow shapes: {set(out_shapes)}"

    assert scheduler.index == 1, f"Expected index 1 after looping through twice, got {scheduler.index}"


def test_numpy_scheduler_skips_bad_file(tmp_path):
    """Covers the branch where an Exception is raised while loading a file.

    The first .npy file is deliberately corrupted so that `np.load` fails.
    The scheduler must log the error, skip that file, and yield the flow
    from the subsequent valid file instead of crashing.
    """
    # -- corrupt file (cannot be read with np.load)
    bad_file = tmp_path / "flow_0.npy"
    bad_file.write_bytes(b"not a valid npy blob")

    # -- good file
    good_file = tmp_path / "flow_1.npy"
    good_flow = np.random.rand(8, 8, 2).astype(np.float32)
    np.save(good_file, good_flow)

    scheduler = NumpyFlowFieldScheduler(
        file_list=[str(bad_file), str(good_file)],
        randomize=False,
        loop=False,
    )

    # The first call should return the *good* flow after silently
    # skipping the corrupted one.
    flow = scheduler.get_batch(1).flow_fields
    assert np.allclose(flow[0, ...], good_flow), "Flow content does not match expected valid file data"

    # No more valid files remain, so a second call must raise StopIteration
    with pytest.raises(StopIteration):
        _ = scheduler.get_batch(1)


# ============================
# Abstract Behavior Tests
# ============================


class DummyScheduler(BaseFlowFieldScheduler):
    """Minimal implementation of `BaseFlowFieldScheduler` for testing abstract logic.
    
    Provides concrete implementations for `load_file`, `get_next_slice`, 
    and `get_flow_fields_shape` to verify base class behavior like 
    randomization and batching.
    """
    def __init__(self, file_list, randomize=False, loop=False, key=None):
        super().__init__(file_list, randomize, loop, key)

    def load_file(self, file_path) -> SchedulerData:
        data = np.random.rand(4, 2, 4, 3).astype(np.float32)
        return SchedulerData(flow_fields=data)

    def get_next_slice(self) -> SchedulerData:
        assert self._cached_data is not None, "Internal error: _cached_data is None in DummyScheduler"
        if self._slice_idx >= self._cached_data.flow_fields.shape[1]:
            raise FileEndedError("End of file data reached.")
        return SchedulerData(
            flow_fields=self._cached_data.flow_fields[:, self._slice_idx, :, :2]
        )

    def get_flow_fields_shape(self) -> tuple[int, int, int]:
        assert self._cached_data is not None, "Internal error: _cached_data is None in DummyScheduler"
        return (
            self._cached_data.flow_fields.shape[0],
            self._cached_data.flow_fields.shape[2] // 2,
            2,
        )

    @classmethod
    def from_config(cls, config: dict) -> "DummyScheduler":
        return cls(
            file_list=config["file_list"],
            randomize=config.get("randomize", False),
            loop=config.get("loop", False),
        )


class FailingDummyScheduler(DummyScheduler):
    """Same as DummyScheduler, but the first file intentionally fails to load."""

    def load_file(self, file_path):
        if "bad" in Path(file_path).stem:
            raise ValueError("Corrupted file")
        return super().load_file(file_path)


def test_abstract_scheduler_iteration(generate_hdf5_file):
    tmp_file = generate_hdf5_file(
        "dummy_test.h5",
        dims={"x_dim": 4, "y_dim": 2, "z_dim": 4, "features": 3},
    )
    scheduler = DummyScheduler([tmp_file])
    count = 0
    while True:
        try:
            _ = scheduler.get_batch(1)
            count += 1
        except StopIteration:
            break
    assert count == 2, f"Expected 2 batches, got {count}"
    os.remove(tmp_file)


def test_reset_calls_random_shuffle(monkeypatch, tmp_path):
    """Verify that `reset()` correctly re-shuffles the file list using JAX.

    Ensures that when `randomize=True`, the `jax.random.permutation` 
    function is used to reorder files upon manual or automatic reset.
    """
    files = [tmp_path / f"f{i}.dat" for i in range(3)]
    for f in files:
        f.write_text("x")
    call_flag = {"called": 0}

    def spy(key, indices):
        call_flag["called"] += 1
        return jnp.flip(indices)

    monkeypatch.setattr(jax.random, "permutation", spy)

    key = jax.random.PRNGKey(0)
    sch = DummyScheduler([str(f) for f in files], randomize=True, key=key)
    original = sch.file_list.copy()

    # Ensure at least one call during init
    assert call_flag["called"] == 1, f"Expected 1 permutation call during init, got {call_flag['called']}"

    # Manually trigger a few resets
    for _ in range(5):
        sch.reset()
    assert call_flag["called"] == 6, f"Expected 6 total permutation calls (init + 5 resets), got {call_flag['called']}"

    assert isinstance(sch.file_list, list), f"Expected file_list to be a list, got {type(sch.file_list)}"

    assert sch.file_list == list(reversed(original)), "File list not reordered as expected by mocked permutation"


def test_directory_initialisation(tmp_path):
    for idx in range(3):  # create three dummy files in a tmp dir
        (tmp_path / f"flow_{idx}.dat").write_text("irrelevant")

    scheduler = DummyScheduler([str(tmp_path)])  # pass *directory*
    assert len(scheduler) == 3, f"Expected 3 files from directory, got {len(scheduler)}"
    # file_list must be sorted (the Base class guarantees this)
    assert scheduler.file_list == sorted(map(str, tmp_path.iterdir()))


def test_error_branch_skips_bad_file(tmp_path):
    bad = tmp_path / "bad_file.dat"
    good = tmp_path / "good_file.dat"
    bad.write_text("✗")  # unreadable by design
    good.write_text("✓")

    scheduler = FailingDummyScheduler(
        [str(bad), str(good)], randomize=False, loop=False
    )

    # should come from *good* file
    first_flow = scheduler.get_batch(1).flow_fields[0, ...]
    # shape (x, z, 2) = (4, 4, 2)
    assert first_flow.shape == (4, 4, 2)
    # exhaust remaining slice to be sure everything still works
    _ = scheduler.get_batch(1)
    with pytest.raises(StopIteration):
        _ = scheduler.get_batch(1)


def test_get_batch_success(tmp_path):
    f = tmp_path / "file.dat"
    f.write_text("data")
    scheduler = DummyScheduler([str(f)], randomize=False, loop=False)

    batch = scheduler.get_batch(2)  # exactly the available slices
    assert batch.flow_fields.shape == (2, 4, 4, 2)  # (batch, x, z, 2)


def test_get_batch_partial_raises_stopiteration(tmp_path):
    f = tmp_path / "one_file.dat"
    f.write_text("data")
    scheduler = DummyScheduler([str(f)], randomize=False, loop=False)

    batch = scheduler.get_batch(3)
    assert batch is not None, "Scheduler yielded None instead of a batch"
    assert batch.flow_fields.shape == (3, 4, 4, 2), f"Expected padded batch shape (3, 4, 4, 2), got {batch.flow_fields.shape}"
    assert batch.mask is not None, "Expected mask to be present in padded batch"
    assert batch.mask.sum() == 2, f"Expected mask sum 2, got {batch.mask.sum()}"
    assert batch.flow_fields[:2].sum() != 0, "Valid slices should have non-zero data"
    assert batch.flow_fields[2:].sum() == 0, "Padded slice should be zeroed out"


def test_get_batch_warning_and_return(tmp_path):
    f = tmp_path / "file.dat"
    f.write_text("x")

    # ── partial-batch branch ──
    sch_partial = DummyScheduler([str(f)])
    batch = sch_partial.get_batch(3)  # asks for >2 slices -> partial batch
    assert batch.flow_fields.shape == (3, 4, 4, 2)
    assert batch.mask is not None, "Expected mask to be present"
    assert batch.mask.sum() == 2, f"Expected mask sum 2, got {batch.mask.sum()}"
    assert batch.flow_fields[:2].sum() != 0, "Valid slices should have non-zero data"

    # ── success branch ──
    sch_full = DummyScheduler([str(f)])
    batch = sch_full.get_batch(2)  # exactly the available slices
    assert batch.flow_fields.shape == (2, 4, 4, 2)


def test_loop_resets_and_continues_dummy(tmp_path):
    # TODO: fix
    f = tmp_path / "one.dat"
    f.write_text("x")
    sch = DummyScheduler([str(f)], loop=True, randomize=False)

    _ = sch.get_batch(1)  # slice 0
    _ = sch.get_batch(1)  # slice 1
    third = sch.get_batch(1)  # after reset → slice 0 again

    assert third.flow_fields.shape == (1, 4, 4, 2)
    assert sch.index == 0 and sch._slice_idx == 1, f"Expected index 0 and _slice_idx 1 after reset, got {sch.index} and {sch._slice_idx}"


@pytest.mark.parametrize(
    "randomize, loop",
    [(True, True), (False, True), (True, False), (False, False)],
)
def test_flow_field_scheduler_init(randomize, loop, temp_file):
    scheduler = HDF5FlowFieldScheduler([temp_file], randomize, loop)
    assert scheduler.randomize is randomize, f"Expected randomize={randomize}, got {scheduler.randomize}"
    assert scheduler.loop is loop, f"Expected loop={loop}, got {scheduler.loop}"
    assert scheduler.index == 0, f"Expected initial index 0, got {scheduler.index}"
    assert scheduler._slice_idx == 0, f"Expected initial _slice_idx 0, got {scheduler._slice_idx}"


@pytest.mark.parametrize(
    "hdf5_test_dims", [{"x_dim": 10, "y_dim": 6, "z_dim": 20}], indirect=True
)
@pytest.mark.parametrize("mock_hdf5_files", [2], indirect=True)
def test_scheduler_iteration(mock_hdf5_files):
    files, dims = mock_hdf5_files

    scheduler = HDF5FlowFieldScheduler(
        file_list=files, randomize=True, loop=False
    )

    num_flows = 0
    while True:
        try:
            batch = scheduler.get_batch(1)
            flow = batch.flow_fields
            expected_shape = (1, dims["x_dim"], dims["z_dim"], 2)
            assert flow.shape == expected_shape, f"Expected shape {expected_shape}, got {flow.shape}"
            num_flows += 1
        except StopIteration:
            break

    assert num_flows == len(files) * dims["y_dim"], f"Expected {len(files) * dims['y_dim']} flows, but got {num_flows}"


@pytest.mark.parametrize(
    "hdf5_test_dims",
    [
        {"x_dim": 10, "y_dim": 6, "z_dim": 20},  # Standard case
        {"x_dim": 1, "y_dim": 1, "z_dim": 1},  # Minimal dimensions
    ],
    indirect=True,
)
@pytest.mark.parametrize("mock_hdf5_files", [2], indirect=True)
def test_scheduler_iteration_with_multiple_files(mock_hdf5_files):
    """Test that the scheduler iterates correctly over multiple HDF5 files."""
    files, dims = mock_hdf5_files

    scheduler = HDF5FlowFieldScheduler(files, randomize=True, loop=False)

    # Validate scheduler configuration
    assert scheduler.randomize is True, "Expected randomize flag to be True"
    assert scheduler.loop is False, "Expected loop flag to be False"

    # Validate iteration
    num_flows = 0
    while True:
        try:
            batch = scheduler.get_batch(1)
            flow = batch.flow_fields
            expected_shape = (1, dims["x_dim"], dims["z_dim"], 2)
            assert flow.shape == expected_shape, f"Expected shape {expected_shape}, got {flow.shape}"
            num_flows += 1
        except StopIteration:
            break

    # Validate the total number of flows
    assert num_flows == len(files) * dims["y_dim"], f"Expected {len(files) * dims['y_dim']} total flows, got {num_flows}"


@pytest.mark.parametrize("randomize", [True, False])
@pytest.mark.parametrize("mock_hdf5_files", [1], indirect=True)
def test_scheduler_time(randomize, mock_hdf5_files):
    files, dims = mock_hdf5_files
    CI = os.environ.get("CI") == "true"
    time_limit = 0.01 if CI else 10

    scheduler = HDF5FlowFieldScheduler(
        file_list=files, randomize=randomize, loop=False
    )

    def iterate_scheduler():
        while True:
            try:
                _ = scheduler.get_batch(1)
            except StopIteration:
                break
        scheduler.index = 0
        scheduler._slice_idx = 0

    total_time = timeit.repeat(
        stmt=iterate_scheduler,
        number=NUMBER_OF_EXECUTIONS,
        repeat=REPETITIONS,
    )
    average_time = min(total_time) / NUMBER_OF_EXECUTIONS
    assert average_time < time_limit, (
        f"Scheduler took too long: {average_time:.2f}s"
    )


# ============================
# Prefetching Scheduler Tests
# ============================


@pytest.mark.parametrize("mock_hdf5_files", [1], indirect=True)
def test_prefetch_batch_shapes(mock_hdf5_files):
    """Test that `PrefetchingFlowFieldScheduler` preserves flow field dimensions.

    Verifies that the prefetching wrapper correctly passes through batches 
    with the expected shape while managing its background thread.
    """
    files, dims = mock_hdf5_files
    scheduler = HDF5FlowFieldScheduler(files, loop=False)
    prefetch = PrefetchingFlowFieldScheduler(
        scheduler=scheduler, batch_size=3, buffer_size=2
    )
    try:
        batch = prefetch.get_batch(3)
        expected_shape = (3, dims["x_dim"], dims["z_dim"], 2)
        flows = batch.flow_fields
        assert flows.shape == expected_shape, f"Expected prefetch shape {expected_shape}, got {flows.shape}"
    finally:
        prefetch.shutdown()


@pytest.mark.parametrize("mock_hdf5_files", [1], indirect=True)
def test_prefetch_batch_size_validation(mock_hdf5_files):
    files, _ = mock_hdf5_files
    scheduler = HDF5FlowFieldScheduler(files, loop=False)
    prefetch = PrefetchingFlowFieldScheduler(scheduler=scheduler, batch_size=4)

    with pytest.raises(ValueError, match="Batch size .* does not match"):
        prefetch.get_batch(8)
    prefetch.shutdown()


@pytest.mark.parametrize("mock_hdf5_files", [1], indirect=True)
def test_prefetch_scheduler_exhaustion(mock_hdf5_files):
    files, _ = mock_hdf5_files
    scheduler = HDF5FlowFieldScheduler(files, loop=False)
    prefetch = PrefetchingFlowFieldScheduler(
        scheduler=scheduler, batch_size=2, buffer_size=2
    )

    results = []
    try:
        while True:
            results.append(prefetch.get_batch(2))
    except StopIteration:
        pass

    assert len(results) > 0, "Prefetch scheduler should have returned at least one batch"
    prefetch.shutdown()


@pytest.mark.parametrize("mock_hdf5_files", [1], indirect=True)
def test_prefetch_scheduler_shutdown(mock_hdf5_files):
    files, _ = mock_hdf5_files
    scheduler = HDF5FlowFieldScheduler(files, loop=True)
    prefetch = PrefetchingFlowFieldScheduler(scheduler=scheduler, batch_size=2)
    prefetch.get_batch(2)
    prefetch.shutdown()
    assert not prefetch._thread.is_alive(), "Prefetching thread should be stopped after shutdown"


@pytest.mark.parametrize("bad_include_images", ["bad_value", None, 123])
@pytest.mark.parametrize("mock_numpy_files", [1], indirect=True)
def test_mat_scheduler_invalid_include_images(
    bad_include_images, mock_numpy_files
):
    """Test that invalid `include_images` values raise a ValueError."""
    files, _ = mock_numpy_files
    with pytest.raises(
        ValueError, match="include_images must be a boolean value."
    ):
        NumpyFlowFieldScheduler.from_config(
            {
                "file_list": files,
                "include_images": bad_include_images,
            }
        )


def test_numpy_scheduler_outputs_files(mock_numpy_files):
    """Test that the scheduler returns correct file paths in the batch."""
    files, dims = mock_numpy_files
    scheduler = NumpyFlowFieldScheduler.from_config(
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
        assert output.files is not None, "Expected file paths to be present"
        assert len(output.files) == batch_size, f"Expected {batch_size} files, got {len(output.files)}"

        for file_path in output.files:
            assert os.path.basename(file_path) in [
                os.path.basename(f) for f in files
            ], f"File {os.path.basename(file_path)} does not match any input files"


def test_hdf5_scheduler_outputs_files(mock_hdf5_files):
    """Test that the scheduler returns correct file paths in the batch."""
    files, dims = mock_hdf5_files
    scheduler = HDF5FlowFieldScheduler.from_config(
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
        assert output.files is not None, "Expected file paths to be present"
        assert len(output.files) == batch_size, f"Expected {batch_size} files, got {len(output.files)}"

        for file_path in output.files:
            assert os.path.basename(file_path) in [
                os.path.basename(f) for f in files
            ], f"File {os.path.basename(file_path)} does not match any input files"
