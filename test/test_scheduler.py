import os
import timeit

import h5py
import numpy as np
import pytest

from synthpix.scheduler import BaseFlowFieldScheduler, HDF5FlowFieldScheduler
from synthpix.utils import load_configuration

config = load_configuration("config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_SCHEDULER"]

# ============================
# Base Class Input Validation Tests
# ============================


@pytest.mark.parametrize("file_list", [[None], [123, "invalid"], [123, "invalid"]])
def test_invalid_file_list_type(file_list):
    with pytest.raises(ValueError, match="All file paths must be strings."):
        HDF5FlowFieldScheduler(file_list)


@pytest.mark.parametrize("file_list", [["nonexistent.h5"]])
def test_invalid_file_paths(file_list):
    with pytest.raises(ValueError, match=f"File {file_list[0]} does not exist."):
        HDF5FlowFieldScheduler(file_list)


@pytest.mark.parametrize("randomize", [None, 123, "invalid"])
def test_invalid_randomize(randomize, temp_file):
    with pytest.raises(ValueError, match="randomize must be a boolean value."):
        HDF5FlowFieldScheduler([temp_file], randomize=randomize)


@pytest.mark.parametrize("loop", [None, 123, "invalid"])
def test_invalid_loop(loop, temp_file):
    with pytest.raises(ValueError, match="loop must be a boolean value."):
        HDF5FlowFieldScheduler([temp_file], loop=loop)


@pytest.mark.parametrize(
    "file_list, randomize, loop",
    [([], True, True), ([], False, True), ([], True, False), ([], False, False)],
)
def test_empty_file_list(file_list, randomize, loop):
    with pytest.raises(ValueError, match="The file_list must not be empty."):
        HDF5FlowFieldScheduler(file_list, randomize=randomize, loop=loop)


# ============================
# Subclass-Specific Validation
# ============================


def test_non_hdf5_file(temp_txt_file):
    """Test that non-HDF5 files raise a ValueError."""
    with pytest.raises(
        ValueError, match="All files must be HDF5 files with .h5 extension."
    ):
        HDF5FlowFieldScheduler(file_list=temp_txt_file)


def test_hdf5_shape(temp_file):
    """Test that the HDF5 file has the correct shape."""
    scheduler = HDF5FlowFieldScheduler(file_list=[temp_file])
    with h5py.File(temp_file, "r") as file:
        temp_file_key = list(file.keys())[0]
        expected_shape = (
            file[temp_file_key].shape[0],
            file[temp_file_key].shape[2] // 2,
            2,
        )
    actual_shape = scheduler.get_flow_fields_shape()
    assert (
        actual_shape == expected_shape
    ), f"Expected {expected_shape}, got {actual_shape}"


# ============================
# Abstract Behavior Tests
# ============================


class DummyScheduler(BaseFlowFieldScheduler):
    def __init__(self, file_list, randomize=False, loop=False):
        super().__init__(file_list, randomize, loop)

    def load_file(self, file_path):
        return np.random.rand(4, 2, 4, 3).astype(np.float32)

    def get_next_slice(self):
        return self._cached_data[:, self._slice_idx, :, :][:, :, :2]

    def get_flow_fields_shape(self):
        return self._cached_data.shape[0], self._cached_data.shape[2] // 2, 2


def test_abstract_scheduler_iteration(generate_hdf5_file):
    tmp_file = generate_hdf5_file(
        "dummy_test.h5", dims={"x_dim": 4, "y_dim": 2, "z_dim": 4, "features": 3}
    )
    scheduler = DummyScheduler([tmp_file])
    count = sum(1 for _ in scheduler)
    assert count == 2
    os.remove(tmp_file)


# ============================
# Core Functional Tests
# ============================


@pytest.mark.parametrize(
    "randomize, loop", [(True, True), (False, True), (True, False), (False, False)]
)
def test_flow_field_scheduler_init(randomize, loop, temp_file):
    scheduler = HDF5FlowFieldScheduler([temp_file], randomize, loop)
    assert scheduler.randomize is randomize
    assert scheduler.loop is loop
    assert scheduler.epoch == 0
    assert scheduler.index == 0
    assert scheduler._slice_idx == 0


@pytest.mark.parametrize(
    "hdf5_test_dims", [{"x_dim": 10, "y_dim": 6, "z_dim": 20}], indirect=True
)
@pytest.mark.parametrize("mock_hdf5_files", [2], indirect=True)
def test_scheduler_iteration(mock_hdf5_files):
    files, dims = mock_hdf5_files

    scheduler = HDF5FlowFieldScheduler(file_list=files, randomize=True, loop=False)

    num_flows = 0
    for flow in scheduler:
        expected_shape = (dims["x_dim"], dims["z_dim"] // 2, 2)
        assert flow.shape == expected_shape
        num_flows += 1

    assert num_flows == len(files) * dims["y_dim"]


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
    assert scheduler.randomize is True
    assert scheduler.loop is False

    # Validate iteration
    num_flows = 0
    for flow in scheduler:
        # Validate the shape of each flow
        expected_shape = (
            dims["x_dim"],
            dims["z_dim"] // 2,
            2,
        )  # Assumes slicing z_dim and selecting 2 features
        assert flow.shape == expected_shape
        num_flows += 1

    # Validate the total number of flows
    assert num_flows == len(files) * dims["y_dim"]


@pytest.mark.parametrize("randomize", [True, False])
@pytest.mark.parametrize("mock_hdf5_files", [1], indirect=True)
def test_scheduler_time(randomize, mock_hdf5_files):
    files, dims = mock_hdf5_files
    CI = os.environ.get("CI") == "true"
    time_limit = 0.01 if CI else 10

    scheduler = HDF5FlowFieldScheduler(file_list=files, randomize=randomize, loop=False)

    def iterate_scheduler():
        for _ in scheduler:
            pass
        scheduler.epoch = 0
        scheduler.index = 0
        scheduler._slice_idx = 0

    total_time = timeit.repeat(
        stmt=iterate_scheduler,
        number=NUMBER_OF_EXECUTIONS,
        repeat=REPETITIONS,
    )
    average_time = min(total_time) / NUMBER_OF_EXECUTIONS
    assert average_time < time_limit, f"Scheduler took too long: {average_time:.2f}s"
