import os
import tempfile
import timeit

import h5py
import numpy as np
import pytest

from src.sym.scheduler import FlowFieldScheduler
from src.utils import load_configuration

config = load_configuration("config/timeit.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["NUMBER_OF_EXECUTIONS"]


def create_mock_hdf5(filename, x_dim=10, y_dim=6, z_dim=5, features=3):
    path = os.path.join(tempfile.gettempdir(), filename)
    with h5py.File(path, "w") as f:
        data = np.random.rand(x_dim, y_dim, z_dim, features).astype(np.float32)
        f.create_dataset("flow", data=data)
    return path


def test_temp_file():
    """Create a temporary file for all other input validation tests."""
    filename = "mock_data.h5"
    path = create_mock_hdf5(filename)
    assert os.path.isfile(path), f"Temporary file {path} was not created."


@pytest.mark.parametrize("file_list", [[None], [123, "invalid"], [123, "invalid"]])
def test_invalid_file_list_type(file_list):
    """Test that invalid file_list types raise a ValueError."""
    with pytest.raises(ValueError, match="All file paths must be strings."):
        FlowFieldScheduler(file_list)


@pytest.mark.parametrize("file_list", [["nonexistent.h5"]])
def test_invalid_file_paths(file_list):
    """Test that invalid file paths raise a ValueError."""
    with pytest.raises(ValueError, match=f"File {file_list[0]} does not exist."):
        FlowFieldScheduler(file_list)


@pytest.mark.parametrize("file_list", [["invalid_file.txt"]])
def test_non_hdf5_file(file_list):
    """Test that non-HDF5 files raise a ValueError."""
    with pytest.raises(ValueError, match=f"File {file_list[0]} is not an HDF5 file."):
        FlowFieldScheduler(file_list)


@pytest.mark.parametrize("randomize", [None, 123, "invalid"])
def test_invalid_randomize(randomize):
    """Test that invalid randomize values raise a ValueError."""
    filename = "mock_data.h5"
    file_path = os.path.join(tempfile.gettempdir(), filename)
    file_list = [file_path]
    with pytest.raises(ValueError, match="randomize must be a boolean value."):
        FlowFieldScheduler(file_list, randomize=randomize)


@pytest.mark.parametrize("loop", [None, 123, "invalid"])
def test_invalid_loop(loop):
    """Test that invalid loop values raise a ValueError."""
    filename = "mock_data.h5"
    file_path = os.path.join(tempfile.gettempdir(), filename)
    file_list = [file_path]
    with pytest.raises(ValueError, match="loop must be a boolean value."):
        FlowFieldScheduler(file_list, loop=loop)


@pytest.mark.parametrize("prefetch", [None, 123, "invalid"])
def test_invalid_prefetch(prefetch):
    """Test that invalid prefetch values raise a ValueError."""
    filename = "mock_data.h5"
    file_path = os.path.join(tempfile.gettempdir(), filename)
    file_list = [file_path]
    with pytest.raises(ValueError, match="prefetch must be a boolean value."):
        FlowFieldScheduler(file_list, prefetch=prefetch)


@pytest.mark.parametrize(
    "file_list, randomize, loop",
    [([], True, True), ([], False, True), ([], True, False), ([], False, False)],
)
def test_empty_file_list(file_list, randomize, loop):
    """Test that an empty file list raises a ValueError."""
    with pytest.raises(ValueError, match="The file_list must not be empty."):
        FlowFieldScheduler(file_list, randomize=randomize, loop=loop)


@pytest.mark.parametrize(
    "randomize, loop, prefetch",
    [
        (True, True, True),
        (False, True, True),
        (True, False, True),
        (False, False, True),
        (True, True, False),
        (False, True, False),
        (True, False, False),
        (False, False, False),
    ],
)
def test_flow_field_scheduler_init(randomize, loop, prefetch):
    """Test the initialization of FlowFieldScheduler with a valid file list."""
    filename = "mock_data.h5"
    file_path = os.path.join(tempfile.gettempdir(), filename)
    file_list = [file_path]
    scheduler = FlowFieldScheduler(file_list, randomize, loop, prefetch)
    assert len(scheduler.file_list) == len(file_list)
    assert scheduler.randomize is randomize
    assert scheduler.loop is loop
    assert scheduler.epoch == 0
    assert scheduler.index == 0
    assert scheduler.y_sel == 0


@pytest.mark.parametrize(
    "randomize, x_dim, y_dim, z_dim, num_files",
    [(True, 10, 6, 20, 2), (False, 20, 4, 50, 4)],
)
def test_scheduler_iteration(randomize, x_dim, y_dim, z_dim, num_files):
    """Test the iteration over the FlowFieldScheduler."""

    files = [
        create_mock_hdf5(f"test_file_{i}.h5", x_dim=x_dim, y_dim=y_dim, z_dim=z_dim)
        for i in range(num_files)
    ]
    scheduler = FlowFieldScheduler(files, randomize=randomize, loop=False)

    all_flows = []
    try:
        while True:
            all_flows.append(next(scheduler))
    except StopIteration:
        pass

    for i in range(num_files):
        os.remove(files[i])

    assert (
        len(all_flows) == num_files * y_dim
    ), "Expected {} flow fields to be returned, got: {}".format(
        num_files * y_dim, len(all_flows)
    )
    assert all(
        flow.shape == (x_dim, z_dim // 2, 2) for flow in all_flows
    ), "All flow fields should have shape ({}, {}, {}). Got: {}".format(
        x_dim, z_dim // 2, 2, [flow.shape for flow in all_flows]
    )


@pytest.mark.parametrize(
    "randomize", [(True), (False)],
)
def test_scheduler_real_file(randomize):
    """Test the iteration over a real size file."""
    # Parameters for the test
    x_dim = 1536
    y_dim = 100
    z_dim = 2048
    features = 3
    
    # Create a mock HDF5 file with the specified dimensions and features    
    filename = "mock_data.h5"
    file_list = [create_mock_hdf5(filename, x_dim=x_dim, y_dim=y_dim, z_dim=z_dim, features=features)]

    scheduler = FlowFieldScheduler(file_list, randomize=randomize, loop=False)

    all_flows = []
    try:
        while True:
            all_flows.append(next(scheduler))
    except StopIteration:
        pass

    assert (
        len(all_flows) == y_dim
    ), f"Expected {y_dim} flow fields to be returned, got: {len(all_flows)}"
    assert all(
        flow.shape == (x_dim, z_dim // 2, 2) for flow in all_flows
    ), "All flow fields should have shape ({}, {}, {}). Got: {}".format(
        x_dim, z_dim // 2, 2, [flow.shape for flow in all_flows]
    )


@pytest.mark.parametrize(
    "randomize, prefetch",
    [
        (True, True),
        (False, False),
    ],
)
def test_scheduler_time(randomize, prefetch):
    """Test the time taken for the scheduler to iterate over one standard file."""
    
    if prefetch:
        time_limit = 10.0
    else:
        time_limit = 20.0
          
    # Create a mock HDF5 file with the specified dimensions
    x_dim = 1536
    y_dim = 100
    z_dim = 2048
    features = 3
    filename = "mock_data.h5"
    file_list = [create_mock_hdf5(filename, x_dim=x_dim, y_dim=y_dim, z_dim=z_dim, features=features)]

    scheduler = FlowFieldScheduler(
        file_list=file_list, randomize=randomize, loop=False, prefetch=prefetch
    )

    def iterate_scheduler():
        try:
            while True:
                next(scheduler)
        except StopIteration:
            scheduler.epoch = 0
            scheduler.index = 0
            scheduler.y_sel = 0
            pass

    # Measure the time of the scheduler iteration
    total_time = timeit.repeat(
        stmt=iterate_scheduler,
        number=NUMBER_OF_EXECUTIONS,
        repeat=REPETITIONS,
    )
    average_time = min(total_time) / NUMBER_OF_EXECUTIONS

    assert (
        average_time < time_limit
    ), f"Scheduler took too long to iterate: {average_time:.2f} seconds"


def test_cleanup():
    """Cleanup function to remove temporary files."""
    files = [
        "mock_data.h5",
        "test_file_0.h5",
        "test_file_1.h5",
        "test_file_2.h5",
        "test_file_3.h5",
    ]

    # Clean up the temporary files
    for file in files:
        file_path = os.path.join(tempfile.gettempdir(), file)
        if os.path.isfile(file_path):
            os.remove(file_path)
