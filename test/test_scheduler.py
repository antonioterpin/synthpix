import os
import tempfile
import timeit

import h5py
import numpy as np
import pytest

from src.sym.scheduler import FlowFieldScheduler


def create_mock_hdf5(filename, x_dim=10, y_dim=6, z_dim=5, features=3):
    path = os.path.join(tempfile.gettempdir(), filename)
    with h5py.File(path, "w") as f:
        data = np.random.rand(x_dim, y_dim, z_dim, features).astype(np.float32)
        f.create_dataset("flow", data=data)
    return path


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
    file_list = ["/shared/fluids/channel_full_ts_0004.h5"]
    with pytest.raises(ValueError, match="randomize must be a boolean value."):
        FlowFieldScheduler(file_list, randomize=randomize)


@pytest.mark.parametrize("loop", [None, 123, "invalid"])
def test_invalid_loop(loop):
    """Test that invalid loop values raise a ValueError."""
    file_list = ["/shared/fluids/channel_full_ts_0004.h5"]
    with pytest.raises(ValueError, match="loop must be a boolean value."):
        FlowFieldScheduler(file_list, loop=loop)


@pytest.mark.parametrize("prefetch", [None, 123, "invalid"])
def test_invalid_prefetch(prefetch):
    """Test that invalid prefetch values raise a ValueError."""
    file_list = ["/shared/fluids/channel_full_ts_0004.h5"]
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
    "file_list, randomize, loop",
    [
        (["/shared/fluids/channel_full_ts_0004.h5"], True, True),
        (["/shared/fluids/channel_full_ts_0008.h5"], False, True),
        (["/shared/fluids/channel_full_ts_0012.h5"], True, False),
        (["/shared/fluids/channel_full_ts_0016.h5"], False, False),
    ],
)
def test_flow_field_scheduler_init(file_list, randomize, loop):
    """Test the initialization of FlowFieldScheduler with a valid file list."""
    scheduler = FlowFieldScheduler(file_list, randomize, loop)
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
    for i in range(num_files):
        os.remove(files[i])


@pytest.mark.parametrize(
    "file_list, randomize",
    [
        (["/shared/fluids/channel_full_ts_0004.h5"], True),
        (["/shared/fluids/channel_full_ts_0008.h5"], False),
    ],
)
def test_scheduler_real_file(file_list, randomize):
    """Test the iteration over a real file."""

    scheduler = FlowFieldScheduler(file_list, randomize=randomize, loop=False)

    all_flows = []
    try:
        while True:
            all_flows.append(next(scheduler))
    except StopIteration:
        pass

    assert (
        len(all_flows) == 100
    ), "Expected 100 flow fields to be returned, got: " + str(len(all_flows))
    assert all(
        flow.shape == (1536, 1024, 2) for flow in all_flows
    ), "All flow fields should have shape (1536, 1024, 2). Got: " + str(
        [flow.shape for flow in all_flows]
    )


@pytest.mark.parametrize(
    "file_list, randomize, prefetch",
    [
        (["/shared/fluids/channel_full_ts_0004.h5"], True, True),
    ],
)
def test_scheduler_time(file_list, randomize, prefetch):
    """Test the time taken for the scheduler to iterate over one standard file."""

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
    num_executions = 1
    total_time = timeit.repeat(
        stmt=iterate_scheduler,
        number=num_executions,
    )
    average_time = min(total_time)

    assert (
        average_time < 10.0
    ), f"Scheduler took too long to iterate: {average_time:.2f} seconds"
