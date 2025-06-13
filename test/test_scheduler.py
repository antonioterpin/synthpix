import os
import time
import timeit

import h5py
import numpy as np
import pytest

from synthpix.scheduler import (
    BaseFlowFieldScheduler,
    EpisodicFlowFieldScheduler,
    HDF5FlowFieldScheduler,
    MATFlowFieldScheduler,
    NumpyFlowFieldScheduler,
    PrefetchingFlowFieldScheduler,
)
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
            file[temp_file_key].shape[2],
            2,
        )
    actual_shape = scheduler.get_flow_fields_shape()
    assert (
        actual_shape == expected_shape
    ), f"Expected {expected_shape}, got {actual_shape}"


@pytest.mark.parametrize("mock_numpy_files", [2], indirect=True)
def test_numpy_scheduler_iteration(mock_numpy_files):
    files, dims = mock_numpy_files
    scheduler = NumpyFlowFieldScheduler(files, randomize=False, loop=False)

    count = 0
    for flow in scheduler:
        assert isinstance(flow, np.ndarray)
        assert flow.shape == (dims["height"], dims["width"], 2)
        count += 1

    assert count == 2


def test_numpy_scheduler_shape(mock_numpy_files):
    files, dims = mock_numpy_files
    scheduler = NumpyFlowFieldScheduler(files)
    shape = scheduler.get_flow_fields_shape()
    assert shape == (dims["height"], dims["width"], 2)


@pytest.mark.parametrize("mock_numpy_files", [2], indirect=True)
def test_numpy_scheduler_init_flags(mock_numpy_files):
    files, _ = mock_numpy_files
    scheduler = NumpyFlowFieldScheduler(files, randomize=True, loop=True)

    assert scheduler.randomize is True
    assert scheduler.loop is True
    assert scheduler.epoch == 0
    assert scheduler.index == 0


def test_numpy_scheduler_invalid_ext(tmp_path):
    bad_file = tmp_path / "invalid.txt"
    bad_file.write_text("invalid content")

    with pytest.raises(
        ValueError, match="All files must be numpy files " "with '.npy' extension"
    ):
        NumpyFlowFieldScheduler([str(bad_file)])


@pytest.mark.parametrize("mock_numpy_files", [2], indirect=True)
def test_numpy_scheduler_get_batch(mock_numpy_files):
    files, dims = mock_numpy_files
    scheduler = NumpyFlowFieldScheduler(files)

    batch_size = len(files)
    batch = scheduler.get_batch(batch_size)
    assert isinstance(batch, np.ndarray)
    assert batch.shape == (batch_size, dims["height"], dims["width"], 2)


@pytest.mark.parametrize("mock_numpy_files", [2], indirect=True)
def test_numpy_scheduler_with_images(mock_numpy_files):
    files, dims = mock_numpy_files
    scheduler = NumpyFlowFieldScheduler(files, include_images=True)

    # when including images, iteration returns dicts with flow and images
    for output in scheduler:
        assert isinstance(output, dict)
        assert set(output.keys()) == {"flow", "img_prev", "img_next"}

        flow = output["flow"]
        img_prev = output["img_prev"]
        img_next = output["img_next"]

        assert isinstance(flow, np.ndarray)
        assert flow.shape == (dims["height"], dims["width"], 2)
        assert isinstance(img_prev, np.ndarray)
        assert img_prev.shape == (dims["height"], dims["width"], 3)
        assert isinstance(img_next, np.ndarray)
        assert img_next.shape == (dims["height"], dims["width"], 3)


@pytest.mark.parametrize("mock_mat_files", [2], indirect=True)
def test_mat_scheduler_iteration(mock_mat_files):
    files, _ = mock_mat_files
    scheduler = MATFlowFieldScheduler(files, randomize=False, loop=False)

    count = 0
    for flow in scheduler:
        assert isinstance(flow, np.ndarray)
        assert flow.shape == (256, 256, 2)
        count += 1

    assert count == 2


@pytest.mark.parametrize("mock_mat_files", [1], indirect=True)
def test_mat_scheduler_shape(mock_mat_files):
    files, _ = mock_mat_files
    scheduler = MATFlowFieldScheduler(files)
    shape = scheduler.get_flow_fields_shape()
    assert shape == (256, 256)


@pytest.mark.parametrize("mock_mat_files", [2], indirect=True)
def test_mat_scheduler_init_flags(mock_mat_files):
    files, _ = mock_mat_files
    scheduler = MATFlowFieldScheduler(files, randomize=True, loop=True)

    assert scheduler.randomize is True
    assert scheduler.loop is True
    assert scheduler.epoch == 0
    assert scheduler.index == 0


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
    assert scheduler.file_list == mat_files


@pytest.mark.parametrize("mock_mat_files", [1], indirect=True)
def test_mat_scheduler_get_batch(mock_mat_files):
    files, _ = mock_mat_files
    scheduler = MATFlowFieldScheduler(files)

    batch_size = len(files)
    batch = scheduler.get_batch(batch_size)
    assert isinstance(batch, np.ndarray)
    assert batch.shape == (batch_size, 256, 256, 2)


@pytest.mark.parametrize("mock_mat_files", [2], indirect=True)
def test_mat_scheduler_with_images(mock_mat_files):
    files, _ = mock_mat_files
    scheduler = MATFlowFieldScheduler(
        files, include_images=True, output_shape=(256, 256)
    )

    for output in scheduler:
        assert isinstance(output, dict)
        assert set(output.keys()) == {"flow", "img_prev", "img_next"}

        flow = output["flow"]
        img_prev = output["img_prev"]
        img_next = output["img_next"]

        assert isinstance(flow, np.ndarray)
        assert flow.shape == (256, 256, 2)
        assert isinstance(img_prev, np.ndarray)
        assert img_prev.shape == (256, 256)
        assert isinstance(img_next, np.ndarray)
        assert img_next.shape == (256, 256)


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
        expected_shape = (dims["x_dim"], dims["z_dim"], 2)
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
            dims["z_dim"],
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


# ============================
# Prefetching Scheduler Tests
# ============================


@pytest.mark.parametrize("mock_hdf5_files", [1], indirect=True)
def test_prefetch_batch_shapes(mock_hdf5_files):
    files, dims = mock_hdf5_files
    scheduler = HDF5FlowFieldScheduler(files, loop=False)
    prefetch = PrefetchingFlowFieldScheduler(
        scheduler=scheduler, batch_size=3, buffer_size=2
    )
    try:
        batch = prefetch.get_batch(3)
        expected_shape = (3, dims["x_dim"], dims["z_dim"], 2)
        assert batch.shape == expected_shape
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
        print("Scheduler exhausted, stopping iteration.")
        pass

    assert len(results) > 0
    prefetch.shutdown()


@pytest.mark.parametrize("mock_hdf5_files", [1], indirect=True)
def test_prefetch_scheduler_shutdown(mock_hdf5_files):
    files, _ = mock_hdf5_files
    scheduler = HDF5FlowFieldScheduler(files, loop=True)
    prefetch = PrefetchingFlowFieldScheduler(scheduler=scheduler, batch_size=2)
    prefetch.get_batch(2)
    prefetch.shutdown()
    assert not prefetch._thread.is_alive()


# ============================
# Episodic Scheduler Tests
# ============================


@pytest.mark.parametrize("mock_mat_files", [64], indirect=True)  # 64 frames
def test_episode_iteration(mock_mat_files):
    files, dims = mock_mat_files
    H, W = dims["height"], dims["width"]

    batch_size = 4
    episode_length = 8

    base = MATFlowFieldScheduler(files, loop=False, output_shape=(H, W))
    epi = EpisodicFlowFieldScheduler(
        scheduler=base, batch_size=batch_size, episode_length=episode_length, seed=123
    )
    steps = []
    for t, (batch) in enumerate(epi):
        # every batch is (batch_size, H, W, 2)
        assert isinstance(batch, np.ndarray)
        assert batch.shape == (batch_size, H, W, 2)
        steps.append(t)
        if t == episode_length - 1:  # horizon reached
            break
    assert steps == list(range(8))  # exactly one episode
    assert epi.steps_remaining() == 0


@pytest.mark.parametrize("mock_mat_files", [64], indirect=True)  # 64 frames
def test_reset_episode_resamples(mock_mat_files):
    files, dims = mock_mat_files
    H, W = dims["height"], dims["width"]

    batch_size = 4
    episode_length = 8

    base = MATFlowFieldScheduler(files, loop=False, output_shape=(H, W))
    epi = EpisodicFlowFieldScheduler(
        scheduler=base, batch_size=batch_size, episode_length=episode_length, seed=123
    )

    reordered_files = base.file_list
    first0 = next(epi)  # t = 0
    epi.reset_episode()
    post_reset_files = base.file_list
    second0 = next(epi)  # new t = 0
    assert epi.steps_remaining() == episode_length - 1
    # ensure we didn’t get the exact same files twice
    assert not np.array_equal(first0, second0)

    # Check which files were used by opening the files
    # with a regular scheduler
    base.file_list = reordered_files[:batch_size]
    base.reset()
    rightfirst0 = base.get_batch(batch_size)
    assert np.array_equal(first0, rightfirst0)

    base.file_list = post_reset_files[:batch_size]
    base.reset()
    rightsecond0 = base.get_batch(batch_size)
    assert np.array_equal(second0, rightsecond0)


@pytest.mark.parametrize("mock_mat_files", [64], indirect=True)  # 64 frames
def test_steps_remaining(mock_mat_files):
    files, dims = mock_mat_files
    H, W = dims["height"], dims["width"]

    batch_size = 4
    episode_length = 8

    base = MATFlowFieldScheduler(files, loop=False, output_shape=(H, W))
    epi = EpisodicFlowFieldScheduler(
        scheduler=base, batch_size=batch_size, episode_length=episode_length, seed=123
    )

    assert epi.steps_remaining() == episode_length
    next(epi)
    next(epi)
    assert epi.steps_remaining() == episode_length - 2


# ======================================
# Prefetcher flushes unfinished episode
# ======================================


@pytest.mark.parametrize("mock_mat_files", [12], indirect=True)
def test_prefetch_next_episode_flush(mock_mat_files):
    # TODO: this test is very hacky

    files, dims = mock_mat_files
    H, W = dims["height"], dims["width"]

    batch_size = 2
    episode_length = 4
    buffer_size = 8

    base = MATFlowFieldScheduler(files, loop=False, output_shape=(H, W))
    epi = EpisodicFlowFieldScheduler(
        base, batch_size=batch_size, episode_length=episode_length
    )
    pre = PrefetchingFlowFieldScheduler(
        epi, batch_size=batch_size, buffer_size=buffer_size
    )

    # Check that the episodic scheduler hasn't started
    start = epi.steps_remaining()
    assert start == episode_length

    # Read two files from the prefetcher
    for t, _ in enumerate(pre):
        if t == 1:
            break

    # wait for prefetching to finish
    time.sleep(0.1)

    # Manually read the files that are supposed to be the first batch
    # of the next episode
    # This is an empirical test, it works only with this episode length
    new_episode_paths = [files[3], files[2]]

    # early reset
    pre.next_episode()

    # Get the batch from the prefetcher
    files = pre.get_batch(batch_size)

    # Compare the files with the expected ones
    new_base = MATFlowFieldScheduler(new_episode_paths, loop=False, output_shape=(H, W))
    right_new_episode = new_base.get_batch(batch_size)

    assert np.array_equal(files, right_new_episode)

    pre.shutdown()


@pytest.mark.parametrize("mock_mat_files", [12], indirect=True)
def test_prefetch_full_episode(mock_mat_files):
    files, dims = mock_mat_files
    H, W = dims["height"], dims["width"]

    batch_size = 2
    episode_length = 4
    buffer_size = 8

    base = MATFlowFieldScheduler(files, loop=False, output_shape=(H, W))
    epi = EpisodicFlowFieldScheduler(
        base,
        batch_size=batch_size,
        episode_length=episode_length,
    )
    pre = PrefetchingFlowFieldScheduler(
        epi, batch_size=batch_size, buffer_size=buffer_size
    )

    # Exhaust the prefetcher
    for t, batch in enumerate(pre):
        print(f"Prefetching batch {t}")
        if t == episode_length - 1:
            final_batch = batch
            break

    pre.shutdown()

    final_files = [files[10], files[9]]
    print(f"Final files: {final_files}")
    new_base = MATFlowFieldScheduler(final_files, loop=False, output_shape=(H, W))

    right_new_episode = new_base.get_batch(batch_size)
    assert np.array_equal(final_batch, right_new_episode)


@pytest.mark.parametrize("mock_mat_files", [12], indirect=True)
def test_prefetch_full_episode_next_episode(mock_mat_files):
    files, dims = mock_mat_files
    H, W = dims["height"], dims["width"]

    batch_size = 2
    episode_length = 4
    buffer_size = 8

    base = MATFlowFieldScheduler(files, loop=False, output_shape=(H, W))
    epi = EpisodicFlowFieldScheduler(
        base,
        batch_size=batch_size,
        episode_length=episode_length,
    )
    pre = PrefetchingFlowFieldScheduler(
        epi, batch_size=batch_size, buffer_size=buffer_size
    )

    # Exhaust the first episode
    for t, _ in enumerate(pre):
        if t == episode_length - 1:
            break

    pre.next_episode()

    new_episode = pre.get_batch(batch_size)

    next_batch_files = [files[3], files[2]]
    new_base = MATFlowFieldScheduler(next_batch_files, loop=False, output_shape=(H, W))
    right_new_episode = new_base.get_batch(batch_size)
    assert np.array_equal(new_episode, right_new_episode)

    pre.shutdown()
