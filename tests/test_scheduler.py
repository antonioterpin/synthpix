import os
import timeit
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from synthpix.scheduler import (
    BaseFlowFieldScheduler,
    HDF5FlowFieldScheduler,
    NumpyFlowFieldScheduler,
    PrefetchingFlowFieldScheduler,
)
from synthpix.types import SchedulerData
from synthpix.utils import load_configuration

config = load_configuration("config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_SCHEDULER"]

# ============================
# Base Class Input Validation Tests
# ============================


@pytest.mark.parametrize("file_list", [[None], [123, "invalid"], [123, "invalid"]])
def test_invalid_file_list_type(file_list):
    with pytest.raises(ValueError, match="file_list must be a list of file paths."):
        HDF5FlowFieldScheduler.from_config({"scheduler_files": file_list})


@pytest.mark.parametrize("file_list", [["nonexistent.h5"]])
def test_invalid_file_paths(file_list):
    with pytest.raises(ValueError, match=f"File {file_list[0]} does not exist."):
        HDF5FlowFieldScheduler.from_config({"scheduler_files": file_list})


@pytest.mark.parametrize("randomize", [None, 123, "invalid"])
def test_invalid_randomize(randomize, temp_file):
    with pytest.raises(ValueError, match="randomize must be a boolean value."):
        HDF5FlowFieldScheduler.from_config(
            {
                "scheduler_files": [temp_file],
                "randomize": randomize,
            }
        )


@pytest.mark.parametrize("loop", [None, 123, "invalid"])
def test_invalid_loop(loop, temp_file):
    with pytest.raises(ValueError, match="loop must be a boolean value."):
        HDF5FlowFieldScheduler.from_config(
            {
                "scheduler_files": [temp_file],
                "loop": loop,
            }
        )


@pytest.mark.parametrize(
    "file_list, randomize, loop",
    [([], True, True), ([], False, True), ([], True, False), ([], False, False)],
)
def test_empty_file_list(file_list, randomize, loop):
    with pytest.raises(ValueError, match="The file_list must not be empty."):
        HDF5FlowFieldScheduler.from_config(
            {
                "scheduler_files": file_list,
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
    files, dims = mock_numpy_files
    scheduler = NumpyFlowFieldScheduler.from_config(
        {
            "scheduler_files": files,
            "randomize": False,
            "loop": False,
        }
    )

    count = 0
    while True:
        try:
            batch = scheduler.get_batch(batch_size=1)
            flow = batch.flow_fields
            assert isinstance(flow, np.ndarray)
            assert flow.shape == (1, dims["height"], dims["width"], 2)
            count += 1
        except StopIteration:
            break

    assert count == 2


def test_numpy_scheduler_shape(mock_numpy_files):
    files, dims = mock_numpy_files
    scheduler = NumpyFlowFieldScheduler.from_config(
        {
            "scheduler_files": files,
        }
    )
    shape = scheduler.get_flow_fields_shape()
    assert shape == (dims["height"], dims["width"], 2)


@pytest.mark.parametrize("mock_numpy_files", [2], indirect=True)
def test_numpy_scheduler_init_flags(mock_numpy_files):
    files, _ = mock_numpy_files
    scheduler = NumpyFlowFieldScheduler.from_config(
        {
            "scheduler_files": files,
            "randomize": True,
            "loop": True,
        }
    )

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


def test_numpy_scheduler_invalid_npy(tmp_path):
    bad_file = tmp_path / "floww_0.npy"
    np.zeros((1, 64, 64, 2)).astype(np.float32).tofile(bad_file)

    with pytest.raises(ValueError, match=f"Bad filename: {bad_file}"):
        NumpyFlowFieldScheduler([str(bad_file)], include_images=True)


def test_numpy_scheduler_missing_images(tmp_path):
    file = tmp_path / "flow_1.npy"
    np.zeros((1, 64, 64, 2)).astype(np.float32).tofile(file)

    pattern = (
        f"Missing images for frame {1}: {tmp_path}/img_0.jpg, {tmp_path}/img_1.jpg"
    )
    with pytest.raises(FileNotFoundError, match=pattern):
        NumpyFlowFieldScheduler([str(file)], include_images=True)


@pytest.mark.parametrize("mock_numpy_files", [2], indirect=True)
def test_numpy_scheduler_get_batch(mock_numpy_files):
    files, dims = mock_numpy_files
    scheduler = NumpyFlowFieldScheduler.from_config(
        {
            "scheduler_files": files,
        }
    )

    batch_size = len(files)
    batch = scheduler.get_batch(batch_size)
    assert isinstance(batch, SchedulerData)
    assert batch.flow_fields.shape == (batch_size, dims["height"], dims["width"], 2)


@pytest.mark.parametrize("mock_numpy_files", [2], indirect=True)
def test_numpy_scheduler_with_images(mock_numpy_files):
    files, dims = mock_numpy_files
    scheduler = NumpyFlowFieldScheduler.from_config(
        {"scheduler_files": files, "include_images": True, "loop": False}
    )

    batch_size = 1
    # when including images, iteration returns dicts with flow and images
    while True:
        try:
            output = scheduler.get_batch(batch_size)
        except StopIteration:
            break
        assert isinstance(output, SchedulerData)
        assert output.images1 is not None
        assert output.images2 is not None

        flow = output.flow_fields
        img_prev = output.images1
        img_next = output.images2

        assert isinstance(flow, np.ndarray)
        assert flow.shape == (batch_size, dims["height"], dims["width"], 2)
        assert isinstance(img_prev, np.ndarray)
        assert img_prev.shape == (batch_size, dims["height"], dims["width"], 3)
        assert isinstance(img_next, np.ndarray)
        assert img_next.shape == (batch_size, dims["height"], dims["width"], 3)


@pytest.mark.parametrize("mock_numpy_files", [2], indirect=True)
def test_numpy_scheduler_loop_reset(mock_numpy_files):
    """Cover the branch where `index >= len(file_list)` and `loop is True`.

    We iterate twice through the same small dataset.  The first time the
    pointer reaches the end of the list, the scheduler should call
    `reset(reset_epoch=False)` and start a new epoch without raising StopIteration.
    """
    files, dims = mock_numpy_files
    scheduler = NumpyFlowFieldScheduler(
        file_list=files,
        loop=True,
        randomize=False,
    )

    # Read exactly two full epochs
    expected_total = len(files) * 2
    out_shapes = [
        scheduler.get_batch(1).flow_fields.shape 
        for _ in range(expected_total)
    ]

    # We should have received the right number of samples
    assert len(out_shapes) == expected_total

    # Every returned flow must have the correct shape
    assert set(out_shapes) == {(1, dims["height"], dims["width"], 2)}

    # After two complete epochs the internal index should be back at 0
    # (because reset was called when the first epoch ended).
    assert scheduler.index == len(files)


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
    assert np.allclose(flow[0, ...], good_flow)

    # No more valid files remain, so a second call must raise StopIteration
    with pytest.raises(StopIteration):
        _ = scheduler.get_batch(1)


# ============================
# Abstract Behavior Tests
# ============================


class DummyScheduler(BaseFlowFieldScheduler):
    def __init__(self, file_list, randomize=False, loop=False, key=None):
        super().__init__(file_list, randomize, loop, key)

    def load_file(self, file_path):
        return np.random.rand(4, 2, 4, 3).astype(np.float32)

    def get_next_slice(self):
        assert self._cached_data is not None
        return SchedulerData(
            flow_fields=self._cached_data.flow_fields[:, self._slice_idx, :, :2]
        )

    def get_flow_fields_shape(self):
        assert self._cached_data is not None
        return self._cached_data.flow_fields.shape[0], self._cached_data.flow_fields.shape[2] // 2, 2
    
    @classmethod
    def from_config(cls, config: dict) -> "DummyScheduler":
        return cls(
            file_list=config["scheduler_files"],
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
        "dummy_test.h5", dims={"x_dim": 4, "y_dim": 2, "z_dim": 4, "features": 3}
    )
    scheduler = DummyScheduler([tmp_file])
    count = 0
    while True:
        try:
            _ = scheduler.get_batch(1)
            count += 1
        except StopIteration:
            break
    assert count == 2
    os.remove(tmp_file)

def test_reset_calls_random_shuffle(monkeypatch, tmp_path):
    files = [tmp_path / f"f{i}.dat" for i in range(3)]
    for f in files:
        f.write_text("x")
    call_flag = {"called": False}

    def spy(key, indices):
        call_flag["called"] = True
        return jnp.flip(indices)

    monkeypatch.setattr(jax.random, "permutation", spy)

    key = jax.random.PRNGKey(0)
    sch = DummyScheduler([str(f) for f in files], randomize=True, key=key)

    assert isinstance(sch.file_list, list)
    original = sch.file_list.copy()
    sch.reset(reset_epoch=True)

    assert call_flag["called"]
    assert sch.file_list == list(reversed(original))


def test_directory_initialisation(tmp_path):
    for idx in range(3):  # create three dummy files in a tmp dir
        (tmp_path / f"flow_{idx}.dat").write_text("irrelevant")

    scheduler = DummyScheduler([str(tmp_path)])  # pass *directory*
    assert len(scheduler) == 3
    # file_list must be sorted (the Base class guarantees this)
    assert scheduler.file_list == sorted(map(str, tmp_path.iterdir()))


def test_reset_preserves_or_resets_epoch(tmp_path):
    files = [tmp_path / f"file_{i}.dat" for i in range(2)]
    for f in files:
        f.write_text("x")

    scheduler = DummyScheduler([str(f) for f in files], randomize=False, loop=False)

    # simulate progress
    scheduler.epoch = 7
    scheduler.index = 1
    scheduler._slice_idx = 1

    # --- reset without touching epoch
    scheduler.reset(reset_epoch=False)
    assert scheduler.epoch == 7  # epoch untouched
    assert scheduler.index == 0  # index rewound
    assert scheduler._slice_idx == 0  # slice counter rewound

    # --- reset with epoch reset
    scheduler.epoch = 5
    scheduler.reset(reset_epoch=True)
    assert scheduler.epoch == 0


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

    with pytest.raises(StopIteration):
        scheduler.get_batch(3)


def test_get_batch_warning_and_return(tmp_path):
    f = tmp_path / "file.dat"
    f.write_text("x")

    # ── partial-batch branch ──
    sch_partial = DummyScheduler([str(f)])
    with pytest.raises(StopIteration):
        sch_partial.get_batch(3)  # asks for >2 slices -> partial batch

    # ── success branch ──
    sch_full = DummyScheduler([str(f)])
    batch = sch_full.get_batch(2)  # exactly the available slices
    assert batch.flow_fields.shape == (2, 4, 4, 2)


def test_loop_resets_and_continues_dummy(tmp_path):
    f = tmp_path / "one.dat"
    f.write_text("x")
    sch = DummyScheduler([str(f)], loop=True, randomize=False)

    _ = sch.get_batch(1)  # slice 0
    _ = sch.get_batch(1)  # slice 1
    third = sch.get_batch(1)  # after reset → slice 0 again

    assert third.flow_fields.shape == (1, 4, 4, 2)
    # internal state: back at first file, having emitted first slice of epoch 2
    assert sch.index == 0 and sch._slice_idx == 1


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
    while True:
        try:
            batch = scheduler.get_batch(1)
            flow = batch.flow_fields
            expected_shape = (1, dims["x_dim"], dims["z_dim"], 2)
            assert flow.shape == expected_shape
            num_flows += 1
        except StopIteration:
            break

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
    while True:
        try:
            batch = scheduler.get_batch(1)
            flow = batch.flow_fields
            expected_shape = (1, dims["x_dim"], dims["z_dim"], 2)
            assert flow.shape == expected_shape
            num_flows += 1
        except StopIteration:
            break

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
        while True:
            try:
                _ = scheduler.get_batch(1)
            except StopIteration:
                break
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


@pytest.mark.parametrize("bad_include_images", ["bad_value", None, 123])
@pytest.mark.parametrize("mock_numpy_files", [1], indirect=True)
def test_mat_scheduler_invalid_include_images(bad_include_images, mock_numpy_files):
    """Test that invalid `include_images` values raise a ValueError."""
    files, _ = mock_numpy_files
    with pytest.raises(ValueError, match="include_images must be a boolean value."):
        NumpyFlowFieldScheduler.from_config(
            {
                "scheduler_files": files,
                "include_images": bad_include_images,
            }
        )
