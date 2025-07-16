import h5py
import numpy as np
import pytest
import scipy.io

from synthpix.scheduler import EpisodicFlowFieldScheduler, MATFlowFieldScheduler
from synthpix.utils import load_configuration

config = load_configuration("config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_SCHEDULER"]


@pytest.mark.parametrize("bad_include_images", ["bad_value", None, 123])
def test_mat_scheduler_invalid_include_images(bad_include_images, mock_mat_files):
    """Test that invalid `include_images` values raise a ValueError."""
    files, _ = mock_mat_files
    with pytest.raises(ValueError, match="include_images must be a boolean value."):
        MATFlowFieldScheduler.from_config(
            {
                "scheduler_files": files,
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
                "scheduler_files": files,
                "output_shape": bad_output_shape,
            }
        )


@pytest.mark.parametrize("bad_output_shape", [(256, -1), (0, 256), (256, 0), (-1, -1)])
def test_mat_scheduler_invalid_output_shape_values(bad_output_shape, mock_mat_files):
    """Test that invalid `output_shape` values raise a ValueError."""
    files, _ = mock_mat_files
    with pytest.raises(
        ValueError, match="output_shape must contain positive integers."
    ):
        MATFlowFieldScheduler.from_config(
            {
                "scheduler_files": files,
                "output_shape": bad_output_shape,
            }
        )


@pytest.mark.parametrize("mock_mat_files", [2], indirect=True)
def test_mat_scheduler_iteration(mock_mat_files):
    files, _ = mock_mat_files
    scheduler = MATFlowFieldScheduler.from_config(
        {
            "scheduler_files": files,
            "randomize": False,
            "loop": False,
        }
    )

    count = 0
    for flow in scheduler:
        assert isinstance(flow, np.ndarray)
        assert flow.shape == (256, 256, 2)
        count += 1

    assert count == 2


@pytest.mark.parametrize("mock_mat_files", [1], indirect=True)
def test_mat_scheduler_shape(mock_mat_files):
    files, _ = mock_mat_files
    scheduler = MATFlowFieldScheduler.from_config(
        {
            "scheduler_files": files,
        }
    )
    shape = scheduler.get_flow_fields_shape()
    assert shape == (256, 256)


@pytest.mark.parametrize("mock_mat_files", [2], indirect=True)
def test_mat_scheduler_init_flags(mock_mat_files):
    files, _ = mock_mat_files
    scheduler = MATFlowFieldScheduler.from_config(
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


def test_path_is_hdf5_nonexistent():
    """_path_is_hdf5 should gracefully handle a missing file."""
    assert MATFlowFieldScheduler._path_is_hdf5("does_not_exist_123.mat") is False
    assert MATFlowFieldScheduler._path_is_hdf5("does_not_exist_123.hdf5") is False
    assert MATFlowFieldScheduler._path_is_hdf5("does_not_exist_123.npy") is False


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
    loader = MATFlowFieldScheduler([fpath], include_images=False, output_shape=(4, 4))

    data = loader.load_file(fpath)

    assert set(data) == {"V"}
    assert data["V"].shape == (4, 4, 2)
    np.testing.assert_array_equal(data["V"], V)


def test_mat_with_images_resize(tmp_path):
    """Images smaller than output shape -> image & flow resized/scaled."""
    I0 = _rand_image((2, 2))
    I1 = _rand_image((2, 2))
    V = np.ones((2, 2, 2), dtype=np.float32)  # all-ones flow
    fpath = make_mat(tmp_path, "img_resize", V=V, I0=I0, I1=I1)

    loader = MATFlowFieldScheduler([fpath], include_images=True, output_shape=(4, 4))
    data = loader.load_file(fpath)

    # images resized to 4×4
    assert data["I0"].shape == data["I1"].shape == (4, 4)
    # flow resized and *scaled* by factor 2 on both axes
    assert data["V"].shape == (4, 4, 2)
    assert np.allclose(data["V"][..., 0], 2.0)
    assert np.allclose(data["V"][..., 1], 2.0)


def test_flow_transposed(tmp_path):
    """Flow stored as (2, H, W) triggers transpose branch."""
    V = _rand_flow((2, 4, 4))
    fpath = make_mat(tmp_path, "transpose", V=V)

    loader = MATFlowFieldScheduler([fpath], output_shape=(4, 4))
    data = loader.load_file(fpath)

    assert data["V"].shape == (4, 4, 2)
    # spot-check one value to prove correct permutation
    assert np.isclose(data["V"][1, 2, 0], V[0, 1, 2])
    assert np.isclose(data["V"][1, 2, 1], V[1, 1, 2])


def test_hdf5_fallback(monkeypatch, tmp_path):
    """SciPy raises NotImplementedError -> h5py branch loads data."""
    # Patch scipy.io.loadmat to always raise NotImplementedError
    monkeypatch.setattr(
        scipy.io, "loadmat", lambda *a, **kw: (_ for _ in ()).throw(NotImplementedError)
    )
    V = _rand_flow((4, 4, 2))
    fpath = make_hdf5(tmp_path, "v73", V=V)

    loader = MATFlowFieldScheduler([fpath], output_shape=(4, 4))
    data = loader.load_file(fpath)

    assert data["V"].shape == (4, 4, 2)
    np.testing.assert_array_equal(data["V"], V)


def test_missing_V_raises(tmp_path):
    """No 'V' present should raise."""
    fpath = make_mat(tmp_path, "noV", I0=_rand_image((4, 4)), I1=_rand_image((4, 4)))
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
        scipy.io, "loadmat", lambda *a, **kw: (_ for _ in ()).throw(ValueError("boom"))
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
    img_prevs, img_nexts, flows = scheduler.get_batch(batch_size)

    assert img_prevs.shape == (batch_size, 256, 256)
    assert img_nexts.shape == (batch_size, 256, 256)
    assert flows.shape == (batch_size, 256, 256, 2)
    assert img_prevs.dtype == np.float32
    assert img_nexts.dtype == np.float32
    assert flows.dtype == np.float32


def test_next_loop_reset(tmp_path):
    flow = _rand_flow((4, 4, 2))
    fpath = make_mat(tmp_path, "one", V=flow)

    sched = MATFlowFieldScheduler(
        [fpath],
        loop=True,
        randomize=False,
        output_shape=(4, 4),
    )

    first = next(sched)  # consumes file, index -> 1
    second = next(sched)  # triggers reset() branch

    # Both iterations must yield the same (resized) flow field
    np.testing.assert_array_equal(first, flow)
    np.testing.assert_array_equal(second, flow)

    # After two successful returns we are back at "end of list"
    assert sched.index == 1  # 0 → reset() -> +1 during 2nd return
    assert sched.loop is True  # sanity-check configuration


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

    sample = next(sched)  # bad file skipped, good file returned

    # The scheduler must now point to the good file it cached
    assert sched._cached_file == good_path
    np.testing.assert_array_equal(sample, good_flow)

    # Both list entries have been consumed (bad skipped, good returned)
    assert sched.index == 2


@pytest.mark.parametrize("mock_mat_files", [2], indirect=True)
def test_mat_scheduler_get_batch_too_large_raises_stopiteration(mock_mat_files, caplog):
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

    with pytest.raises(StopIteration):
        scheduler.get_batch(batch_size)


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
    assert "V" in data
    assert "lvl1/lvl2/extra" in data  # proves recursion/flattening
    np.testing.assert_array_equal(data["V"], flow)
