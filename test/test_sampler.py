import os
import tempfile
import timeit

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.sym.image_sampler import SyntheticImageSampler
from src.sym.processing import generate_images_from_flow
from src.sym.scheduler import FlowFieldScheduler
from src.utils import load_configuration

config = load_configuration("config/timeit.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_SAMPLER"]


def create_mock_hdf5(filename, x_dim=3, y_dim=4, z_dim=5, features=3):
    path = os.path.join(tempfile.gettempdir(), filename)
    with h5py.File(path, "w") as f:
        data = np.random.rand(x_dim, y_dim, z_dim * 2, features).astype(np.float32)
        f.create_dataset("flow", data=data)
    return path


def dummy_img_gen_fn(
    key,
    flow_field,
    position_bounds,
    image_shape,
    img_offset,
    num_images,
    num_particles,
    p_hide_img1,
    p_hide_img2,
    diameter_range,
    intensity_range,
    rho_range,
    dt,
):
    """Simulates generating a batch of synthetic images based on a single key."""
    return (
        jnp.ones((num_images, image_shape[0], image_shape[1]))
        * (jnp.sum(flow_field) + jnp.sum(key)),
        jnp.ones((num_images, image_shape[0], image_shape[1]))
        * (jnp.sum(flow_field) + jnp.sum(key)),
    )


def test_temp_file():
    """Create a temporary file for all other input validation tests."""
    filename = "mock_data.h5"
    path = create_mock_hdf5(filename)
    assert os.path.isfile(path), f"Temporary file {path} was not created."


@pytest.mark.parametrize("scheduler", [None, "invalid_scheduler"])
def test_invalid_scheduler(scheduler):
    """Test that invalid scheduler raises a ValueError."""
    with pytest.raises(ValueError, match="scheduler must be an iterable object."):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=10,
            seed=0,
        )


def test_invalid_img_gen_fn():
    """Test that invalid img_gen_fn raises a ValueError."""
    filename = "mock_data.h5"
    file_path = os.path.join(tempfile.gettempdir(), filename)
    files = [file_path]
    scheduler = FlowFieldScheduler(files, loop=False, prefetch=False)
    with pytest.raises(ValueError, match="img_gen_fn must be a callable function."):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=None,
            batch_size=2,
            images_per_field=10,
            seed=0,
        )


@pytest.mark.parametrize("batch_size", [-1, 0, 1.5])
def test_invalid_batch_size(batch_size):
    """Test that invalid batch_size raises a ValueError."""
    filename = "mock_data.h5"
    file_path = os.path.join(tempfile.gettempdir(), filename)
    files = [file_path]
    scheduler = FlowFieldScheduler(files, loop=False, prefetch=False)
    with pytest.raises(ValueError, match="batch_size must be a positive integer."):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=batch_size,
            images_per_field=10,
            seed=0,
        )


@pytest.mark.parametrize("images_per_field", [-1, 0, 1.5])
def test_invalid_images_per_field(images_per_field):
    """Test that invalid images_per_field raises a ValueError."""
    filename = "mock_data.h5"
    file_path = os.path.join(tempfile.gettempdir(), filename)
    files = [file_path]
    scheduler = FlowFieldScheduler(files, loop=False, prefetch=False)
    with pytest.raises(
        ValueError, match="images_per_field must be a positive integer."
    ):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=images_per_field,
            seed=0,
        )


@pytest.mark.parametrize(
    "image_shape", [(-1, 128), (128, -1), (0, 128), (128, 0), (128.5, 128.5)]
)
def test_invalid_image_shape(image_shape):
    """Test that invalid image_shape raises a ValueError."""
    filename = "mock_data.h5"
    file_path = os.path.join(tempfile.gettempdir(), filename)
    files = [file_path]
    scheduler = FlowFieldScheduler(files, loop=False, prefetch=False)
    with pytest.raises(
        ValueError, match="image_shape must be a tuple of two positive integers."
    ):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=10,
            image_shape=image_shape,
            seed=0,
        )


@pytest.mark.parametrize(
    "position_bounds", [(-1, 128), (128, -1), (0, 128), (128, 0), (128.5, 128.5)]
)
def test_invalid_position_bounds(position_bounds):
    """Test that invalid position_bounds raises a ValueError."""
    filename = "mock_data.h5"
    file_path = os.path.join(tempfile.gettempdir(), filename)
    files = [file_path]
    scheduler = FlowFieldScheduler(files, loop=False, prefetch=False)
    with pytest.raises(
        ValueError, match="position_bounds must be a tuple of two positive integers."
    ):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=10,
            position_bounds=position_bounds,
            seed=0,
        )


@pytest.mark.parametrize("num_particles", [-1, 0, 1.5])
def test_invalid_num_particles(num_particles):
    """Test that invalid num_particles raises a ValueError."""
    filename = "mock_data.h5"
    file_path = os.path.join(tempfile.gettempdir(), filename)
    files = [file_path]
    scheduler = FlowFieldScheduler(files, loop=False, prefetch=False)
    with pytest.raises(ValueError, match="num_particles must be a positive integer."):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=10,
            num_particles=num_particles,
            seed=0,
        )


@pytest.mark.parametrize("p_hide_img1", [-0.1, 1.1])
def test_invalid_p_hide_img1(p_hide_img1):
    """Test that invalid p_hide_img1 raises a ValueError."""
    filename = "mock_data.h5"
    file_path = os.path.join(tempfile.gettempdir(), filename)
    files = [file_path]
    scheduler = FlowFieldScheduler(files, loop=False, prefetch=False)
    with pytest.raises(ValueError, match="p_hide_img1 must be between 0 and 1."):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=10,
            p_hide_img1=p_hide_img1,
            seed=0,
        )


@pytest.mark.parametrize("p_hide_img2", [-0.1, 1.1])
def test_invalid_p_hide_img2(p_hide_img2):
    """Test that invalid p_hide_img2 raises a ValueError."""
    filename = "mock_data.h5"
    file_path = os.path.join(tempfile.gettempdir(), filename)
    files = [file_path]
    scheduler = FlowFieldScheduler(files, loop=False, prefetch=False)
    with pytest.raises(ValueError, match="p_hide_img2 must be between 0 and 1."):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=10,
            p_hide_img2=p_hide_img2,
            seed=0,
        )


@pytest.mark.parametrize("diameter_range", [(0, 1), (1, 0), (-1, 1)])
def test_invalid_diameter_range(diameter_range):
    """Test that invalid diameter_range raises a ValueError."""
    filename = "mock_data.h5"
    file_path = os.path.join(tempfile.gettempdir(), filename)
    files = [file_path]
    scheduler = FlowFieldScheduler(files, loop=False, prefetch=False)
    with pytest.raises(
        ValueError, match="diameter_range must be a tuple of two positive floats."
    ):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=10,
            diameter_range=diameter_range,
            seed=0,
        )


@pytest.mark.parametrize("intensity_range", [(-1, 200), (50, -1)])
def test_invalid_intensity_range(intensity_range):
    """Test that invalid intensity_range raises a ValueError."""
    filename = "mock_data.h5"
    file_path = os.path.join(tempfile.gettempdir(), filename)
    files = [file_path]
    scheduler = FlowFieldScheduler(files, loop=False, prefetch=False)
    with pytest.raises(
        ValueError, match="intensity_range must be a tuple of two non-negative floats."
    ):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=10,
            intensity_range=intensity_range,
            seed=0,
        )


@pytest.mark.parametrize("rho_range", [(-1.1, 1), (1, -1.1)])
def test_invalid_rho_range(rho_range):
    """Test that invalid rho_range raises a ValueError."""
    filename = "mock_data.h5"
    file_path = os.path.join(tempfile.gettempdir(), filename)
    files = [file_path]
    scheduler = FlowFieldScheduler(files, loop=False, prefetch=False)
    with pytest.raises(
        ValueError, match="rho_range must be a tuple of two floats between -1 and 1."
    ):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=10,
            rho_range=rho_range,
            seed=0,
        )


@pytest.mark.parametrize("dt", ["invalid_dt", jnp.array([1]), jnp.array([1.0, 2.0])])
def test_invalid_dt(dt):
    """Test that invalid dt raises a ValueError."""
    filename = "mock_data.h5"
    file_path = os.path.join(tempfile.gettempdir(), filename)
    files = [file_path]
    scheduler = FlowFieldScheduler(files, loop=False, prefetch=False)
    with pytest.raises(ValueError, match="dt must be a scalar \\(int or float\\)"):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=10,
            dt=dt,
            seed=0,
        )


@pytest.mark.parametrize(
    "batch_size, images_per_field, image_shape",
    [(4, 16, (8, 8)), (2, 8, (8, 8)), (1, 5, (8, 8))],
)
def test_synthetic_sampler_batches(batch_size, images_per_field, image_shape):
    files = [create_mock_hdf5(f"test_file_{i}.h5") for i in range(2)]
    scheduler = FlowFieldScheduler(files, loop=False, prefetch=False)

    sampler = SyntheticImageSampler(
        scheduler=scheduler,
        img_gen_fn=dummy_img_gen_fn,
        image_shape=image_shape,
        images_per_field=images_per_field,
        batch_size=batch_size,
        seed=123,
    )

    iterator = iter(sampler)
    all_batches = []
    for _ in range(images_per_field // batch_size):
        batch = next(iterator)
        all_batches.append(batch)
        assert batch[0].shape[0] >= batch_size, (
            "Each batch should have a number of "
            "images bigger or equal than batch_size"
        )
        assert batch[0][0].shape >= image_shape, "Incorrect image shape"
        assert isinstance(batch[0], jnp.ndarray), "Output should be a JAX array"

    for i, _ in enumerate(files):
        os.remove(files[i])  # Clean up the temporary file

    assert (
        len(all_batches) == images_per_field // batch_size
    ), f"Should yield {images_per_field // batch_size} batches"


@pytest.mark.parametrize("batch_size, images_per_field", [(2, 4), (1, 3)])
def test_sampler_switches_flow_fields(batch_size, images_per_field):
    files = [create_mock_hdf5("test_file_0.h5")]
    scheduler = FlowFieldScheduler(files, loop=False, prefetch=False)

    sampler = SyntheticImageSampler(
        scheduler=scheduler,
        img_gen_fn=dummy_img_gen_fn,
        images_per_field=images_per_field,
        batch_size=batch_size,
        seed=0,
    )

    batch1 = next(sampler)
    batch2 = next(sampler)

    # Clean up the temporary file
    os.remove(files[0])  # Clean up the temporary file

    assert not jnp.allclose(
        batch1[0], batch2[0]
    ), "Different flow fields should yield different image values"


@pytest.mark.parametrize(
    "image_shape, num_images, num_particles",
    [
        ((32, 32), 4, 100),
        ((64, 64), 4, 200),
    ],
)
@pytest.mark.parametrize("batch_size", [4])
def test_sampler_with_real_img_gen_fn(
    image_shape, num_images, num_particles, batch_size
):
    files = [
        create_mock_hdf5(
            "test_file_0.h5",
            x_dim=image_shape[0],
            y_dim=5,
            z_dim=image_shape[1],
            features=3,
        )
    ]
    scheduler = FlowFieldScheduler(files, loop=False, prefetch=True)
    sampler = SyntheticImageSampler(
        scheduler=scheduler,
        img_gen_fn=generate_images_from_flow,
        image_shape=image_shape,
        position_bounds=(image_shape[0] * 2, image_shape[1] * 2),
        num_particles=num_particles,
        images_per_field=num_images,
        batch_size=batch_size,
        seed=0,
    )

    batch = next(sampler)

    os.remove(files[0])  # Clean up the temporary file
    assert isinstance(
        batch[0], jnp.ndarray
    ), f"Output should be a JAX array, got {type(batch[0])}"
    assert batch[0].shape == (
        batch_size,
        *image_shape,
    ), f"Image batch should have shape {(batch_size, *image_shape)}, got {batch[0].shape}"
    assert batch[1].shape == (
        batch_size,
        *image_shape,
    ), f"Image batch should have shape {(batch_size, *image_shape)}, got {batch[1].shape}"


@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize("file_path", [["test_file_0.h5"]])
def test_speed_sampler_dummy_fn(file_path):
    batch_size = 200
    images_per_field = 1000
    x_dim = 1536
    z_dim = 2048
    image_shape = (1216, 1936)
    position_bounds = (1536, 2048)

    if len(jax.devices()) == 1:
        limit_time = 0.5
    elif len(jax.devices()) == 2:
        limit_time = 1.1e-1
    elif len(jax.devices()) == 4:
        limit_time = 7e-2

    # Create a temporary HDF5 file with mock data
    files = [
        create_mock_hdf5(
            filename=file_path[0],
            x_dim=x_dim,
            y_dim=NUMBER_OF_EXECUTIONS * REPETITIONS + 1,
            z_dim=z_dim,
        )
    ]

    scheduler = FlowFieldScheduler(files, loop=False, prefetch=True)
    sampler = SyntheticImageSampler(
        scheduler=scheduler,
        img_gen_fn=dummy_img_gen_fn,
        image_shape=image_shape,
        position_bounds=position_bounds,
        images_per_field=images_per_field,
        batch_size=batch_size,
        seed=0,
    )

    def run_sampler():
        for i, batch in enumerate(sampler):
            batch[0].block_until_ready()
            batch[1].block_until_ready()
            if i >= images_per_field // batch_size:
                break

    # Warm up the function
    run_sampler()

    total_time = timeit.repeat(
        stmt=run_sampler, number=NUMBER_OF_EXECUTIONS, repeat=REPETITIONS
    )
    avg_time = min(total_time) / NUMBER_OF_EXECUTIONS

    # Clean up the temporary file
    os.remove(files[0])

    assert (
        avg_time < limit_time
    ), f"The average time is {avg_time}, time limit: {limit_time}"


@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize("file_path", [["mock_data.h5"]])
@pytest.mark.parametrize("batch_size", [250])
@pytest.mark.parametrize("images_per_field", [1000])
@pytest.mark.parametrize("seed", [0])
def test_speed_sampler_real_fn(file_path, batch_size, images_per_field, seed):
    """Test that the sampler with the real image generation function is fast enough."""
    # Set the dimensions of the mock data to simulate the real data
    x_dim = 1536
    z_dim = 2048
    image_shape = (1216, 1936)
    position_bounds = (1536, 2048)

    # Create a temporary HDF5 file with mock data
    files = [
        create_mock_hdf5(
            filename=file_path[0],
            x_dim=x_dim,
            y_dim=NUMBER_OF_EXECUTIONS * REPETITIONS + 1,
            z_dim=z_dim,
        )
    ]

    # Check how many GPUs are available
    num_devices = len(jax.devices())

    # Limit time in seconds (depends on the number of GPUs)
    if num_devices == 1:
        limit_time = 2e-1
    elif num_devices == 2:
        limit_time = 1.1e-1
    elif num_devices == 4:
        limit_time = 7e-2

    # Create the scheduler and sampler
    scheduler = FlowFieldScheduler(files, loop=False, prefetch=True)
    sampler = SyntheticImageSampler(
        scheduler=scheduler,
        img_gen_fn=generate_images_from_flow,
        images_per_field=images_per_field,
        batch_size=batch_size,
        seed=seed,
        image_shape=image_shape,
        position_bounds=position_bounds,
    )

    def run_sampler():
        # Generates len(file_path) * images_per_field // batch_size batches
        # of size batch_size
        for i, batch in enumerate(sampler):
            batch[0].block_until_ready()
            batch[1].block_until_ready()
            if i >= len(file_path) * images_per_field // batch_size:
                break

    # Warm up the function
    run_sampler()

    # Measure the time taken to run the sampler
    total_time = timeit.repeat(
        stmt=run_sampler, number=NUMBER_OF_EXECUTIONS, repeat=REPETITIONS
    )
    avg_time = min(total_time) / NUMBER_OF_EXECUTIONS / len(file_path)

    # Clean up the temporary file
    os.remove(files[0])

    assert (
        avg_time < limit_time
    ), f"The average time is {avg_time}, time limit: {limit_time}"


def test_cleanup():
    """Cleanup function to remove temporary files."""
    files = ["mock_data.h5", "test_file_0.h5", "test_file_1.h5"]

    # Clean up the temporary files
    for file in files:
        file_path = os.path.join(tempfile.gettempdir(), file)
        if os.path.isfile(file_path):
            os.remove(file_path)
