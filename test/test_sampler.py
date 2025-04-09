import timeit

import jax
import jax.numpy as jnp
import pytest

from src.sym.data_generate import generate_images_from_flow
from src.sym.image_sampler import SyntheticImageSampler
from src.sym.scheduler import HDF5FlowFieldScheduler
from src.utils import load_configuration, logger

config = load_configuration("config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_SAMPLER"]


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


@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_img_gen_fn(scheduler):
    """Test that invalid img_gen_fn raises a ValueError."""
    with pytest.raises(ValueError, match="img_gen_fn must be a callable function."):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=None,
            batch_size=2,
            images_per_field=10,
            seed=0,
        )


@pytest.mark.parametrize("batch_size", [-1, 0, 1.5])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_batch_size(batch_size, scheduler):
    """Test that invalid batch_size raises a ValueError."""
    with pytest.raises(ValueError, match="batch_size must be a positive integer."):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=batch_size,
            images_per_field=10,
            seed=0,
        )


@pytest.mark.parametrize("images_per_field", [-1, 0, 1.5])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_images_per_field(images_per_field, scheduler):
    """Test that invalid images_per_field raises a ValueError."""
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
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_image_shape(image_shape, scheduler):
    """Test that invalid image_shape raises a ValueError."""
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
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_position_bounds(position_bounds, scheduler):
    """Test that invalid position_bounds raises a ValueError."""
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
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_num_particles(num_particles, scheduler):
    """Test that invalid num_particles raises a ValueError."""
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
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_p_hide_img1(p_hide_img1, scheduler):
    """Test that invalid p_hide_img1 raises a ValueError."""
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
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_p_hide_img2(p_hide_img2, scheduler):
    """Test that invalid p_hide_img2 raises a ValueError."""

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
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_diameter_range(diameter_range, scheduler):
    """Test that invalid diameter_range raises a ValueError."""
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
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_intensity_range(intensity_range, scheduler):
    """Test that invalid intensity_range raises a ValueError."""
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
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_rho_range(rho_range, scheduler):
    """Test that invalid rho_range raises a ValueError."""
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
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_dt(dt, scheduler):
    """Test that invalid dt raises a ValueError."""
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
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_synthetic_sampler_batches(
    batch_size, images_per_field, image_shape, scheduler
):
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
        assert batch[0].shape[0] >= batch_size
        assert batch[0][0].shape >= image_shape
        assert isinstance(batch[0], jnp.ndarray)

    assert len(all_batches) == images_per_field // batch_size


@pytest.mark.parametrize("batch_size, images_per_field", [(2, 4), (1, 3)])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_sampler_switches_flow_fields(batch_size, images_per_field, scheduler):
    sampler = SyntheticImageSampler(
        scheduler=scheduler,
        img_gen_fn=dummy_img_gen_fn,
        images_per_field=images_per_field,
        batch_size=batch_size,
        seed=0,
    )

    batch1 = next(sampler)
    batch2 = next(sampler)

    assert not jnp.allclose(batch1[0], batch2[0])


@pytest.mark.parametrize(
    "image_shape, num_images, num_particles", [((32, 32), 4, 100), ((64, 64), 4, 200)]
)
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("mock_hdf5_files", [1], indirect=True)
def test_sampler_with_real_img_gen_fn(
    image_shape, num_images, num_particles, batch_size, mock_hdf5_files
):
    files, _ = mock_hdf5_files
    scheduler = HDF5FlowFieldScheduler(files, loop=False)

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

    assert isinstance(batch[0], jnp.ndarray)
    assert batch[0].shape == (batch_size, *image_shape)
    assert batch[1].shape == (batch_size, *image_shape)


@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_speed_sampler_dummy_fn(scheduler):
    batch_size = 200
    images_per_field = 1000
    image_shape = (1216, 1936)
    position_bounds = (1536, 2048)

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

    run_sampler()
    total_time = timeit.repeat(
        stmt=run_sampler, number=NUMBER_OF_EXECUTIONS, repeat=REPETITIONS
    )
    avg_time = min(total_time) / NUMBER_OF_EXECUTIONS

    num_devices = len(jax.devices())
    limit_time = 0.5 if num_devices == 1 else 0.11 if num_devices == 2 else 0.07

    assert avg_time < limit_time


@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize("batch_size", [250])
@pytest.mark.parametrize("images_per_field", [1000])
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("num_particles", [40000])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_speed_sampler_real_fn(
    batch_size, images_per_field, seed, num_particles, scheduler
):
    image_shape = (1216, 1936)
    position_bounds = (1536, 2048)
    img_offset = (160, 56)

    # Check how many GPUs are available
    num_devices = len(jax.devices())

    # Limit time in seconds (depends on the number of GPUs)
    if num_devices == 1:
        limit_time = 1.5e-1
    elif num_devices == 2:
        limit_time = 8.5e-2
    elif num_devices == 4:
        limit_time = 6e-2

    # Create the sampler
    sampler = SyntheticImageSampler(
        scheduler=scheduler,
        img_gen_fn=generate_images_from_flow,
        images_per_field=images_per_field,
        batch_size=batch_size,
        seed=seed,
        image_shape=image_shape,
        position_bounds=position_bounds,
        img_offset=img_offset,
        num_particles=num_particles,
    )

    def run_sampler():
        # Generates images_per_field // batch_size batches
        # of size batch_size
        for i, batch in enumerate(sampler):
            logger.debug(scheduler._cached_data.shape)
            batch[0].block_until_ready()
            batch[1].block_until_ready()
            batch[2].block_until_ready()
            if i >= images_per_field // batch_size:
                break

    # Warm up the function
    run_sampler()

    # Measure the time taken to run the sampler
    total_time = timeit.repeat(
        stmt=run_sampler, number=NUMBER_OF_EXECUTIONS, repeat=REPETITIONS
    )
    avg_time = min(total_time) / NUMBER_OF_EXECUTIONS

    assert (
        avg_time < limit_time
    ), f"The average time is {avg_time}, time limit: {limit_time}"
