import re
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
    flow_field_res_x,
    flow_field_res_y,
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


@pytest.mark.parametrize("flow_field_size", [(-1, 128), (128, -1), (0, 128), (128, 0)])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_flow_field_size(flow_field_size, scheduler):
    """Test that invalid flow_field_size raises a ValueError."""
    with pytest.raises(
        ValueError, match="flow_field_size must be a tuple of two positive numbers."
    ):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=10,
            flow_field_size=flow_field_size,
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


@pytest.mark.parametrize("resolution", [-1, 0, "invalid_resolution"])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_resolution(resolution, scheduler):
    """Test that invalid resolution raises a ValueError."""
    with pytest.raises(ValueError, match="resolution must be a positive number."):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=10,
            resolution=resolution,
            seed=0,
        )


@pytest.mark.parametrize(
    "velocities_per_pixel", [-1, 0, "invalid_velocities_per_pixel"]
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_velocities_per_pixel(velocities_per_pixel, scheduler):
    """Test that invalid velocities_per_pixel raises a ValueError."""
    with pytest.raises(
        ValueError, match="velocities_per_pixel must be a positive number."
    ):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=10,
            velocities_per_pixel=velocities_per_pixel,
            seed=0,
        )


@pytest.mark.parametrize("img_offset", [(-1, 128), (128, -1)])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_img_offset(img_offset, scheduler):
    """Test that invalid img_offset raises a ValueError."""
    with pytest.raises(
        ValueError, match="img_offset must be a tuple of two non-negative numbers."
    ):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=10,
            img_offset=img_offset,
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
    "seed", ["invalid_seed", jnp.array([1]), jnp.array([1.0, 2.0])]
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_seed(seed, scheduler):
    """Test that invalid seed raises a ValueError."""
    with pytest.raises(ValueError, match="seed must be a positive integer."):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=10,
            seed=seed,
        )


@pytest.mark.parametrize("config_path", [None, 0])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_config_path(config_path, scheduler):
    """Test that invalid config_path raises a ValueError."""
    with pytest.raises(ValueError, match="config_path must be a string."):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=10,
            config_path=config_path,
            seed=0,
        )

    with pytest.raises(ValueError, match="config_path must be a .yaml file."):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=10,
            config_path="invalid_config.txt",
            seed=0,
        )

    with pytest.raises(ValueError, match="config_path does not exist."):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=10,
            config_path="non_existent_file.yaml",
            seed=0,
        )


@pytest.mark.parametrize("min_speed_x, max_speed_x", [(1, -1), (2, 1)])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_min_max_speed_x(min_speed_x, max_speed_x, scheduler):
    """Test that invalid min_speed_x and max_speed_x raises a ValueError."""
    with pytest.raises(
        ValueError, match="max_speed_x must be greater than min_speed_x."
    ):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=10,
            min_speed_x=min_speed_x,
            max_speed_x=max_speed_x,
            max_speed_y=0.0,
            min_speed_y=0.0,
            seed=0,
        )


@pytest.mark.parametrize("min_speed_y, max_speed_y", [(1, -1), (2, 1)])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_min_max_speed_y(min_speed_y, max_speed_y, scheduler):
    """Test that invalid min_speed_y and max_speed_y raises a ValueError."""
    with pytest.raises(
        ValueError, match="max_speed_y must be greater than min_speed_y."
    ):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=10,
            min_speed_y=min_speed_y,
            max_speed_y=max_speed_y,
            min_speed_x=0.0,
            max_speed_x=0.0,
            seed=0,
        )


@pytest.mark.parametrize("img_gen_fn", [None, "invalid_img_gen_fn"])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_img_gen_fn_type(img_gen_fn, scheduler):
    """Test that invalid img_gen_fn raises a ValueError."""
    with pytest.raises(ValueError, match="img_gen_fn must be a callable function."):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=img_gen_fn,
            batch_size=2,
            images_per_field=10,
            seed=0,
        )


@pytest.mark.parametrize(
    "img_offset, max_speed_x, max_speed_y, dt",
    [((0, 0), 1.0, 1.0, 1.0), ((0, 0), 0.0, 1.0, 1.0), ((0, 0), 1.0, 0.0, 1.0)],
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_img_offset_and_speed(
    img_offset, max_speed_x, max_speed_y, dt, scheduler
):
    """Test that invalid img_offset and speed raises a ValueError."""
    expected_message = re.escape(
        f"The image is too near the flow field left or top edge. "
        f"The minimum image offset is ({max_speed_y * dt}, {max_speed_x * dt})."
    )
    with pytest.raises(ValueError, match=expected_message):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=10,
            img_offset=img_offset,
            max_speed_x=max_speed_x,
            max_speed_y=max_speed_y,
            min_speed_x=0.0,
            min_speed_y=0.0,
            seed=0,
        )


@pytest.mark.parametrize(
    "flow_field_size, img_offset, image_shape,"
    " resolution, max_speed_x, max_speed_y, min_speed_x, min_speed_y, dt",
    [
        ((1, 4), (5, 10), (10, 5), 1, 1.0, 1.0, 1.0, -1.0, 0.1),
        ((2, 5), (5, 5), (5, 10), 1, 1.0, 1.0, 0.0, -1.0, 0.1),
        ((3, 6), (10, 5), (5, 5), 1, 1.0, 1.0, -1.0, -1.0, 0.1),
    ],
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_flow_field_size_and_img_offset(
    flow_field_size,
    img_offset,
    image_shape,
    resolution,
    max_speed_x,
    max_speed_y,
    min_speed_x,
    min_speed_y,
    dt,
    scheduler,
):
    """Test that invalid flow_field_size and img_offset raises a ValueError."""
    if max_speed_x < 0 or max_speed_y < 0:
        max_speed_x = 0.0
        max_speed_y = 0.0
    if min_speed_x > 0 or min_speed_y > 0:
        min_speed_x = 0.0
        min_speed_y = 0.0

    position_bounds = (
        image_shape[0] / resolution + max_speed_y * dt - min_speed_y * dt,
        image_shape[1] / resolution + max_speed_x * dt - min_speed_x * dt,
    )
    position_bounds_offset = (
        img_offset[0] - max_speed_y * dt,
        img_offset[1] - max_speed_x * dt,
    )
    expected_message = re.escape(
        f"The size of the flow field is too small."
        f"it must be at least "
        f"({position_bounds[0] + position_bounds_offset[0]},"
        f"{position_bounds[1] + position_bounds_offset[1]})."
    )

    logger.debug("test: " + str(position_bounds_offset))
    logger.debug(position_bounds)

    with pytest.raises(ValueError, match=expected_message):
        SyntheticImageSampler(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            batch_size=2,
            images_per_field=10,
            flow_field_size=flow_field_size,
            img_offset=img_offset,
            image_shape=image_shape,
            resolution=resolution,
            max_speed_x=max_speed_x,
            max_speed_y=max_speed_y,
            min_speed_x=min_speed_x,
            min_speed_y=min_speed_y,
            dt=dt,
            seed=0,
        )


@pytest.mark.parametrize(
    "batch_size, images_per_field, image_shape",
    [(4, 16, (256, 256)), (2, 8, (256, 256)), (1, 5, (256, 256))],
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
        num_particles=num_particles,
        images_per_field=num_images,
        batch_size=batch_size,
        seed=0,
    )

    batch = next(sampler)

    image_shape = sampler.image_shape
    res = sampler.resolution
    velocities_per_pixel = sampler.velocities_per_pixel
    output_size = jnp.array(
        [
            batch[2].shape[0] / velocities_per_pixel / res,
            batch[2].shape[1] / velocities_per_pixel / res,
            batch[2].shape[2],
        ]
    )

    expected_size = jnp.array([image_shape[0] / res, image_shape[1] / res, 2])

    assert isinstance(batch[0], jnp.ndarray)
    assert isinstance(batch[1], jnp.ndarray)
    assert isinstance(batch[2], jnp.ndarray)
    assert batch[0].shape == (batch_size, *image_shape)
    assert batch[1].shape == (batch_size, *image_shape)
    assert jnp.allclose(output_size, expected_size, atol=0.01)


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
def test_speed_sampler_dummy_fn(
    scheduler, batch_size, images_per_field, seed, num_particles
):
    """Test the speed of the sampler with a dummy image generation function."""
    # Define the parameters for the test
    image_shape = (1216, 1936)
    img_offset = (2.5e-2, 5e-2)
    flow_field_size = (3 * jnp.pi, 4 * jnp.pi)
    resolution = 155
    max_speed_x = 1.37
    max_speed_y = 0.56
    min_speed_x = -0.16
    min_speed_y = -0.72
    dt = 2.6e-2

    # Create the sampler
    sampler = SyntheticImageSampler(
        scheduler=scheduler,
        img_gen_fn=dummy_img_gen_fn,
        images_per_field=images_per_field,
        batch_size=batch_size,
        flow_field_size=flow_field_size,
        resolution=resolution,
        seed=seed,
        image_shape=image_shape,
        img_offset=img_offset,
        num_particles=num_particles,
        dt=dt,
        max_speed_x=max_speed_x,
        max_speed_y=max_speed_y,
        min_speed_x=min_speed_x,
        min_speed_y=min_speed_y,
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
    img_offset = (2.5e-2, 5e-2)
    flow_field_size = (3 * jnp.pi, 4 * jnp.pi)
    resolution = 155
    max_speed_x = 1.37
    max_speed_y = 0.56
    min_speed_x = -0.16
    min_speed_y = -0.72
    dt = 2.6e-2

    # Check how many GPUs are available
    num_devices = len(jax.devices())

    # Limit time in seconds (depends on the number of GPUs)
    if num_devices == 1:
        limit_time = 1.45e-1
    elif num_devices == 2:
        limit_time = 8e-2
    elif num_devices == 4:
        limit_time = 5.5e-2

    # Create the sampler
    sampler = SyntheticImageSampler(
        scheduler=scheduler,
        img_gen_fn=generate_images_from_flow,
        images_per_field=images_per_field,
        batch_size=batch_size,
        flow_field_size=flow_field_size,
        resolution=resolution,
        seed=seed,
        image_shape=image_shape,
        img_offset=img_offset,
        num_particles=num_particles,
        dt=dt,
        max_speed_x=max_speed_x,
        max_speed_y=max_speed_y,
        min_speed_x=min_speed_x,
        min_speed_y=min_speed_y,
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
