import re
import timeit

import jax
import jax.numpy as jnp
import pytest

from synthpix.data_generate import generate_images_from_flow
from synthpix.image_sampler import SyntheticImageSampler
from synthpix.scheduler import HDF5FlowFieldScheduler
from synthpix.utils import load_configuration, logger

config = load_configuration("config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_SAMPLER"]

sampler_config = load_configuration("config/test_data.yaml")


def dummy_img_gen_fn(
    key,
    flow_field,
    position_bounds,
    image_shape,
    img_offset,
    num_images,
    seeding_density,
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
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=sampler_config,
        )


@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
@pytest.mark.parametrize(
    "missing_key",
    [
        "batch_size",
        "batches_per_flow_batch",
        "flow_fields_per_batch",
        "image_shape",
        "flow_field_size",
        "resolution",
        "max_speed_x",
        "max_speed_y",
        "min_speed_x",
        "min_speed_y",
        "dt",
        "img_offset",
        "seeding_density",
        "p_hide_img1",
        "p_hide_img2",
        "diameter_range",
        "intensity_range",
        "rho_range",
        "velocities_per_pixel",
    ],
)
def test_from_config_missing_key_raises(scheduler, missing_key):
    config = sampler_config.copy()
    config.pop(missing_key)

    with pytest.raises(KeyError):
        SyntheticImageSampler.from_config(
            scheduler=scheduler, img_gen_fn=dummy_img_gen_fn, config=config
        )


@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_img_gen_fn(scheduler):
    """Test that invalid img_gen_fn raises a ValueError."""
    with pytest.raises(ValueError, match="img_gen_fn must be a callable function."):
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=None,
            config=sampler_config,
        )


@pytest.mark.parametrize("batch_size", [-1, 0, 1.5])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_batch_size(batch_size, scheduler):
    """Test that invalid batch_size raises a ValueError."""
    with pytest.raises(ValueError, match="batch_size must be a positive integer."):
        config = sampler_config.copy()
        config["batch_size"] = batch_size
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
        )


@pytest.mark.parametrize("batches_per_flow_batch", [-1, 0, 1.5])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_batches_per_flow_batch(batches_per_flow_batch, scheduler):
    """Test that invalid batches_per_flow_batch raises a ValueError."""
    with pytest.raises(
        ValueError, match="batches_per_flow_batch must be a positive integer."
    ):
        config = sampler_config.copy()
        config["batches_per_flow_batch"] = batches_per_flow_batch
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
        )


@pytest.mark.parametrize("flow_fields_per_batch", [-1, 0, 1.5])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_flow_fields_per_batch(flow_fields_per_batch, scheduler):
    """Test that invalid flow_fields_per_batch raises a ValueError."""
    with pytest.raises(
        ValueError, match="flow_fields_per_batch must be a positive integer."
    ):
        config = sampler_config.copy()
        config["flow_fields_per_batch"] = flow_fields_per_batch
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
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
        config = sampler_config.copy()
        config["flow_field_size"] = flow_field_size
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
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
        config = sampler_config.copy()
        config["image_shape"] = image_shape
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
        )


@pytest.mark.parametrize("resolution", [-1, 0, "invalid_resolution"])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_resolution(resolution, scheduler):
    """Test that invalid resolution raises a ValueError."""
    with pytest.raises(ValueError, match="resolution must be a positive number."):
        config = sampler_config.copy()
        config["resolution"] = resolution
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
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
        config = sampler_config.copy()
        config["velocities_per_pixel"] = velocities_per_pixel
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
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
        config = sampler_config.copy()
        config["img_offset"] = img_offset
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
        )


@pytest.mark.parametrize("seeding_density", [-1, 0, 1.5])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_seeding_density(seeding_density, scheduler):
    """Test that invalid seeding_density raises a ValueError."""
    with pytest.raises(
        ValueError, match="seeding_density must be a float between 0 and 1."
    ):
        config = sampler_config.copy()
        config["seeding_density"] = seeding_density
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
        )


@pytest.mark.parametrize("p_hide_img1", [-0.1, 1.1])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_p_hide_img1(p_hide_img1, scheduler):
    """Test that invalid p_hide_img1 raises a ValueError."""
    with pytest.raises(ValueError, match="p_hide_img1 must be between 0 and 1."):
        config = sampler_config.copy()
        config["p_hide_img1"] = p_hide_img1
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
        )


@pytest.mark.parametrize("p_hide_img2", [-0.1, 1.1])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_p_hide_img2(p_hide_img2, scheduler):
    """Test that invalid p_hide_img2 raises a ValueError."""

    with pytest.raises(ValueError, match="p_hide_img2 must be between 0 and 1."):
        config = sampler_config.copy()
        config["p_hide_img2"] = p_hide_img2
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
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
        config = sampler_config.copy()
        config["diameter_range"] = diameter_range
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
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
        config = sampler_config.copy()
        config["intensity_range"] = intensity_range
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
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
        config = sampler_config.copy()
        config["rho_range"] = rho_range
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
        )


@pytest.mark.parametrize("dt", ["invalid_dt", jnp.array([1]), jnp.array([1.0, 2.0])])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_dt(dt, scheduler):
    """Test that invalid dt raises a ValueError."""
    with pytest.raises(ValueError, match="dt must be a scalar \\(int or float\\)"):
        config = sampler_config.copy()
        config["dt"] = dt
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
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
        config = sampler_config.copy()
        config["seed"] = seed
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
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
        config = sampler_config.copy()
        config["min_speed_x"] = min_speed_x
        config["max_speed_x"] = max_speed_x
        config["min_speed_y"] = 0.0
        config["max_speed_y"] = 0.0
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
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
        config = sampler_config.copy()
        config["min_speed_x"] = 0.0
        config["max_speed_x"] = 0.0
        config["min_speed_y"] = min_speed_y
        config["max_speed_y"] = max_speed_y
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
        )


@pytest.mark.parametrize("img_gen_fn", [None, "invalid_img_gen_fn"])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_img_gen_fn_type(img_gen_fn, scheduler):
    """Test that invalid img_gen_fn raises a ValueError."""
    with pytest.raises(ValueError, match="img_gen_fn must be a callable function."):
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=img_gen_fn,
            config=sampler_config,
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
        f"The image is too close the flow field left or top edge. "
        f"The minimum image offset is ({max_speed_y * dt}, {max_speed_x * dt})."
    )
    with pytest.raises(ValueError, match=expected_message):
        config = sampler_config.copy()
        config["img_offset"] = img_offset
        config["max_speed_x"] = max_speed_x
        config["max_speed_y"] = max_speed_y
        config["min_speed_x"] = 0.0
        config["min_speed_y"] = 0.0
        config["dt"] = dt
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
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

    config = sampler_config.copy()
    config["flow_field_size"] = flow_field_size
    config["img_offset"] = img_offset
    config["image_shape"] = image_shape
    config["resolution"] = resolution
    config["max_speed_x"] = max_speed_x
    config["max_speed_y"] = max_speed_y
    config["min_speed_x"] = min_speed_x
    config["min_speed_y"] = min_speed_y
    config["dt"] = dt

    with pytest.raises(ValueError, match=expected_message):
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
        )


@pytest.mark.parametrize("output_units", [None, 123, "invalid_output_units"])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_output_units(output_units, scheduler):
    """Test that invalid output units raises a ValueError."""
    expected_message = "output_units must be 'pixels' or 'measure units per second'."
    with pytest.raises(ValueError, match=expected_message):
        config = sampler_config.copy()
        config["output_units"] = output_units
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
        )


@pytest.mark.parametrize(
    "batch_size, batches_per_flow_batch, image_shape",
    [(4, 16, (256, 256)), (2, 8, (256, 256)), (1, 5, (256, 256))],
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_synthetic_sampler_batches(
    batch_size, batches_per_flow_batch, image_shape, scheduler
):
    config = sampler_config.copy()
    config["batch_size"] = batch_size
    config["batches_per_flow_batch"] = batches_per_flow_batch
    config["image_shape"] = image_shape
    sampler = SyntheticImageSampler.from_config(
        scheduler=scheduler,
        img_gen_fn=dummy_img_gen_fn,
        config=config,
    )

    iterator = iter(sampler)
    all_batches = []
    for _ in range(batches_per_flow_batch):
        batch = next(iterator)
        all_batches.append(batch)
        assert batch[0].shape[0] >= batch_size
        assert batch[0][0].shape >= image_shape
        assert isinstance(batch[0], jnp.ndarray)

    assert len(all_batches) == batches_per_flow_batch


@pytest.mark.parametrize("batch_size, batches_per_flow_batch", [(2, 4), (1, 3)])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_sampler_switches_flow_fields(batch_size, batches_per_flow_batch, scheduler):
    config = sampler_config.copy()
    config["batch_size"] = batch_size
    config["images_per_field"] = batches_per_flow_batch
    sampler = SyntheticImageSampler.from_config(
        scheduler=scheduler,
        img_gen_fn=dummy_img_gen_fn,
        config=config,
    )

    batch1 = next(sampler)
    for _ in range(batches_per_flow_batch - 1):
        next(sampler)
    batch2 = next(sampler)

    assert not jnp.allclose(batch1[0], batch2[0])


@pytest.mark.parametrize(
    "image_shape, batches_per_flow_batch, seeding_density",
    [((32, 32), 4, 0.1), ((64, 64), 4, 0.04)],
)
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("mock_hdf5_files", [1], indirect=True)
def test_sampler_with_real_img_gen_fn(
    image_shape, batches_per_flow_batch, seeding_density, batch_size, mock_hdf5_files
):
    files, _ = mock_hdf5_files
    scheduler = HDF5FlowFieldScheduler(files, loop=False)

    config = sampler_config.copy()
    config["batch_size"] = batch_size
    config["batches_per_flow_batch"] = batches_per_flow_batch
    config["image_shape"] = image_shape
    config["seeding_density"] = seeding_density
    sampler = SyntheticImageSampler.from_config(
        scheduler=scheduler,
        img_gen_fn=generate_images_from_flow,
        config=config,
    )

    batch = next(sampler)

    image_shape = sampler.image_shape
    res = sampler.resolution
    velocities_per_pixel = sampler.velocities_per_pixel
    output_size = jnp.array(
        [
            batch[2].shape[1] / velocities_per_pixel / res,
            batch[2].shape[2] / velocities_per_pixel / res,
            batch[2].shape[3],
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
@pytest.mark.parametrize("seeding_density", [0.03])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_speed_sampler_dummy_fn(
    scheduler, batch_size, images_per_field, seed, seeding_density
):
    """Test the speed of the sampler with a dummy image generation function."""
    # Define the parameters for the test
    config = sampler_config.copy()
    config["batch_size"] = batch_size
    config["images_per_field"] = images_per_field
    config["image_shape"] = (1216, 1936)
    config["seeding_density"] = seeding_density
    config["img_offset"] = (2.5e-2, 5e-2)
    config["flow_field_size"] = (3 * jnp.pi, 4 * jnp.pi)
    config["resolution"] = 155
    config["max_speed_x"] = 1.37
    config["max_speed_y"] = 0.56
    config["min_speed_x"] = -0.16
    config["min_speed_y"] = -0.72
    config["dt"] = 2.6e-2
    config["batch_size"] = batch_size
    config["images_per_field"] = images_per_field
    config["seeding_density"] = seeding_density
    config["seed"] = seed

    # Create the sampler
    sampler = SyntheticImageSampler.from_config(
        scheduler=scheduler,
        img_gen_fn=dummy_img_gen_fn,
        config=config,
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
@pytest.mark.parametrize("batch_size", [150])
@pytest.mark.parametrize("batches_per_flow_batch", [333])
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("seeding_density", [0.016])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": True}], indirect=True
)
def test_speed_sampler_real_fn(
    batch_size, batches_per_flow_batch, seed, seeding_density, scheduler
):
    config = sampler_config.copy()
    config["batch_size"] = batch_size
    config["batches_per_flow_batch"] = batches_per_flow_batch
    config["seeding_density"] = seeding_density
    config["seed"] = seed
    config["image_shape"] = (1216, 1936)
    config["img_offset"] = (2.5e-2, 5e-2)
    config["flow_field_size"] = (3 * jnp.pi, 4 * jnp.pi)
    config["resolution"] = 155
    config["max_speed_x"] = 1.37
    config["max_speed_y"] = 0.56
    config["min_speed_x"] = -0.16
    config["min_speed_y"] = -0.72
    config["dt"] = 2.6e-2
    config["flow_fields_per_batch"] = 50
    # Check how many GPUs are available
    num_devices = len(jax.devices())
    # Limit time in seconds (depends on the number of GPUs)
    if num_devices == 1:
        limit_time = 7.0
    elif num_devices == 2:
        limit_time = 4.5
    elif num_devices == 4:
        limit_time = 3.0  # TODO: fix times for 4 GPUs when available
    # Create the sampler
    sampler = SyntheticImageSampler.from_config(
        scheduler=scheduler,
        img_gen_fn=generate_images_from_flow,
        config=config,
    )

    def run_sampler():
        # Generates images_per_field // batch_size batches
        # of size batch_size
        for i, batch in enumerate(sampler):
            logger.debug(f"Cached_data shape: {scheduler._cached_data.shape}")
            # batch[0].block_until_ready()
            # batch[1].block_until_ready()
            # batch[2].block_until_ready()
            if i >= batches_per_flow_batch:
                logger.debug(f"Finished iteration {i}")
                sampler.reset()
                batch[0].block_until_ready()
                batch[1].block_until_ready()
                batch[2].block_until_ready()
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
