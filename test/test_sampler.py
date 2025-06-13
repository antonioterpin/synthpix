import re
import time
import timeit

import jax
import jax.numpy as jnp
import pytest

from synthpix.data_generate import generate_images_from_flow
from synthpix.sampler import SyntheticImageSampler
from synthpix.scheduler import (
    EpisodicFlowFieldScheduler,
    HDF5FlowFieldScheduler,
    MATFlowFieldScheduler,
    PrefetchingFlowFieldScheduler,
)
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
    seeding_density_range,
    p_hide_img1,
    p_hide_img2,
    diameter_range,
    diameter_var,
    intensity_range,
    intensity_var,
    rho_range,
    rho_var,
    dt,
    flow_field_res_x,
    flow_field_res_y,
    noise_level,
):
    """Simulates generating a batch of synthetic images based on a single key."""
    return (
        jnp.ones((num_images, image_shape[0], image_shape[1]))
        * (jnp.sum(flow_field) + jnp.sum(key)),
        jnp.ones((num_images, image_shape[0], image_shape[1]))
        * (jnp.sum(flow_field) + jnp.sum(key)),
        jnp.ones((num_images,)),
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
        "seeding_density_range",
        "p_hide_img1",
        "p_hide_img2",
        "diameter_range",
        "diameter_var",
        "intensity_range",
        "intensity_var",
        "rho_range",
        "rho_var",
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


@pytest.mark.parametrize("flow_fields_per_batch", [10, 20, 500])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_more_flows_per_batch_than_batch_size(flow_fields_per_batch, scheduler):
    """Test that flow_fields_per_batch is less than or equal to batch_size."""
    with pytest.raises(
        ValueError,
        match="flow_fields_per_batch must be less than or equal to batch_size.",
    ):
        config = sampler_config.copy()
        config["flow_fields_per_batch"] = flow_fields_per_batch
        config["batch_size"] = flow_fields_per_batch - 1
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
def test_invalid_velocities_per_pixel(velocities_per_pixel, scheduler):
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


@pytest.mark.parametrize(
    "seeding_density_range, expected_message",
    [
        (
            (-1.0, 1.0),
            "seeding_density_range must be a tuple of two non-negative numbers.",
        ),
        (
            (0.0, -1.0),
            "seeding_density_range must be a tuple of two non-negative numbers.",
        ),
        (
            (-0.5, -0.5),
            "seeding_density_range must be a tuple of two non-negative numbers.",
        ),
        ((1.0, 0.5), "seeding_density_range must be in the form \\(min, max\\)."),
        ((0.5, 0.1), "seeding_density_range must be in the form \\(min, max\\)."),
    ],
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_seeding_density_range(
    seeding_density_range, expected_message, scheduler
):
    """Test that invalid seeding_density_range raises a ValueError."""
    with pytest.raises(ValueError, match=expected_message):
        config = sampler_config.copy()
        config["seeding_density_range"] = seeding_density_range
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


@pytest.mark.parametrize(
    "diameter_range, expected_message",
    [
        ((-1.0, 1.0), "diameter_range must be a tuple of two positive floats."),
        ((0.0, -1.0), "diameter_range must be a tuple of two positive floats."),
        ((-0.5, -0.5), "diameter_range must be a tuple of two positive floats."),
        ((1.0, 0.5), "diameter_range must be in the form \\(min, max\\)."),
        ((0.5, 0.1), "diameter_range must be in the form \\(min, max\\)."),
    ],
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_diameter_range(diameter_range, expected_message, scheduler):
    """Test that invalid diameter_range raises a ValueError."""
    with pytest.raises(ValueError, match=expected_message):
        config = sampler_config.copy()
        config["diameter_range"] = diameter_range
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
        )


@pytest.mark.parametrize(
    "diameter_var",
    [-1, "invalid_diameter_var", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])],
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_diameter_var(diameter_var, scheduler):
    """Test that invalid diameter_var raises a ValueError."""
    with pytest.raises(ValueError, match="diameter_var must be a non-negative number."):
        config = sampler_config.copy()
        config["diameter_var"] = diameter_var
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
        )


@pytest.mark.parametrize(
    "intensity_range, expected_message",
    [
        ((-1.0, 1.0), "intensity_range must be a tuple of two non-negative floats."),
        ((0.0, -1.0), "intensity_range must be a tuple of two non-negative floats."),
        ((-0.5, -0.5), "intensity_range must be a tuple of two non-negative floats."),
        ((1.0, 0.5), "intensity_range must be in the form \\(min, max\\)."),
        ((0.5, 0.1), "intensity_range must be in the form \\(min, max\\)."),
    ],
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_intensity_range(intensity_range, expected_message, scheduler):
    """Test that invalid intensity_range raises a ValueError."""
    with pytest.raises(ValueError, match=expected_message):
        config = sampler_config.copy()
        config["intensity_range"] = intensity_range
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
        )


@pytest.mark.parametrize(
    "intensity_var",
    [-1, "invalid_intensity_var", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])],
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_intensity_var(intensity_var, scheduler):
    """Test that invalid intensity_var raises a ValueError."""
    with pytest.raises(
        ValueError, match="intensity_var must be a non-negative number."
    ):
        config = sampler_config.copy()
        config["intensity_var"] = intensity_var
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
        )


@pytest.mark.parametrize(
    "rho_range, expected_message",
    [
        ((-1.1, 1.0), "rho_range must be a tuple of two floats between -1 and 1."),
        ((0.0, 1.1), "rho_range must be a tuple of two floats between -1 and 1."),
        ((0.9, 0.5), "rho_range must be in the form \\(min, max\\)."),
        ((0.5, 0.1), "rho_range must be in the form \\(min, max\\)."),
    ],
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_rho_range(rho_range, expected_message, scheduler):
    """Test that invalid rho_range raises a ValueError."""
    with pytest.raises(ValueError, match=expected_message):
        config = sampler_config.copy()
        config["rho_range"] = rho_range
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
        )


@pytest.mark.parametrize(
    "rho_var", [-1, "invalid_rho_var", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])]
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_rho_var(rho_var, scheduler):
    """Test that invalid rho_var raises a ValueError."""
    with pytest.raises(ValueError, match="rho_var must be a non-negative number."):
        config = sampler_config.copy()
        config["rho_var"] = rho_var
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
    "noise_level", [-1, "a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])]
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_noise_level(noise_level, scheduler):
    """Test that invalid noise_level raises a ValueError."""
    with pytest.raises(ValueError, match="noise_level must be a non-negative number."):
        config = sampler_config.copy()
        config["noise_level"] = noise_level
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
        f" It must be at least "
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
    [(12, 16, (256, 256)), (12, 4, (256, 256))],
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

    for batch in sampler:
        assert batch[0].shape[0] >= batch_size
        assert batch[0][0].shape >= image_shape
        assert isinstance(batch[0], jnp.ndarray)


@pytest.mark.parametrize(
    "batch_size, batches_per_flow_batch, flow_fields_per_batch",
    [(24, 4, 12), (12, 6, 12)],
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": True}], indirect=True
)
def test_sampler_switches_flow_fields(
    batch_size, batches_per_flow_batch, flow_fields_per_batch, scheduler
):
    config = sampler_config.copy()
    config["batch_size"] = batch_size
    config["batches_per_flow_batch"] = batches_per_flow_batch
    config["flow_fields_per_batch"] = flow_fields_per_batch
    sampler = SyntheticImageSampler.from_config(
        scheduler=scheduler,
        img_gen_fn=dummy_img_gen_fn,
        config=config,
    )

    for i, batch in enumerate(sampler):
        if i >= batches_per_flow_batch - 1:
            batch1 = batch
            break

    batch2 = next(sampler)

    assert not jnp.allclose(batch1[0], batch2[0])
    assert not jnp.allclose(batch1[1], batch2[1])
    assert not jnp.allclose(batch1[2], batch2[2])


@pytest.mark.parametrize(
    "image_shape, batches_per_flow_batch, seeding_density_range",
    [((32, 32), 4, (0.1, 0.1)), ((64, 64), 4, (0.0, 0.04))],
)
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("mock_hdf5_files", [1], indirect=True)
def test_sampler_with_real_img_gen_fn(
    image_shape,
    batches_per_flow_batch,
    seeding_density_range,
    batch_size,
    mock_hdf5_files,
):
    files, _ = mock_hdf5_files
    scheduler = HDF5FlowFieldScheduler(files, loop=False)

    config = sampler_config.copy()
    config["batch_size"] = batch_size
    config["flow_fields_per_batch"] = batch_size
    config["batches_per_flow_batch"] = batches_per_flow_batch
    config["image_shape"] = image_shape
    config["seeding_density_range"] = seeding_density_range
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
    assert batch[0].shape == (sampler.batch_size, *image_shape)
    assert batch[1].shape == (sampler.batch_size, *image_shape)
    assert jnp.allclose(output_size, expected_size, atol=0.01)


@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("batches_per_flow_batch", [100])
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("seeding_density_range", [(0.0, 0.03)])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": True}], indirect=True
)
def test_speed_sampler_dummy_fn(
    scheduler, batch_size, batches_per_flow_batch, seed, seeding_density_range
):
    """Test the speed of the sampler with a dummy image generation function."""
    # Define the parameters for the test
    config = sampler_config.copy()
    config["batch_size"] = batch_size
    config["flow_fields_per_batch"] = 64
    config["image_shape"] = (1216, 1936)
    config["seeding_density_range"] = seeding_density_range
    config["img_offset"] = (2.5e-2, 5e-2)
    config["flow_field_size"] = (3 * jnp.pi, 4 * jnp.pi)
    config["resolution"] = 155
    config["max_speed_x"] = 1.37
    config["max_speed_y"] = 0.56
    config["min_speed_x"] = -0.16
    config["min_speed_y"] = -0.72
    config["dt"] = 2.6e-2
    config["noise_level"] = 0.0
    config["batch_size"] = batch_size
    config["seed"] = seed

    if config["flow_fields_per_batch"] % len(jax.devices()) != 0:
        pytest.skip("flow_fields_per_batch must be divisible by the number of devices.")

    # Check how many GPUs are available
    num_devices = len(jax.devices())
    # Limit time in seconds (depends on the number of GPUs)
    # The test should not depend much on the number of GPUs.
    if num_devices == 1:
        limit_time = 3.0
    elif num_devices == 2:
        limit_time = 2.6
    elif num_devices == 4:
        limit_time = 2.6

    # Create the sampler
    prefetching_scheduler = PrefetchingFlowFieldScheduler(
        scheduler=scheduler,
        batch_size=config["flow_fields_per_batch"],
        buffer_size=4,
    )
    sampler = SyntheticImageSampler.from_config(
        scheduler=prefetching_scheduler,
        img_gen_fn=dummy_img_gen_fn,
        config=config,
    )

    def run_sampler():
        # Generates images_per_field // batch_size batches
        # of size batch_size
        for i, batch in enumerate(sampler):
            batch[0].block_until_ready()
            batch[1].block_until_ready()
            batch[2].block_until_ready()
            logger.info(i)
            if i >= batches_per_flow_batch:
                sampler.reset(scheduler_reset=False)
                break

    try:
        # Warm up the function
        run_sampler()

        # Measure the time taken to run the sampler
        total_time = timeit.repeat(
            stmt=run_sampler, number=NUMBER_OF_EXECUTIONS, repeat=REPETITIONS
        )
        avg_time = min(total_time) / NUMBER_OF_EXECUTIONS
    finally:
        prefetching_scheduler.shutdown()

    assert (
        avg_time < limit_time
    ), f"The average time is {avg_time}, time limit: {limit_time}"


@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("batches_per_flow_batch", [100])
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("seeding_density_range", [(0.001, 0.004)])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": True}], indirect=True
)
def test_speed_sampler_real_fn(
    batch_size, batches_per_flow_batch, seed, seeding_density_range, scheduler
):
    config = sampler_config.copy()
    config["batch_size"] = batch_size
    config["batches_per_flow_batch"] = batches_per_flow_batch
    config["seeding_density_range"] = seeding_density_range
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
    config["noise_level"] = 0.0
    config["flow_fields_per_batch"] = 64

    # Check how many GPUs are available

    if config["flow_fields_per_batch"] % len(jax.devices()) != 0:
        pytest.skip("flow_fields_per_batch must be divisible by the number of devices.")
    num_devices = len(jax.devices())

    # Limit time in seconds (depends on the number of GPUs)
    if num_devices == 1:
        limit_time = 1.7
    elif num_devices == 2:
        limit_time = 1.2
    elif num_devices == 4:
        limit_time = 1.14

    # Create the sampler
    prefetching_scheduler = PrefetchingFlowFieldScheduler(
        scheduler=scheduler,
        batch_size=config["flow_fields_per_batch"],
        buffer_size=4,
    )
    sampler = SyntheticImageSampler.from_config(
        scheduler=prefetching_scheduler,
        img_gen_fn=generate_images_from_flow,
        config=config,
    )

    def run_sampler():
        # Generates batches_per_flow_batch batches
        # of size batch_size
        for i, batch in enumerate(sampler):
            batch[0].block_until_ready()
            batch[1].block_until_ready()
            batch[2].block_until_ready()
            batch[3].block_until_ready()
            if i >= batches_per_flow_batch:
                sampler.reset(scheduler_reset=False)
                break

    try:
        # Warm up the function
        run_sampler()

        # Measure the time taken to run the sampler
        total_time = timeit.repeat(
            stmt=run_sampler, number=NUMBER_OF_EXECUTIONS, repeat=REPETITIONS
        )
        avg_time = min(total_time) / NUMBER_OF_EXECUTIONS
    finally:
        prefetching_scheduler.shutdown()

    assert (
        avg_time < limit_time
    ), f"The average time is {avg_time}, time limit: {limit_time}"


# -----------------------------------------------------------------------------
# Dummy image generator – ultra‑lightweight
# -----------------------------------------------------------------------------


def _dummy_img_gen_fn(*, key, flow_field, num_images, image_shape, **_):
    """Return zero‑filled image pairs and constant densities.

    The sampler tests are *control‑flow* tests – we don’t need heavyweight
    rendering; we just need correctly‑shaped outputs so the JIT/shard_map path
    executes.
    """
    h, w = image_shape
    imgs1 = jnp.zeros((num_images, h, w), dtype=jnp.float32)
    imgs2 = jnp.zeros_like(imgs1)
    rho = jnp.full((num_images,), 0.1, dtype=jnp.float32)
    return imgs1, imgs2, rho


# -----------------------------------------------------------------------------
# Global parameters – tweak once here if you change defaults in the code base
# -----------------------------------------------------------------------------

BATCH_SIZE = 4
EPISODE_LENGTH = 4
FLOW_BATCH_SIZE = BATCH_SIZE  # one flow‑field per episode step
BATCHES_PER_FLOW_BATCH = 1  # keep simple: one synthetic batch per step
BUFFER_SIZE = 3 * EPISODE_LENGTH  # queue can hold an entire episode
FLOW_FIELD_SIZE = (64, 64)  # arbitrary physical dimensions (metres)
DT = 1.0
IMG_SHAPE = (64, 64)  # small so CPU JAX is quick
NUM_EPISODES = 6


# -----------------------------------------------------------------------------
# Fixture: full stack (MAT → Episodic → Prefetch → SyntheticImageSampler)
# -----------------------------------------------------------------------------


@pytest.fixture(name="sampler")
def _build_sampler(mock_mat_files):
    """Build a *fully‑stacked* sampler and ensure the background thread is closed."""

    files, dims = mock_mat_files
    H, W = dims["height"], dims["width"]

    print(files)

    base = MATFlowFieldScheduler(files, loop=False, output_shape=(H, W))
    epi = EpisodicFlowFieldScheduler(
        scheduler=base,
        batch_size=BATCH_SIZE,
        episode_length=EPISODE_LENGTH,
        seed=123,
    )
    pre = PrefetchingFlowFieldScheduler(
        epi, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE
    )

    sampler = SyntheticImageSampler(
        scheduler=pre,
        img_gen_fn=_dummy_img_gen_fn,
        batches_per_flow_batch=BATCHES_PER_FLOW_BATCH,
        batch_size=BATCH_SIZE,
        flow_fields_per_batch=FLOW_BATCH_SIZE,
        flow_field_size=FLOW_FIELD_SIZE,
        image_shape=IMG_SHAPE,
        resolution=1.0,
        velocities_per_pixel=1.0,
        img_offset=(0.0, 0.0),
        seeding_density_range=(0.01, 0.01),
        p_hide_img1=0.0,
        p_hide_img2=0.0,
        diameter_range=(1.0, 1.0),
        diameter_var=0.0,
        intensity_range=(1.0, 1.0),
        intensity_var=0.0,
        rho_range=(0.0, 0.0),
        rho_var=0.0,
        dt=DT,
        seed=0,
        max_speed_x=0.0,
        max_speed_y=0.0,
        min_speed_x=0.0,
        min_speed_y=0.0,
        output_units="pixels",
        noise_level=0.0,
    )
    yield sampler

    # ---------------------------------------------------------------------
    # Teardown – make sure the background thread is gone so the test runner
    # exits cleanly even under pytest‑xdist or when only collecting tests.
    # ---------------------------------------------------------------------
    sampler.scheduler.shutdown()


@pytest.mark.parametrize("mock_mat_files", [128], indirect=True)
def test_done_flag_and_horizon(sampler):
    """`done` should be True *exactly* once (the final step of each episode)."""

    dones = []
    for i in range(NUM_EPISODES):
        imgs1, _, _, _, done = sampler.next_episode()
        dones.append(done)
        for j in range(EPISODE_LENGTH - 1):
            imgs1, _, _, _, done = next(sampler)
            assert imgs1.shape[0] == BATCH_SIZE
            assert imgs1[0].shape == IMG_SHAPE
            assert isinstance(imgs1, jnp.ndarray)
            dones.append(done)

    true_flags = sum(int(flag) for d in dones for flag in d)
    assert (
        true_flags == NUM_EPISODES * BATCH_SIZE
    ), "`done` must become True once per episode"

    last_step_dones = dones[EPISODE_LENGTH - 1 :: EPISODE_LENGTH]

    # For each episode's last batch, require *all* BATCH_SIZE flags to be True
    assert all(
        bool(jnp.all(d)) for d in last_step_dones
    ), "`done` must be True for every env on the *last* step of each episode"


@pytest.mark.parametrize("mock_mat_files", [64], indirect=True)
def test_next_episode_flushes_queue(sampler):
    """Calling `next_episode()` should discard buffered batches and restart."""

    # Consume half an episode …
    for _ in range(EPISODE_LENGTH // 2):
        next(sampler)

    # … then jump straight to the next one.
    first_batch_new_ep = sampler.next_episode()

    # Give the producer thread a moment to refill.
    time.sleep(0.1)

    # steps_remaining() should be reset to horizon‑1.
    remaining = sampler.scheduler.steps_remaining()
    assert remaining == EPISODE_LENGTH - 1

    # The very first `done` after reset must be False.
    assert not first_batch_new_ep[-1].any()


@pytest.mark.parametrize("mock_mat_files", [64], indirect=True)
def test_stop_after_max_episodes(mock_mat_files):
    """Sampler raises StopIteration after the configured `num_episodes`."""

    files, dims = mock_mat_files
    H, W = dims["height"], dims["width"]

    num_episodes = 2
    base = MATFlowFieldScheduler(files, loop=True, output_shape=(H, W))
    epi = EpisodicFlowFieldScheduler(base, batch_size=4, episode_length=2, seed=0)
    pre = PrefetchingFlowFieldScheduler(epi, batch_size=4, buffer_size=90)

    sampler = SyntheticImageSampler(
        scheduler=pre,
        img_gen_fn=dummy_img_gen_fn,
        batches_per_flow_batch=1,
        batch_size=4,
        flow_fields_per_batch=4,
        flow_field_size=(H, W),
        image_shape=(H, W),
        resolution=1.0,
        velocities_per_pixel=1.0,
        img_offset=(0.0, 0.0),
        seeding_density_range=(0.01, 0.01),
        p_hide_img1=0.0,
        p_hide_img2=0.0,
        diameter_range=(1.0, 1.0),
        diameter_var=0.0,
        intensity_range=(1.0, 1.0),
        intensity_var=0.0,
        rho_range=(0.0, 0.0),
        rho_var=0.0,
        dt=1.0,
        seed=0,
        max_speed_x=0.0,
        max_speed_y=0.0,
        min_speed_x=0.0,
        min_speed_y=0.0,
        output_units="pixels",
        noise_level=0.0,
    )

    # We expect exactly num_episodes × episode_length iterations.
    n_batches = 0

    for i in range(num_episodes):
        imgs1, imgs2, flows, _, done = sampler.next_episode()
        print(f"episode {i} batch {n_batches}")
        n_batches += 1
        while not any(done):
            logger.debug(f"episode {i} batch {n_batches}")
            imgs1, imgs2, flows, _, done = next(sampler)
            assert imgs1.shape[0] == 4
            assert imgs1[0].shape == (H, W)
            assert isinstance(imgs1, jnp.ndarray)
            print(f"episode {i} batch {n_batches}")
            n_batches += 1

    assert (
        n_batches == epi.episode_length * num_episodes
    ), f"Expected {epi.episode_length * num_episodes} batches, but got {n_batches}"

    # Clean up background thread
    sampler.scheduler.shutdown()
