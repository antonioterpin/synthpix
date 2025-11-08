import os
import re
import timeit

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from synthpix.sampler import RealImageSampler, SyntheticImageSampler
from synthpix.scheduler import (
    EpisodicFlowFieldScheduler,
    HDF5FlowFieldScheduler,
    MATFlowFieldScheduler,
    PrefetchingFlowFieldScheduler,
)
from synthpix.scheduler.base import BaseFlowFieldScheduler
from synthpix.scheduler.protocol import EpisodeEnd, EpisodicSchedulerProtocol
from synthpix.types import ImageGenerationSpecification, SchedulerData
from synthpix.utils import load_configuration

config = load_configuration("config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_SAMPLER"]

sampler_config = load_configuration("config/test_data.yaml")


@pytest.mark.parametrize("scheduler", [None, "invalid_scheduler"])
def test_invalid_scheduler(scheduler):
    """Test that invalid scheduler raises a ValueError."""
    with pytest.raises(
        TypeError, match="scheduler must implement the SchedulerProtocol interface."
    ):
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
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
        "diameter_ranges",
        "diameter_var",
        "intensity_ranges",
        "intensity_var",
        "rho_ranges",
        "rho_var",
        "velocities_per_pixel",
    ],
)
def test_from_config_missing_key_raises(scheduler, missing_key):
    config = sampler_config.copy()
    config.pop(missing_key)

    with pytest.raises(KeyError):
        SyntheticImageSampler.from_config(scheduler=scheduler, config=config)


@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_no_device_provided(scheduler):
    config = sampler_config.copy()
    config["device_ids"] = []
    with pytest.raises(ValueError, match="No valid device IDs provided."):
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            config=config,
        )


@pytest.mark.parametrize("n_devices", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_batch_size_n_devices(n_devices, scheduler):
    config = sampler_config.copy()
    all_devices = jax.devices()
    if n_devices > len(all_devices):
        pytest.skip(
            f"Only {len(all_devices)} devices available, "
            f"cannot test {n_devices} devices."
        )

    config["device_ids"] = [d.id for d in all_devices[:n_devices]]
    config["batch_size"] = 1
    config["flow_fields_per_batch"] = 1
    config["batches_per_flow_batch"] = 1
    sampler = SyntheticImageSampler.from_config(
        scheduler=scheduler,
        config=config,
    )
    assert (
        sampler.batch_size == n_devices
    ), "Batch size should match number of devices when set to 1."


@pytest.mark.parametrize("batch_size", [-1, 0, 1.5])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_batch_size(batch_size, scheduler):
    """Test that invalid batch_size raises a ValueError."""
    with pytest.raises(ValueError, match="batch_size must be a positive integer."):
        config = sampler_config.copy()
        config["batch_size"] = batch_size
        config["flow_fields_per_batch"] = batch_size
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            config=config,
        )


@pytest.mark.parametrize("flow_shape", [(-1, 128), (128, -1), (0, 128), (128, 0)])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_flow_shape(flow_shape, scheduler):
    """Test that invalid flow_shape raises a ValueError."""
    with pytest.raises(
        ValueError, match="flow_field_size must be a tuple of two positive numbers."
    ):
        config = sampler_config.copy()
        config["flow_field_size"] = flow_shape
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
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
            config=config,
        )


@pytest.mark.parametrize("flow_fields_per_batch", [10, 20, 500])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_more_flows_per_batch_than_batch_size(flow_fields_per_batch, scheduler):
    """Test that flow_fields_per_batch is <= batch_size."""
    with pytest.raises(
        ValueError,
        match="flow_fields_per_batch must be <= batch_size.",
    ):
        config = sampler_config.copy()
        config["flow_fields_per_batch"] = flow_fields_per_batch
        config["batch_size"] = flow_fields_per_batch - 1
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            config=config,
        )


@pytest.mark.parametrize(
    "flow_field_size",
    [
        (-1, 128),
        (128, -1),
        (0, 128),
        (128, 0),
        (128.5, 128.5),
        ("invalid", "size"),
        (128,),
        (128, 128, 128),
    ],
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_flow_field_size_in_scheduler(flow_field_size, scheduler):
    """Test that invalid flow_field_size raises a ValueError."""
    scheduler.get_flow_fields_shape = lambda: flow_field_size + (2,)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "scheduler.get_flow_fields_shape must return a tuple "
            "of three positive integers with the last being 2 or 3; "
            f"got {flow_field_size + (2,)}."
        ),
    ):
        config = sampler_config.copy()
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
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
            config=config,
        )


@pytest.mark.parametrize(
    "velocities_per_pixel", ["invalid_velocities_per_pixel", None, (-1, 1)]
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_velocities_per_pixel(velocities_per_pixel, scheduler):
    """Test that invalid velocities_per_pixel raises a ValueError."""
    with pytest.raises(ValueError, match="velocities_per_pixel must be a number."):
        config = sampler_config.copy()
        config["velocities_per_pixel"] = velocities_per_pixel
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            config=config,
        )


@pytest.mark.parametrize("velocities_per_pixel", [0, -1, -0.5])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_non_positive_velocities_per_pixel(velocities_per_pixel, scheduler):
    """Test that invalid velocities_per_pixel raises a ValueError."""
    with pytest.raises(
        ValueError, match="velocities_per_pixel must be a positive number."
    ):
        config = sampler_config.copy()
        config["velocities_per_pixel"] = velocities_per_pixel
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
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
            config=config,
        )


@pytest.mark.parametrize(
    "diameter_ranges, expected_message",
    [
        (
            [[-1.0, 1.0]],
            "Each diameter_range must satisfy 0 < min <= max.",
        ),
        (
            [[0.0, -1.0]],
            "Each diameter_range must satisfy 0 < min <= max.",
        ),
        (
            [[-0.5, -0.5]],
            "Each diameter_range must satisfy 0 < min <= max.",
        ),
        (
            [[1.0, 0.5]],
            "Each diameter_range must satisfy 0 < min <= max.",
        ),
        (
            [[0.5, 0.1]],
            "Each diameter_range must satisfy 0 < min <= max.",
        ),
    ],
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_diameter_ranges(diameter_ranges, expected_message, scheduler):
    """Test that invalid diameter_range raises a ValueError."""
    with pytest.raises(ValueError, match=expected_message):
        config = sampler_config.copy()
        config["diameter_ranges"] = diameter_ranges
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
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
            config=config,
        )


@pytest.mark.parametrize(
    "intensity_ranges, expected_message",
    [
        (
            [[-1.0, 1.0]],
            "Each intensity_range must satisfy 0 < min <= max.",
        ),
        (
            [[0.0, -1.0]],
            "Each intensity_range must satisfy 0 < min <= max.",
        ),
        (
            [[-0.5, -0.5]],
            "Each intensity_range must satisfy 0 < min <= max.",
        ),
        (
            [[1.0, 0.5]],
            "Each intensity_range must satisfy 0 < min <= max.",
        ),
        (
            [[0.5, 0.1]],
            "Each intensity_range must satisfy 0 < min <= max.",
        ),
    ],
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_intensity_ranges(intensity_ranges, expected_message, scheduler):
    """Test that invalid intensity_ranges raises a ValueError."""
    with pytest.raises(ValueError, match=expected_message):
        config = sampler_config.copy()
        config["intensity_ranges"] = intensity_ranges
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
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
            config=config,
        )


@pytest.mark.parametrize(
    "rho_ranges, expected_message",
    [
        (
            [[-1.1, 1.0]],
            "Each rho_range must satisfy -1 < min <= max < 1.",
        ),
        (
            [[0.0, 1.1]],
            "Each rho_range must satisfy -1 < min <= max < 1.",
        ),
        (
            [[0.9, 0.5]],
            "Each rho_range must satisfy -1 < min <= max < 1.",
        ),
        (
            [[0.5, 0.1]],
            "Each rho_range must satisfy -1 < min <= max < 1.",
        ),
    ],
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_rho_range(rho_ranges, expected_message, scheduler):
    """Test that invalid rho_ranges raises a ValueError."""
    with pytest.raises(ValueError, match=expected_message):
        config = sampler_config.copy()
        config["rho_ranges"] = rho_ranges
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
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
            config=config,
        )


@pytest.mark.parametrize("dt", ["invalid_dt", jnp.array([1]), jnp.array([1.0, 2.0])])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_dt(dt, scheduler):
    """Test that invalid dt raises a ValueError."""
    with pytest.raises(ValueError, match="dt must be a positive number."):
        config = sampler_config.copy()
        config["dt"] = dt
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            config=config,
        )


@pytest.mark.parametrize(
    "noise_uniform", [-1, "a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])]
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_noise_uniform(noise_uniform, scheduler):
    """Test that invalid noise_uniform raises a ValueError."""
    with pytest.raises(
        ValueError, match="noise_uniform must be a non-negative number."
    ):
        config = sampler_config.copy()
        config["noise_uniform"] = noise_uniform
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            config=config,
        )


@pytest.mark.parametrize(
    "noise_gaussian_mean, noise_gaussian_std",
    [(-5.0, 1.0), (5.0, -1.0)],
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_noise_gaussian_params(
    noise_gaussian_mean, noise_gaussian_std, scheduler
):
    """Test that invalid gaussian noise params raise a ValueError."""
    with pytest.raises(ValueError):
        config = sampler_config.copy()
        config["noise_gaussian_mean"] = noise_gaussian_mean
        config["noise_gaussian_std"] = noise_gaussian_std
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
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
            config=config,
        )


@pytest.mark.parametrize(
    "min_speed_x, max_speed_x", [(1, -1), (2, 1), ("invalid", 1), (1, "invalid")]
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_min_max_speed_x(min_speed_x, max_speed_x, scheduler):
    """Test that invalid min_speed_x and max_speed_x raises a ValueError."""
    with pytest.raises(ValueError):
        config = sampler_config.copy()
        config["min_speed_x"] = min_speed_x
        config["max_speed_x"] = max_speed_x
        config["min_speed_y"] = 0.0
        config["max_speed_y"] = 0.0
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            config=config,
        )


@pytest.mark.parametrize(
    "min_speed_y, max_speed_y", [(1, -1), (2, 1), ("invalid", 1), (1, "invalid")]
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_min_max_speed_y(min_speed_y, max_speed_y, scheduler):
    """Test that invalid min_speed_y and max_speed_y raises a ValueError."""
    with pytest.raises(ValueError):
        config = sampler_config.copy()
        config["min_speed_x"] = 0.0
        config["max_speed_x"] = 0.0
        config["min_speed_y"] = min_speed_y
        config["max_speed_y"] = max_speed_y
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            config=config,
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
        f"The size {flow_field_size} of the flow field is too small."
        f" It must be at least "
        f"({position_bounds[0] + position_bounds_offset[0]},"
        f"{position_bounds[1] + position_bounds_offset[1]})."
    )

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
        config=config,
    )

    for batch in sampler:
        assert batch.images1.shape[0] >= batch_size
        assert batch.images1[0].shape >= image_shape
        assert isinstance(batch.images1, jnp.ndarray)


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="This test is skipped in CI for missing data.",
)
@pytest.mark.parametrize(
    "batch_size, batches_per_flow_batch, flow_fields_per_batch",
    [(24, 4, 12), (12, 6, 12)],
)
@pytest.mark.parametrize("mock_mat_files", [64], indirect=True)
def test_sampler_switches_flow_fields(
    batch_size, batches_per_flow_batch, flow_fields_per_batch, mock_mat_files
):

    files, dims = mock_mat_files
    H, W = dims["height"], dims["width"]

    CI = os.getenv("CI") == "true"

    if CI:
        devices = None
    else:
        devices = jax.devices()
        if len(devices) > 4:
            devices = devices[:4]

    batch_size = 3 * 4  # multiple of all number of devices
    scheduler = MATFlowFieldScheduler(files, loop=True, output_shape=(H, W))

    config = sampler_config.copy()
    config["batch_size"] = batch_size
    config["batches_per_flow_batch"] = batches_per_flow_batch
    config["flow_fields_per_batch"] = flow_fields_per_batch
    config["devices"] = devices

    sampler = SyntheticImageSampler.from_config(
        scheduler=scheduler,
        config=config,
    )

    for i, batch in enumerate(sampler):
        if i == 0:
            batch1 = batch
        if i >= batches_per_flow_batch - 1:
            batch2 = batch
            break

    batch3 = next(sampler)

    assert not jnp.allclose(batch1.images1, batch2.images1)
    assert not jnp.allclose(batch1.images2, batch2.images2)
    assert jnp.allclose(batch1.flow_fields, batch2.flow_fields)
    assert not jnp.allclose(batch2.images1, batch3.images1)
    assert not jnp.allclose(batch2.images2, batch3.images2)
    assert not jnp.allclose(batch2.flow_fields, batch3.flow_fields)


@pytest.mark.slow
@pytest.mark.parametrize(
    "image_shape, batches_per_flow_batch, seeding_density_range",
    [((32, 32), 4, (0.1, 0.1)), ((64, 64), 4, (0.0, 0.04))],
)
@pytest.mark.parametrize("batch_size", [12])
@pytest.mark.parametrize("mock_hdf5_files", [64], indirect=True)
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
        config=config,
    )

    batch = next(sampler)

    res = sampler.resolution
    velocities_per_pixel = sampler.velocities_per_pixel
    output_size = jnp.array(
        [
            batch.flow_fields.shape[1] / velocities_per_pixel / res,
            batch.flow_fields.shape[2] / velocities_per_pixel / res,
            batch.flow_fields.shape[3],
        ]
    )

    expected_size = jnp.array([image_shape[0] / res, image_shape[1] / res, 2])

    assert isinstance(batch.images1, jnp.ndarray)
    assert isinstance(batch.images2, jnp.ndarray)
    assert isinstance(batch.flow_fields, jnp.ndarray)
    assert batch.images1.shape == (sampler.batch_size, *image_shape)
    assert batch.images2.shape == (sampler.batch_size, *image_shape)
    assert jnp.allclose(output_size, expected_size, atol=0.01)


@pytest.mark.slow
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
    # Check how many GPUs are available
    devices = jax.devices()
    if len(devices) == 3:
        devices = devices[:2]
    elif len(devices) > 4:
        devices = devices[:4]
    num_devices = len(devices)

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
    config["noise_uniform"] = 0.0
    config["flow_fields_per_batch"] = 64
    config["device_ids"] = [d.id for d in devices]

    if config["flow_fields_per_batch"] % num_devices != 0:
        pytest.skip("flow_fields_per_batch must be divisible by the number of devices.")

    # Limit time in seconds (depends on the number of GPUs)
    if num_devices == 1:
        limit_time = 1.5
    elif num_devices == 2:
        limit_time = 1.3
    elif num_devices == 4:
        limit_time = 1.3

    # Create the sampler
    prefetching_scheduler = PrefetchingFlowFieldScheduler(
        scheduler=scheduler,
        batch_size=config["flow_fields_per_batch"],
        buffer_size=4,
    )
    sampler = SyntheticImageSampler.from_config(
        scheduler=prefetching_scheduler,
        config=config,
    )

    def run_sampler():
        # Generates batches_per_flow_batch batches
        # of size batch_size
        for i, batch in enumerate(sampler):
            batch.images1.block_until_ready()
            batch.images2.block_until_ready()
            batch.flow_fields.block_until_ready()
            batch.params.seeding_densities.block_until_ready()  # type: ignore
            batch.params.diameter_ranges.block_until_ready()  # type: ignore
            batch.params.intensity_ranges.block_until_ready()  # type: ignore
            batch.params.rho_ranges.block_until_ready()  # type: ignore
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


@pytest.mark.parametrize("mock_mat_files", [64], indirect=True)
def test_stop_after_max_episodes(mock_mat_files):
    """Sampler raises StopIteration after the configured `num_episodes`."""

    files, dims = mock_mat_files
    H, W = dims["height"], dims["width"]

    devices = None
    if os.getenv("CI") != "true":
        devices = jax.devices()
        if len(devices) > 4:
            devices = devices[:4]

    num_episodes = 2
    batch_size = 3 * 4  # multiple of all number of devices
    base = MATFlowFieldScheduler(files, loop=True, output_shape=(H, W))
    epi = EpisodicFlowFieldScheduler(
        base,
        batch_size=batch_size,
        episode_length=2,
        key=jax.random.PRNGKey(0),
    )
    pre = PrefetchingFlowFieldScheduler(epi, batch_size=batch_size, buffer_size=90)

    sampler = SyntheticImageSampler(
        scheduler=pre,
        batches_per_flow_batch=1,
        flow_fields_per_batch=batch_size,
        flow_field_size=(H, W),
        resolution=1.0,
        velocities_per_pixel=1.0,
        seed=0,
        max_speed_x=0.0,
        max_speed_y=0.0,
        min_speed_x=0.0,
        min_speed_y=0.0,
        output_units="pixels",
        generation_specification=ImageGenerationSpecification(
            batch_size=batch_size,
            image_shape=(H, W),
            img_offset=(0.0, 0.0),
            seeding_density_range=(0.01, 0.01),
            p_hide_img1=0.0,
            p_hide_img2=0.0,
            diameter_ranges=[(1.0, 1.0)],
            diameter_var=0.0,
            intensity_ranges=[(1.0, 1.0)],
            intensity_var=0.0,
            rho_ranges=[(0.0, 0.0)],
            rho_var=0.0,
            dt=1.0,
            noise_uniform=0.0,
            noise_gaussian_mean=0.0,
            noise_gaussian_std=0.0,
        ),
        device_ids=[d.id for d in devices] if devices is not None else None,
    )

    # We expect exactly num_episodes × episode_length iterations.
    n_batches = 0

    for i in range(num_episodes):
        if i != 0:
            sampler.next_episode()
        batch = next(sampler)
        imgs1 = batch.images1
        done = batch.done
        assert done is not None
        n_batches += 1
        while not any(done):
            batch = next(sampler)
            imgs1 = batch.images1
            done = batch.done
            assert imgs1.shape[0] == batch_size
            assert imgs1[0].shape == (H, W)
            assert isinstance(imgs1, jnp.ndarray)
            n_batches += 1
            assert done is not None

    assert (
        n_batches == epi.episode_length * num_episodes
    ), f"Expected {epi.episode_length * num_episodes} batches, but got {n_batches}"

    # Clean up background thread
    sampler.shutdown()


@pytest.mark.parametrize("mock_mat_files", [64], indirect=True)
def test_index_error_if_no_next_episode(mock_mat_files):
    """Sampler raises IndexError if next_episode() is not called."""

    files, dims = mock_mat_files
    H, W = dims["height"], dims["width"]

    CI = os.getenv("CI") == "true"

    if CI:
        devices = None
    else:
        devices = jax.devices()
        if len(devices) > 4:
            devices = devices[:4]

    batch_size = 3 * 4  # multiple of all number of devices
    base = MATFlowFieldScheduler(files, loop=True, output_shape=(H, W))
    epi = EpisodicFlowFieldScheduler(
        base,
        batch_size=batch_size,
        episode_length=2,
        key=jax.random.PRNGKey(0),
    )
    pre = PrefetchingFlowFieldScheduler(epi, batch_size=batch_size, buffer_size=90)

    sampler = SyntheticImageSampler(
        scheduler=pre,
        batches_per_flow_batch=1,
        flow_fields_per_batch=batch_size,
        flow_field_size=(H, W),
        resolution=1.0,
        velocities_per_pixel=1.0,
        generation_specification=ImageGenerationSpecification(
            batch_size=batch_size,
            image_shape=(H, W),
            img_offset=(0.0, 0.0),
            seeding_density_range=(0.01, 0.01),
            p_hide_img1=0.0,
            p_hide_img2=0.0,
            diameter_ranges=[(1.0, 1.0)],
            diameter_var=0.0,
            intensity_ranges=[(1.0, 1.0)],
            intensity_var=0.0,
            rho_ranges=[(0.0, 0.0)],
            rho_var=0.0,
            dt=1.0,
            noise_uniform=0.0,
            noise_gaussian_mean=0.0,
            noise_gaussian_std=0.0,
        ),
        seed=0,
        max_speed_x=0.0,
        max_speed_y=0.0,
        min_speed_x=0.0,
        min_speed_y=0.0,
        output_units="pixels",
        device_ids=[d.id for d in devices] if devices is not None else None,
    )

    sampler.next_episode()
    batch = next(sampler)
    assert batch is not None
    imgs1 = batch.images1
    done = batch.done
    assert done is not None
    while not any(done):
        batch = next(sampler)
        imgs1 = batch.images1
        done = batch.done
        assert imgs1.shape[0] == batch_size
        assert imgs1[0].shape == (H, W)
        assert isinstance(imgs1, jnp.ndarray)
        assert done is not None
    with pytest.raises(
        EpisodeEnd,
        match=re.escape(
            "Episode ended. No more flow fields available. "
            "Use next_episode() to continue."
        ),
    ):
        next(sampler)

    # Clean up background thread
    sampler.shutdown()


@pytest.mark.parametrize("mock_mat_files", [10], indirect=True)
def test_real_sampler(mock_mat_files):
    """Test the RealImageSampler with a mock HDF5 scheduler."""
    files, _ = mock_mat_files
    scheduler = MATFlowFieldScheduler(files, loop=False, include_images=True)

    prefetcher = PrefetchingFlowFieldScheduler(
        scheduler=scheduler, batch_size=2, buffer_size=4
    )

    sampler = RealImageSampler(scheduler=prefetcher, batch_size=2)

    for batch in sampler:
        assert isinstance(batch.images1, jnp.ndarray)
        assert isinstance(batch.images2, jnp.ndarray)
        assert isinstance(batch.flow_fields, jnp.ndarray)
        assert batch.images1.shape[0] == 2  # batch size
        assert batch.params is None


@pytest.mark.parametrize("mock_mat_files", [10], indirect=True)
def test_reject_wrong_scheduler_for_real_images(mock_mat_files):
    """Test the RealImageSampler with a mock HDF5 scheduler."""
    files, _ = mock_mat_files
    scheduler = MATFlowFieldScheduler(files, loop=False, include_images=False)

    prefetcher = PrefetchingFlowFieldScheduler(
        scheduler=scheduler, batch_size=2, buffer_size=4
    )

    with pytest.raises(
        ValueError,
        match="Base scheduler must have include_images "
        "set to True to use RealImageSampler.",
    ):
        # This should raise an error because the scheduler does not include images
        RealImageSampler(scheduler=prefetcher, batch_size=2)


class _BaseDummy(BaseFlowFieldScheduler):
    """Common helpers for all dummy schedulers."""

    include_images = True
    _batch_ctr = 0

    def __init__(self):
        pass

    def load_file(self, file_path: str) -> SchedulerData:
        assert False, "Not implemented for dummy scheduler."

    def get_next_slice(self) -> SchedulerData:
        assert False, "Not implemented for dummy scheduler."

    def _make_arrays(self, batch_size) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return three arrays that differ per call so we can detect resets."""
        val = float(self._batch_ctr)
        self._batch_ctr += 1
        imgs1 = np.full((batch_size, 2, 2), val, dtype=np.float32)
        imgs2 = imgs1 + 1.0
        flows = np.full((batch_size, 2, 2, 2), -val, dtype=np.float32)
        return imgs1, imgs2, flows

    def get_batch(self, batch_size) -> SchedulerData:
        imgs1, imgs2, flows = self._make_arrays(batch_size)
        return SchedulerData(flow_fields=flows, images1=imgs1, images2=imgs2)

    def get_flow_fields_shape(self):
        """Return the shape of the flow fields."""
        return (2, 2, 2)

    @classmethod
    def from_config(cls, config):
        return cls()


class EpisodicDummy(_BaseDummy, EpisodicSchedulerProtocol):
    """Implements the full episodic API."""

    def __init__(self, episode_length=3):
        self._episode_length = episode_length
        self._step = 0
        self.reset_called = False
        self.shutdown_called = False

    @property
    def episode_length(self):
        return self._episode_length

    # ---------- RealImageSampler hooks ---------------------------------
    def steps_remaining(self):
        return max(self.episode_length - self._step, 0)

    def get_batch(self, batch_size):
        batch = super().get_batch(batch_size=batch_size)
        self._step += 1
        done = jnp.array(self._step >= self.episode_length)
        return batch.update(done=jnp.full((batch_size,), done))

    def next_episode(self):
        self._step = 0

    def reset(self):
        self.reset_called = True
        self._step = 0

    def shutdown(self):
        self.shutdown_called = True


class PlainDummy(_BaseDummy):
    """Non-episodic scheduler - *no* next_episode/steps_remaining/reset/shutdown."""

    pass


class NoResetShutdownDummy(_BaseDummy):
    """Has next_episode but deliberately *omits* reset / shutdown for branch tests."""

    def next_episode(self):
        pass


class SyntheticImageSamplerWrapper:
    """A wrapper for SyntheticImageSampler to use with pytest."""

    @classmethod
    def from_config(cls, scheduler, config):
        full_config = sampler_config.copy()
        full_config["batch_size"] = config.get("batch_size", 4)
        full_config["flow_fields_per_batch"] = full_config["batch_size"]
        return SyntheticImageSampler.from_config(
            scheduler=scheduler,
            config=full_config,
        )


@pytest.mark.parametrize(
    "sampler_class", [RealImageSampler, SyntheticImageSamplerWrapper]
)
def test_episodic_done_and_episode_end(sampler_class):
    sched = EpisodicDummy(episode_length=2)
    sampler = sampler_class.from_config(sched, {"batch_size": 4})

    first = next(sampler)
    assert first.done is not None and not first.done.any()

    last = next(sampler)
    assert last.done is not None and last.done.all()  # last step → all True

    with pytest.raises(EpisodeEnd):
        next(sampler)  # overrun episode


@pytest.mark.parametrize(
    "sampler_class", [RealImageSampler, SyntheticImageSamplerWrapper]
)
def test_reset_and_shutdown(sampler_class):
    sched = EpisodicDummy(episode_length=2)
    if sampler_class is SyntheticImageSamplerWrapper:
        old_get_batch = sched.get_batch
        sched.get_batch = lambda *a, **kw: old_get_batch(*a, **kw)
    sampler = sampler_class.from_config(sched, {"batch_size": 4})

    next(sampler)
    # reset the sampler
    sampler.reset()
    assert sched.reset_called, "Scheduler reset was not called"

    sampler.shutdown()  # shutdown the sampler
    assert sched.shutdown_called, "Scheduler shutdown was not called"


@pytest.mark.parametrize(
    "sampler_class", [RealImageSampler, SyntheticImageSamplerWrapper]
)
def test_next_episode_attribute_error(sampler_class):
    sampler = sampler_class.from_config(
        PlainDummy(),
        {
            "batch_size": 1,
        },
    )
    with pytest.raises(AttributeError, match="next_episode"):
        sampler.next_episode()


@pytest.mark.parametrize(
    "sampler_class", [RealImageSampler, SyntheticImageSamplerWrapper]
)
def test_make_done_not_implemented(sampler_class):
    sampler = sampler_class.from_config(PlainDummy(), {"batch_size": 1})
    with pytest.raises(NotImplementedError):
        sampler._make_done()


def test_batch_size_adjusted_when_not_divisible_by_ndevices(monkeypatch):
    # Simulate two JAX devices on CPU
    class FakeDevice:
        pass

    fake_devices = [FakeDevice(), FakeDevice()]
    monkeypatch.setattr(jax, "devices", lambda: fake_devices)

    # batch_size=3 is not divisible by ndevices=2 -> adjusted to 4
    sampler = SyntheticImageSampler(
        scheduler=_BaseDummy(),
        batches_per_flow_batch=1,
        flow_field_size=(8, 8),
        resolution=1.0,
        velocities_per_pixel=1.0,
        generation_specification=ImageGenerationSpecification(
            image_shape=(5, 5),
            img_offset=(1.0, 1.0),
            seeding_density_range=(1.0, 1.0),
            p_hide_img1=0.0,
            p_hide_img2=0.0,
            diameter_ranges=[(1.0, 1.0)],
            diameter_var=0.1,
            intensity_ranges=[(1.0, 1.0)],
            intensity_var=0.1,
            rho_ranges=[(0.0, 0.0)],
            rho_var=0.1,
            dt=1.0,
            noise_uniform=0.0,
            noise_gaussian_mean=0.0,
            noise_gaussian_std=0.0,
            batch_size=3,
        ),
        seed=0,
        max_speed_x=1.0,
        max_speed_y=1.0,
        min_speed_x=0.0,
        min_speed_y=0.0,
        output_units="pixels",
        flow_fields_per_batch=1,
        device_ids=None,  # use all devices (the two fakes)
    )

    # should have bumped batch_size from 3 to 4
    assert sampler.batch_size == 4, "Expected batch_size to be 4"


def test_warning_when_batch_size_not_divisible_by_flow_fields(monkeypatch):
    # Collect warning messages
    logged = []

    import synthpix.sampler.synthetic as sampler_mod

    # Patch the logger that that module is already using
    monkeypatch.setattr(
        sampler_mod.logger,
        "warning",
        lambda msg: logged.append(msg),
    )
    # Use a batch_size that isn't divisible by flow_fields_per_batch
    sampler = SyntheticImageSampler(
        scheduler=_BaseDummy(),
        batches_per_flow_batch=1,
        flow_field_size=(8, 8),
        resolution=1.0,
        velocities_per_pixel=1.0,
        generation_specification=ImageGenerationSpecification(
            image_shape=(5, 5),
            img_offset=(1.0, 1.0),
            seeding_density_range=(1.0, 1.0),
            p_hide_img1=0.0,
            p_hide_img2=0.0,
            diameter_ranges=[(1.0, 1.0)],
            diameter_var=0.1,
            intensity_ranges=[(1.0, 1.0)],
            intensity_var=0.1,
            rho_ranges=[(0.0, 0.0)],
            rho_var=0.1,
            dt=1.0,
            noise_uniform=0.0,
            noise_gaussian_mean=0.0,
            noise_gaussian_std=0.0,
            batch_size=4,
        ),
        seed=0,
        max_speed_x=1.0,
        max_speed_y=1.0,
        min_speed_x=0.0,
        min_speed_y=0.0,
        output_units="pixels",
        flow_fields_per_batch=3,
        device_ids=None,
    )

    # batch_size itself should be unchanged (1 device)
    assert sampler.batch_size == 4, "Expected batch_size to be 4"

    # And we should have logged the expected warning
    expected = (
        "batch_size was not divisible by number of flows per batch. "
        "There will be one more sample for the first 1 flow fields of each batch."
    )
    assert any(expected in m for m in logged), (
        f"Expected warning: {expected}, " f"but got: {logged}"
    )
