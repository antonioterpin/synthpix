import timeit

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec

from synthpix.generate import (
    add_noise_to_image,
    img_gen_from_data,
    img_gen_from_density,
    input_check_img_gen_from_data,
)
from synthpix.utils import load_configuration

config = load_configuration("config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_IMG_GEN"]


@pytest.mark.parametrize(
    "img_gen", [img_gen_from_density, input_check_img_gen_from_data]
)
@pytest.mark.parametrize(
    "image_shape", [(-1, 128), (128, -1), (0, 128), (128, 0), (128.2, 128.2)]
)
def test_invalid_image_shape(img_gen, image_shape):
    """Test that invalid image shapes raise a ValueError."""
    key = jax.random.PRNGKey(0)
    with pytest.raises(
        ValueError, match="image_shape must be a tuple of two positive integers."
    ):
        img_gen(key, image_shape=image_shape)


@pytest.mark.parametrize("seeding_density", [-0.1, 0, 1.1])
def test_invalid_seeding_density(seeding_density):
    """Test that invalid seeding densities raise a ValueError."""
    key = jax.random.PRNGKey(0)
    with pytest.raises(
        ValueError, match="seeding_density must be a float between 0 and 1."
    ):
        img_gen_from_density(key, seeding_density=seeding_density)


@pytest.mark.parametrize("particle_positions", [jnp.array([1]), jnp.array([1, 1, 1])])
def test_invalid_particle_positions(particle_positions):
    """Test that invalid particle positions raise a ValueError."""
    key = jax.random.PRNGKey(0)
    with pytest.raises(
        ValueError, match="Particle positions must be a 2D array with shape \\(N, 2\\)"
    ):
        input_check_img_gen_from_data(key, particle_positions=particle_positions)


@pytest.mark.parametrize(
    "img_gen", [img_gen_from_density, input_check_img_gen_from_data]
)
@pytest.mark.parametrize("diameter_range", [(0, 1), (1, 0), (-1, 1), (1, -1)])
def test_invalid_diameter_range(img_gen, diameter_range):
    """Test that invalid diameter ranges raise a ValueError."""
    key = jax.random.PRNGKey(0)
    with pytest.raises(
        ValueError, match="diameter_range must be a tuple of two positive floats."
    ):
        img_gen(key, diameter_range=diameter_range)


@pytest.mark.parametrize(
    "img_gen", [img_gen_from_density, input_check_img_gen_from_data]
)
@pytest.mark.parametrize("intensity_range", [(-1, 200), (50, -1), (1, 1, 1)])
def test_invalid_intensity_range(img_gen, intensity_range):
    """Test that invalid intensity ranges raise a ValueError."""
    key = jax.random.PRNGKey(0)
    with pytest.raises(
        ValueError, match="intensity_range must be a tuple of two positive floats"
    ):
        img_gen(key, intensity_range=intensity_range)


@pytest.mark.parametrize("rho_range", [(-1.1, 1), (1, -1.1), (1, 1, 1)])
def test_invalid_rho_range(rho_range):
    """Test that invalid rho ranges raise a ValueError."""
    key = jax.random.PRNGKey(0)
    with pytest.raises(
        ValueError,
        match="rho_range must be a tuple of two floats in the range \\[-1, 1\\]",
    ):
        input_check_img_gen_from_data(key, rho_range=rho_range)


@pytest.mark.parametrize("clip", [-1, "invalid", 1.1])
def test_invalid_clip(clip):
    """Test that invalid clip values raise a ValueError."""
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError, match="clip must be a boolean value."):
        input_check_img_gen_from_data(key, clip=clip)


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("image_shape", [(128, 128)])
@pytest.mark.parametrize(
    "img_gen, input_val",
    [
        (img_gen_from_density, 0.06),
        (img_gen_from_density, 0.99),
        (img_gen_from_data, jnp.array([[1, 1], [64, 64], [127, 127], [127, 1]])),
    ],
)
@pytest.mark.parametrize("noise_level", [0.0, 5.0, 255.0])
def test_generate_image(
    seed, image_shape, img_gen, input_val, noise_level, visualize=False
):
    """Test that we can generate a synthetic particle image."""
    key = jax.random.PRNGKey(seed)
    img = img_gen(
        key,
        image_shape,
        input_val,
        diameter_range=(0.5, 3.5),
        intensity_range=(500, 500),
    )

    img_background = add_noise_to_image(key, img, noise_level=noise_level)

    if visualize:
        import matplotlib.pyplot as plt
        import numpy as np

        plt.imsave("img.png", np.array(img), cmap="gray")
        plt.imsave("img_background.png", np.array(img_background), cmap="gray")

    assert img.shape == image_shape, "Image shape is incorrect"
    assert img.min() >= 0, "Image contains negative values"
    assert img.max() <= 255, "Image contains values above 255"

    assert img_background.shape == image_shape, "Image shape is incorrect"
    assert img_background.min() >= 0, "Image contains negative values"
    assert img_background.max() <= 255, "Image contains values above 255"


# skipif is used to skip the test if there is no GPU available
@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize("seeding_density", [0.016])
@pytest.mark.parametrize("image_shape", [(1216, 1936)])
def test_speed_img_gen(seeding_density, image_shape):
    """Test that img_gen_from_data is faster than a limit time."""

    # Name of the axis for the device mesh
    shard_particles = "particles"

    # Check how many GPUs are available
    num_devices = len(jax.devices())

    # Limit time in seconds (depends on the number of GPUs)
    if num_devices == 1:
        limit_time = 8.5e-5
    elif num_devices == 2:
        limit_time = 1.3e-4
    elif num_devices == 4:
        limit_time = 9e-5

    # Setup device mesh
    # We want to shard the particles along the first axis
    # and send a key to each device.
    # The idea is that each device will generate a image
    # and then stack it with the images generated by the other GPUs.
    devices = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(devices, axis_names=(shard_particles))

    # 1. Generate random particles and keys
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    keys = jax.random.split(key, num_devices)
    keys = jnp.stack(keys)

    particles_number = int(image_shape[0] * image_shape[1] * seeding_density)
    particles = jax.random.uniform(
        subkey,
        (particles_number * num_devices, 2),
        minval=0.0,
        maxval=jnp.array(image_shape) - 1,
    )

    # 2. Create the jit function
    img_gen_from_data_jit = jax.jit(
        shard_map(
            lambda keys, particles: img_gen_from_data(keys, image_shape, particles),
            mesh=mesh,
            in_specs=(PartitionSpec(shard_particles), PartitionSpec(shard_particles)),
            out_specs=PartitionSpec(shard_particles),
        )
    )

    def run_img_gen_jit():
        img = img_gen_from_data_jit(keys, particles)
        img.block_until_ready()

    # Warm up the function
    run_img_gen_jit()

    # 3. Measure the time of the jit function
    # We divide by the number of devices because shard_map
    # will return Number of devices results, like this we keep the number of
    # images generated the same as the number of devices changes
    total_time_jit = timeit.repeat(
        stmt=run_img_gen_jit,
        number=NUMBER_OF_EXECUTIONS // num_devices,
        repeat=REPETITIONS,
    )

    # Average time
    average_time_jit = min(total_time_jit) / NUMBER_OF_EXECUTIONS

    # 4. Check if the time is less than the limit
    assert (
        average_time_jit < limit_time
    ), f"The average time is {average_time_jit}, time limit: {limit_time}"


if __name__ == "__main__":
    test_generate_image(
        seed=0, image_shape=(16, 16), density=0.1, noise_level=5.0, visualize=True
    )
