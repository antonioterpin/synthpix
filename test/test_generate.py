import timeit

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec

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
    "image_shape", [(-1, 128), (128, -1), (0, 128), (128, 0), (128.2, 128.2)]
)
def test_invalid_image_shape_img_gen_from_density(image_shape):
    """Test that invalid image shapes raise a ValueError."""
    key = jax.random.PRNGKey(0)
    with pytest.raises(
        ValueError, match="image_shape must be a tuple of two positive integers."
    ):
        img_gen_from_density(key, image_shape=image_shape)


@pytest.mark.parametrize(
    "image_shape", [(-1, 128), (128, -1), (0, 128), (128, 0), (128.2, 128.2)]
)
def test_invalid_image_shape_img_gen_from_data(image_shape):
    """Test that invalid image shapes raise a ValueError."""
    with pytest.raises(
        ValueError, match="image_shape must be a tuple of two positive integers."
    ):
        input_check_img_gen_from_data(image_shape=image_shape)


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
    with pytest.raises(
        ValueError, match="Particle positions must be a 2D array with shape \\(N, 2\\)"
    ):
        input_check_img_gen_from_data(particle_positions=particle_positions)


@pytest.mark.parametrize("diameter_range", [(0, 1), (1, 0), (-1, 1), (1, -1)])
def test_invalid_diameter_range(diameter_range):
    """Test that invalid diameter ranges raise a ValueError."""
    key = jax.random.PRNGKey(0)
    with pytest.raises(
        ValueError, match="diameter_range must be a tuple of two positive floats."
    ):
        img_gen_from_density(key, diameter_range=diameter_range)


@pytest.mark.parametrize("intensity_range", [(-1, 200), (50, -1), (1, 1, 1)])
def test_invalid_intensity_range(intensity_range):
    """Test that invalid intensity ranges raise a ValueError."""
    key = jax.random.PRNGKey(0)
    with pytest.raises(
        ValueError, match="intensity_range must be a tuple of two positive floats"
    ):
        img_gen_from_density(key, intensity_range=intensity_range)


@pytest.mark.parametrize("max_diameter", [-1, 0, "invalid"])
def test_invalid_max_diameter(max_diameter):
    """Test that invalid max_diameter values raise a ValueError."""
    particle_positions = jnp.ones((2, 2))
    with pytest.raises(ValueError, match="max_diameter must be a positive number."):
        input_check_img_gen_from_data(
            particle_positions=particle_positions, max_diameter=max_diameter
        )


@pytest.mark.parametrize("diameters_x", [1, jnp.array([1]), jnp.array([1, 1, 1])])
def test_invalid_diameters_x(diameters_x):
    """Test that invalid diameters_x values raise a ValueError."""
    particle_positions = jnp.ones((2, 2))
    with pytest.raises(
        ValueError,
        match="diameters_x must be a 1D array "
        "with the same length as particle_positions.",
    ):
        input_check_img_gen_from_data(
            particle_positions=particle_positions, diameters_x=diameters_x
        )


@pytest.mark.parametrize("diameters_y", [1, jnp.array([1]), jnp.array([1, 1, 1])])
def test_invalid_diameters_y(diameters_y):
    """Test that invalid diameters_y values raise a ValueError."""
    particle_positions = jnp.ones((2, 2))
    diameters_x = jnp.ones(particle_positions.shape[0])
    with pytest.raises(
        ValueError,
        match="diameters_y must be a 1D array "
        "with the same length as particle_positions.",
    ):
        input_check_img_gen_from_data(
            particle_positions=particle_positions,
            diameters_x=diameters_x,
            diameters_y=diameters_y,
        )


@pytest.mark.parametrize("intensities", [1, jnp.array([1]), jnp.array([1, 1, 1])])
def test_invalid_intensities(intensities):
    """Test that invalid intensities values raise a ValueError."""
    particle_positions = jnp.ones((2, 2))
    diameters_x = jnp.ones(particle_positions.shape[0])
    diameters_y = jnp.ones(particle_positions.shape[0])
    with pytest.raises(
        ValueError,
        match="intensities must be a 1D array "
        "with the same length as particle_positions.",
    ):
        input_check_img_gen_from_data(
            particle_positions=particle_positions,
            diameters_x=diameters_x,
            diameters_y=diameters_y,
            intensities=intensities,
        )


@pytest.mark.parametrize("rho", [1, jnp.array([1]), jnp.array([1, 1, 1])])
def test_invalid_rho(rho):
    """Test that invalid rho values raise a ValueError."""
    particle_positions = jnp.ones((2, 2))
    diameters_x = jnp.ones(particle_positions.shape[0])
    diameters_y = jnp.ones(particle_positions.shape[0])
    intensities = jnp.ones(particle_positions.shape[0])
    with pytest.raises(
        ValueError,
        match="rho must be a 1D array with the same length as particle_positions.",
    ):
        input_check_img_gen_from_data(
            particle_positions=particle_positions,
            diameters_x=diameters_x,
            diameters_y=diameters_y,
            intensities=intensities,
            rho=rho,
        )


@pytest.mark.parametrize("clip", [-1, "invalid", 1.1])
def test_invalid_clip(clip):
    """Test that invalid clip values raise a ValueError."""
    particle_positions = jnp.ones((2, 2))
    diameters_x = jnp.ones(particle_positions.shape[0])
    diameters_y = jnp.ones(particle_positions.shape[0])
    intensities = jnp.ones(particle_positions.shape[0])
    rho = jnp.ones(particle_positions.shape[0])
    with pytest.raises(ValueError, match="clip must be a boolean value."):
        input_check_img_gen_from_data(
            clip=clip,
            particle_positions=particle_positions,
            diameters_x=diameters_x,
            diameters_y=diameters_y,
            intensities=intensities,
            rho=rho,
        )


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("image_shape", [(128, 128)])
@pytest.mark.parametrize(
    "particle_positions",
    [
        jnp.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]]),
        jnp.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5], [3.5, 3.5]]),
    ],
)
@pytest.mark.parametrize("noise_level", [0.0, 5.0, 255.0])
def test_generate_image_from_data(
    seed, image_shape, particle_positions, noise_level, visualize=False
):
    """Test that we can generate a synthetic particle image."""
    key = jax.random.PRNGKey(seed)
    diameters_x = jnp.ones(particle_positions.shape[0])
    diameters_y = jnp.ones(particle_positions.shape[0])
    intensities = jnp.ones(particle_positions.shape[0]) * 255
    rho = jnp.ones(particle_positions.shape[0]) * 0.5

    img = img_gen_from_data(
        image_shape=image_shape,
        particle_positions=particle_positions,
        diameters_x=diameters_x,
        diameters_y=diameters_y,
        intensities=intensities,
        rho=rho,
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


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("image_shape", [(128, 128)])
@pytest.mark.parametrize("seeding_density", [0.06, 0.99])
@pytest.mark.parametrize("noise_level", [0.0, 5.0, 255.0])
def test_generate_image_from_density(
    seed, image_shape, seeding_density, noise_level, visualize=False
):
    """Test that we can generate a synthetic particle image."""
    key = jax.random.PRNGKey(seed)
    img = img_gen_from_density(
        key,
        image_shape,
        seeding_density=seeding_density,
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
@pytest.mark.parametrize("diameter_range", [(0.5, 3.5)])
@pytest.mark.parametrize("intensity_range", [(0, 250)])
@pytest.mark.parametrize("rho_range", [(-0.5, 0.5)])
def test_speed_img_gen(
    seeding_density, image_shape, diameter_range, intensity_range, rho_range
):
    """Test that img_gen_from_data is faster than a limit time."""

    # Name of the axis for the device mesh
    shard_particles = "particles"

    # Check how many GPUs are available
    num_devices = len(jax.devices())

    # Limit time in seconds (depends on the number of GPUs)
    if num_devices == 1:
        limit_time = 7.1e-5
    elif num_devices == 2:
        limit_time = 4.5e-5
    elif num_devices == 4:
        limit_time = 2.1e-5

    # Setup device mesh
    # We want to shard the particles and their characteristics
    # across the GPUs.
    # The idea is that each device will generate a image
    # and then stack it with the images generated by the other GPUs.
    devices = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(devices, axis_names=(shard_particles))

    # 1. Generate random particles and their characteristics
    key = jax.random.PRNGKey(0)
    subkey1, subkey2, subkey3, subkey4, subkey5 = jax.random.split(key, 5)

    particles_number = int(image_shape[0] * image_shape[1] * seeding_density)
    particles = jax.random.uniform(
        subkey1,
        (particles_number * num_devices, 2),
        minval=0.0,
        maxval=jnp.array(image_shape) - 1,
    )
    diameters_x = jax.random.uniform(
        subkey2,
        (particles_number * num_devices,),
        minval=diameter_range[0],
        maxval=diameter_range[1],
    )
    diameters_y = jax.random.uniform(
        subkey3,
        (particles_number * num_devices,),
        minval=diameter_range[0],
        maxval=diameter_range[1],
    )
    intensities = jax.random.uniform(
        subkey4,
        (particles_number * num_devices,),
        minval=intensity_range[0],
        maxval=intensity_range[1],
    )
    rho = jax.random.uniform(
        subkey5,
        (particles_number * num_devices,),
        minval=rho_range[0],
        maxval=rho_range[1],
    )

    # Sending the variables to the devices
    # To make the test also consider the time of sending the variables to the devices
    # comment the next lines
    particles = jax.device_put(
        particles, NamedSharding(mesh, PartitionSpec(shard_particles))
    )
    diameters_x = jax.device_put(
        diameters_x, NamedSharding(mesh, PartitionSpec(shard_particles))
    )
    diameters_y = jax.device_put(
        diameters_y, NamedSharding(mesh, PartitionSpec(shard_particles))
    )
    intensities = jax.device_put(
        intensities, NamedSharding(mesh, PartitionSpec(shard_particles))
    )
    rho = jax.device_put(rho, NamedSharding(mesh, PartitionSpec(shard_particles)))

    _img_gen_fun = (
        lambda particles, diameters_x, diameters_y, intensities, rho: img_gen_from_data(
            image_shape=image_shape,
            particle_positions=particles,
            diameters_x=diameters_x,
            diameters_y=diameters_y,
            intensities=intensities,
            rho=rho,
        )
    )

    # 2. Create the jit function
    img_gen_from_data_jit = jax.jit(
        shard_map(
            _img_gen_fun,
            mesh=mesh,
            in_specs=(
                PartitionSpec(shard_particles),
                PartitionSpec(shard_particles),
                PartitionSpec(shard_particles),
                PartitionSpec(shard_particles),
                PartitionSpec(shard_particles),
            ),
            out_specs=PartitionSpec(shard_particles),
        )
    )

    def run_img_gen_jit():
        img = img_gen_from_data_jit(
            particles,
            diameters_x,
            diameters_y,
            intensities,
            rho,
        )
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
    test_generate_image_from_density(
        seed=0, image_shape=(16, 16), density=0.1, noise_level=5.0, visualize=True
    )
