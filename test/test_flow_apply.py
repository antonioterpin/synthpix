import timeit

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from synthpix.apply import (
    apply_flow_to_image_callable,
    apply_flow_to_particles,
    input_check_apply_flow,
)
from synthpix.example_flows import get_flow_function
from synthpix.generate import img_gen_from_data, img_gen_from_density
from synthpix.utils import generate_array_flow_field, load_configuration

config = load_configuration("config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_APPLY"]


@pytest.mark.parametrize(
    "image_shape", [(16, 16), (64, 32), (32, 64), (256, 128), (128, 256), (256, 256)]
)
def test_flow_apply_to_image(image_shape, visualize=False):
    """Test that we can apply a flow field to a synthetic image."""
    # 1. Generate a synthetic particle image
    key = jax.random.PRNGKey(0)
    img = img_gen_from_density(
        key,
        image_shape=image_shape,
        seeding_density=0.1,
        diameter_range=(0.1, 1.0),
        intensity_range=(50, 200),
    )

    # 2. Apply a simple horizontal flow
    def flow_f(_t, _x, _y):
        return 1.0, 0.0

    img_warped = apply_flow_to_image_callable(img, flow_f, t=0.0)
    if visualize:
        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(np.array(img), cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title("Warped Image")
        plt.imshow(np.array(img_warped), cmap="gray")
        plt.show()

    # 3. Check image shapes
    assert img.shape == img_warped.shape, "Image shapes do not match"


@pytest.mark.parametrize("selected_flow", ["vertical"])
@pytest.mark.parametrize("seeding_density", [0.1])
@pytest.mark.parametrize("image_shape", [(128, 128)])
def test_particles_flow_apply_array(
    selected_flow, seeding_density, image_shape, visualize=False
):
    """Test that we can apply a flow field as a jax array to random particles."""

    # 1. Generate random particles
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key, 2)

    particles_number = int(image_shape[0] * image_shape[1] * seeding_density)
    particles = jax.random.uniform(
        subkey, (particles_number, 2), minval=0.0, maxval=jnp.array(image_shape) - 1
    )

    # 2. create a synthetic image
    img = img_gen_from_data(
        key,
        image_shape=image_shape,
        particle_positions=particles,
        diameter_range=(2, 3.5),
        intensity_range=(10, 255),
        rho_range=(-0.1, 0.1),
    )

    # 3. create a flow field
    flow_field = generate_array_flow_field(
        get_flow_function(selected_flow, image_shape), image_shape
    )

    # 4. Apply the flow to the particles
    new_particles = apply_flow_to_particles(particles, flow_field)

    # 5. create a synthetic image with the new particles
    img_warped = img_gen_from_data(
        key,
        image_shape=image_shape,
        particle_positions=new_particles,
        diameter_range=(2, 3.5),
        intensity_range=(10, 250),
        rho_range=(-0.5, 0.5),
    )

    if visualize:
        import matplotlib.pyplot as plt
        import numpy as np

        plt.imsave("img.png", np.array(img), cmap="gray")
        plt.imsave("img_warped.png", np.array(img_warped), cmap="gray")

    # 6. Check particles shapes
    assert particles.shape == new_particles.shape, "Particles shapes do not match"


@pytest.mark.parametrize(
    "particle_positions", [1, [[1, 2], [3, 4]], [[1, 2, 3], [4, 5, 6]]]
)
def test_invalid_particle_positions(particle_positions):
    """Test that invalid particle_positions raise a ValueError."""
    flow_field = jnp.zeros((128, 128, 2))
    with pytest.raises(
        ValueError,
        match=(
            "Particle_positions must be a 2D jnp.ndarray with shape "
            "\\(N, 2\\) or \\(N, 3\\)"
        ),
    ):
        input_check_apply_flow(particle_positions, flow_field)


@pytest.mark.parametrize(
    "flow_field",
    [1, jnp.array([1, 2, 3]), [[[10, 20]]], jnp.array([1, 2, 3]), [[[10, 20, 30]]]],
)
def test_invalid_flow_field(flow_field):
    """Test that invalid flow_field raise a ValueError."""
    particle_positions = jnp.zeros((1, 2))
    with pytest.raises(
        ValueError,
        match=(
            "Flow_field must be a 3D jnp.ndarray with shape "
            "\\(H, W, 2\\) or \\(H, W, 3\\)"
        ),
    ):
        input_check_apply_flow(particle_positions, flow_field)


@pytest.mark.parametrize(
    "flow_field, particle_positions, error_msg",
    [
        (
            jnp.zeros((128, 128, 3)),
            jnp.zeros((1, 2)),
            "Particle positions are in 2D, but the flow field is in 3D.",
        ),
        (
            jnp.zeros((128, 128, 2)),
            jnp.zeros((1, 3)),
            "Particle positions are in 3D, but the flow field is in 2D.",
        ),
    ],
)
def test_invalid_flow_field_shape(flow_field, particle_positions, error_msg):
    """Test that invalid flow_field shape raise a ValueError."""
    with pytest.raises(
        ValueError,
        match=error_msg,
    ):
        input_check_apply_flow(particle_positions, flow_field)


@pytest.mark.parametrize("dt", ["a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])])
def test_invalid_dt(dt):
    """Test that invalid dt raise a ValueError."""
    particle_positions = jnp.zeros((1, 2))
    flow_field = jnp.zeros((128, 128, 2))
    with pytest.raises(ValueError, match="dt must be a scalar \\(int or float\\)"):
        input_check_apply_flow(particle_positions, flow_field, dt)


@pytest.mark.parametrize(
    "flow_field_res_x",
    ["a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])],
)
def test_invalid_flow_field_res_x(flow_field_res_x):
    """Test that invalid flow_field_res_x raise a ValueError."""
    particle_positions = jnp.zeros((1, 2))
    flow_field = jnp.zeros((128, 128, 2))
    with pytest.raises(
        ValueError,
        match="flow_field_res_x must be a positive scalar \\(int or float\\)",
    ):
        input_check_apply_flow(
            particle_positions, flow_field, flow_field_res_x=flow_field_res_x
        )


@pytest.mark.parametrize(
    "flow_field_res_y",
    ["a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])],
)
def test_invalid_flow_field_res_y(flow_field_res_y):
    """Test that invalid flow_field_res_y raise a ValueError."""
    particle_positions = jnp.zeros((1, 2))
    flow_field = jnp.zeros((128, 128, 2))
    with pytest.raises(
        ValueError,
        match="flow_field_res_y must be a positive scalar \\(int or float\\)",
    ):
        input_check_apply_flow(
            particle_positions, flow_field, flow_field_res_y=flow_field_res_y
        )


@pytest.mark.parametrize(
    "flow_field_res_z",
    ["a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])],
)
def test_invalid_flow_field_res_z(flow_field_res_z):
    """Test that invalid flow_field_res_z raise a ValueError."""
    particle_positions = jnp.zeros((1, 2))
    flow_field = jnp.zeros((128, 128, 2))
    with pytest.raises(
        ValueError,
        match="flow_field_res_z must be a positive scalar \\(int or float\\)",
    ):
        input_check_apply_flow(
            particle_positions, flow_field, flow_field_res_z=flow_field_res_z
        )


# skipif is used to skip the test if the user is not connected to the server
@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize("selected_flow", ["horizontal"])
@pytest.mark.parametrize("seeding_density", [0.016])
@pytest.mark.parametrize("image_shape", [(1216, 1936)])
def test_speed_apply_flow_to_particles(seeding_density, selected_flow, image_shape):
    """Test that apply_flow_to_particles is faster than a limit time."""

    # Name of the axis for the device mesh
    shard_particles = "particles"

    # Check how many GPUs are available
    num_devices = len(jax.devices())

    # Limit time in seconds (depends on the number of GPUs)
    if num_devices == 1:
        limit_time = 4e-5
    elif num_devices == 2:
        limit_time = 7e-5
    elif num_devices == 4:
        limit_time = 8e-4

    # Setup device mesh
    # We want to shard the particles along the first axis
    # and replicate the flow field along all devices.
    # The idea is that each device will apply the flow to a part of the particles
    # and then we will combine the results.
    devices = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(devices, axis_names=(shard_particles))

    # 1. Generate random particles
    key = jax.random.PRNGKey(0)

    # Compute the number of particles and round it to the number of devices
    particles_number = int(image_shape[0] * image_shape[1] * seeding_density)
    particles_number = (particles_number // num_devices + 1) * num_devices
    particles = jax.random.uniform(
        key, (particles_number, 2), minval=0.0, maxval=jnp.array(image_shape) - 1
    )

    # 2. Send the particles to the devices
    sharding_particles = NamedSharding(mesh, PartitionSpec(shard_particles))
    particles_sharded = jax.device_put(particles, sharding_particles)

    # 3. Create a flow field
    flow_field = generate_array_flow_field(
        get_flow_function(selected_flow, image_shape), image_shape
    )

    # 4. Duplicate the flow field to all devices
    sharding_flow_field = NamedSharding(mesh, PartitionSpec())
    flow_field_replicated = jax.device_put(flow_field, sharding_flow_field)

    # 5. Create the jit function
    apply_flow_to_particles_jit = jax.jit(apply_flow_to_particles)

    def run_apply_jit():
        result = apply_flow_to_particles_jit(particles_sharded, flow_field_replicated)
        result.block_until_ready()

    # Warm up the function
    run_apply_jit()

    # Measure the time of the jit function
    total_time_jit = timeit.repeat(
        stmt=run_apply_jit, number=NUMBER_OF_EXECUTIONS, repeat=REPETITIONS
    )
    average_time_jit = min(total_time_jit) / NUMBER_OF_EXECUTIONS

    # Check if the time is less than the limit
    assert (
        average_time_jit < limit_time
    ), f"The average time is {average_time_jit}, time limit: {limit_time}"


if __name__ == "__main__":
    for d in jax.devices():
        print(d.id, d.device_kind, d.platform)
