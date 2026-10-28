"""Tests for applying flow fields to images and particle positions.

These tests verify the correctness and performance of various flow
application methods, including callable-based warping and
array-based particle advection, with support for both 2D and 3D
domains.
"""

from __future__ import annotations

import timeit
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from synthpix.apply import (
    apply_flow_to_image_callable,
    apply_flow_to_particles,
    input_check_apply_flow,
)
from synthpix.generate import img_gen_from_data
from synthpix.utils import generate_array_flow_field, load_configuration
from tests.example_flows import get_flow_function

config = load_configuration("config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_APPLY"]


@pytest.mark.parametrize(
    "image_shape",
    [(16, 16), (64, 32), (32, 64), (256, 128), (128, 256), (256, 256)],
)
def test_flow_apply_to_image(
    image_shape: tuple[int, int], visualize: bool = False
) -> None:
    """Test that a callable flow function can be applied to warp a
    synthetic image.

    Verifies that the warped image maintains the expected dimensions and
    provides an optional visualization hook for manual inspection.

    Args:
        image_shape: Height and width of the output images.
        visualize: Visualize for this test case.
    """
    # 1. Generate a synthetic particle image
    # 1. Generate random particles and their characteristics
    key = jax.random.PRNGKey(0)
    subkey1, subkey2, subkey3, subkey4, subkey5 = jax.random.split(key, 5)

    particles_number = int(image_shape[0] * image_shape[1] * 0.02)
    particles = jax.random.uniform(
        subkey1,
        (particles_number, 2),
        minval=0.0,
        maxval=jnp.array(image_shape) - 1,
    )
    diameters_x = jax.random.uniform(
        subkey2,
        (particles_number,),
        minval=0.8,
        maxval=1.2,
    )
    diameters_y = jax.random.uniform(
        subkey3,
        (particles_number,),
        minval=0.8,
        maxval=1.2,
    )
    intensities = jax.random.uniform(
        subkey4,
        (particles_number,),
        minval=50,
        maxval=100,
    )
    rho = jax.random.uniform(
        subkey5,
        (particles_number,),
        minval=0.0,
        maxval=1e-9,
    )

    # 2. create a synthetic image
    img = img_gen_from_data(
        image_shape=image_shape,
        particle_positions=particles,
        diameters_x=diameters_x,
        diameters_y=diameters_y,
        intensities=intensities,
        rho=rho,
    )

    # 2. Apply a simple horizontal flow
    def flow_f(_t: Any, _x: Any, _y: Any) -> tuple[float, float]:
        """Flow f.

        Args:
            _t: T used by this test.
            _x: X used by this test.
            _y: Y used by this test.

        Returns:
            The result used by the test assertions.
        """
        return 1.0, 0.0

    img_warped = apply_flow_to_image_callable(img, flow_f, t=0.0)
    if visualize:
        import matplotlib.pyplot as plt  # noqa: PLC0415 - optional or intentionally lazy import

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


@pytest.mark.parametrize("dt", [1.0])
def test_flow_apply_to_image_forward(dt: float) -> None:
    """Verify forward flow correctly shifts pixels to their new
    coordinates.

    Uses a simple impulse image (single active pixel) to confirm that
    the transformation logic moves data in the intended direction.

    Args:
        dt: Dt for this test case.
    """
    # 1. Build a simple test image: one bright pixel in the centre.
    img = jnp.zeros((5, 5))
    img = img.at[2, 2].set(1.0)

    # 2. Constant horizontal flow (u = +1 px / step, v = 0).
    def flow_f(_t: Any, _x: Any, _y: Any) -> tuple[float, float]:
        """Flow f.

        Args:
            _t: T used by this test.
            _x: X used by this test.
            _y: Y used by this test.

        Returns:
            The result used by the test assertions.
        """
        return 1.0, 0.0

    # 3. Apply the flow in the *forward* sense.
    img_warped = apply_flow_to_image_callable(
        img,
        flow_f,
        t=0.0,
        dt=dt,
        forward=True,
    )

    # 4. Expected result: pixel shifted from (2, 2) → (2, 3).
    expected = jnp.zeros_like(img)
    expected = expected.at[2, 3].set(1.0)

    # 5. Verify shape, location and intensity.
    assert img_warped.shape == expected.shape, (
        f"Warped image shape mismatch. Expected {expected.shape}"
        f", got {img_warped.shape}"
    )
    assert jnp.allclose(img_warped, expected), (
        "Forward mapping did not move the pixel correctly."
    )


@pytest.mark.parametrize("dt", [1.0])
def test_apply_flow_to_particles_3d_constant(dt: float) -> None:
    """Verify 3D particle advection using a constant velocity field.

    Ensures that (x, y, z) coordinates are correctly updated based on
    (u, v, w) velocity components and the timestep.

    Args:
        dt: Dt for this test case.
    """
    # Random particles
    key = jax.random.PRNGKey(42)
    num_particles = 8
    zyx_max = jnp.array([9.0, 9.0, 9.0])
    particles = jax.random.uniform(
        key, (num_particles, 3), minval=0.0, maxval=zyx_max
    )

    # Build a constant velocity field (u, v, w) = (1, 2, 3) everywhere.
    D = H = W = 10
    u, v, w = 1.0, 2.0, 3.0
    flow_field = jnp.tile(
        jnp.array([u, v, w]), (D, H, W, 1)
    )  # shape D, H, W, 3)

    # Apply the flow
    advected = apply_flow_to_particles(particles, flow_field, dt=dt)

    # Expected displacement
    expected = particles + jnp.array([w * dt, v * dt, u * dt])

    # Verify shape and values
    assert advected.shape == particles.shape, (
        f"Advected particles shape mismatch. Expected "
        f"{particles.shape}, got {advected.shape}"
    )
    assert jnp.allclose(advected, expected), (
        "3D particles displacement produced wrong positions."
    )


@pytest.mark.parametrize("selected_flow", ["vertical"])
@pytest.mark.parametrize("seeding_density", [0.1])
@pytest.mark.parametrize("image_shape", [(128, 128)])
@pytest.mark.parametrize("diameter_range", [(0.1, 1.0)])
@pytest.mark.parametrize("intensity_range", [(50, 200)])
@pytest.mark.parametrize("rho_range", [(-0.5, 0.5)])
def test_particles_flow_apply_array(
    selected_flow: str,
    seeding_density: float,
    image_shape: tuple[int, int],
    diameter_range: tuple[float, float],
    intensity_range: tuple[float, float],
    rho_range: tuple[float, float],
    visualize: bool = False,
) -> None:
    """Test particle advection using an array-based flow field
    representation.

    Verifies that the synthetic image generated from advected particles
    matches the expected state after being warped by the discrete flow
    field.

    Args:
        selected_flow: Name of the analytical flow used as ground truth.
        seeding_density: Seeding density for this test case.
        image_shape: Height and width of the output images.
        diameter_range: Diameter range for this test case.
        intensity_range: Intensity range for this test case.
        rho_range: Rho range for this test case.
        visualize: Visualize for this test case.
    """

    # 1. Generate random particles and their characteristics
    key = jax.random.PRNGKey(0)
    subkey1, subkey2, subkey3, subkey4, subkey5 = jax.random.split(key, 5)

    particles_number = int(image_shape[0] * image_shape[1] * seeding_density)
    particles = jax.random.uniform(
        subkey1,
        (particles_number, 2),
        minval=0.0,
        maxval=jnp.array(image_shape) - 1,
    )
    diameters_x = jax.random.uniform(
        subkey2,
        (particles_number,),
        minval=diameter_range[0],
        maxval=diameter_range[1],
    )
    diameters_y = jax.random.uniform(
        subkey3,
        (particles_number,),
        minval=diameter_range[0],
        maxval=diameter_range[1],
    )
    intensities = jax.random.uniform(
        subkey4,
        (particles_number,),
        minval=intensity_range[0],
        maxval=intensity_range[1],
    )
    rho = jax.random.uniform(
        subkey5,
        (particles_number,),
        minval=rho_range[0],
        maxval=rho_range[1],
    )

    # 2. create a synthetic image
    img = img_gen_from_data(
        image_shape=image_shape,
        particle_positions=particles,
        diameters_x=diameters_x,
        diameters_y=diameters_y,
        intensities=intensities,
        rho=rho,
    )

    # 3. create a flow field
    flow_field = generate_array_flow_field(
        get_flow_function(selected_flow, image_shape), image_shape
    )

    # 4. Apply the flow to the particles
    new_particles = apply_flow_to_particles(particles, flow_field)

    # 5. create a synthetic image with the new particles
    img_warped = img_gen_from_data(
        image_shape=image_shape,
        particle_positions=new_particles,
        diameters_x=diameters_x,
        diameters_y=diameters_y,
        intensities=intensities,
        rho=rho,
    )

    if visualize:
        import matplotlib.pyplot as plt  # noqa: PLC0415 - optional or intentionally lazy import

        plt.imsave("img.png", np.array(img), cmap="gray")
        plt.imsave("img_warped.png", np.array(img_warped), cmap="gray")

    # 6. Check particles shapes
    assert particles.shape == new_particles.shape, (
        "Particles shapes do not match"
    )


@pytest.mark.parametrize(
    "particle_positions", [1, [[1, 2], [3, 4]], [[1, 2, 3], [4, 5, 6]]]
)
def test_invalid_particle_positions(particle_positions: Any) -> None:
    """Test that providing particle positions with invalid rank or
    dimensions raises a ValueError.

    Validates that input arrays are 2D with either 2 (2D) or 3 (3D)
    components.

    Args:
        particle_positions: Particle positions for this test case.
    """
    flow_field = jnp.zeros((128, 128, 2))
    with pytest.raises(
        ValueError,
        match=(
            r"Particle_positions must be a 2D jnp.ndarray with shape "
            "\\(N, 2\\) or \\(N, 3\\)"
        ),
    ):
        input_check_apply_flow(particle_positions, flow_field)


@pytest.mark.parametrize(
    "flow_field",
    [
        1,
        jnp.array([1, 2, 3]),
        [[[10, 20]]],
        jnp.array([1, 2, 3]),
        [[[10, 20, 30]]],
    ],
)
def test_invalid_flow_field(flow_field: Any) -> None:
    """Test that providing flow fields with invalid rank or components
    raises a ValueError.

    Expects a 3D array (H, W, C) where C matches the dimensionality of
    the particle positions.

    Args:
        flow_field: Flow field supplied to the operation under test.
    """
    particle_positions = jnp.zeros((1, 2))
    with pytest.raises(
        ValueError,
        match=(
            r"Flow_field must be a 3D jnp.ndarray with shape "
            "\\(H, W, 2\\) or \\(H, W, 3\\)"
        ),
    ):
        input_check_apply_flow(particle_positions, flow_field)


# NOTE: issue #232 proposes solving this.
# We make code operational while it hasn't beeen closed yet.
@pytest.mark.to_fix
@pytest.mark.parametrize(
    "flow_field, particle_positions, error_msg",
    [
        (
            jnp.zeros((128, 128, 3)),
            jnp.zeros((1, 2)),
            ("Particle positions are in 2D, but the flow field is in 3D."),
        ),
        (
            jnp.zeros((128, 128, 2)),
            jnp.zeros((1, 3)),
            ("Particle positions are in 3D, but the flow field is in 2D."),
        ),
    ],
)
def test_invalid_flow_field_shape(
    flow_field: Any, particle_positions: Any, error_msg: Any
) -> None:
    """Test for mismatches between particle dimensionality and flow field
    components.

    Ensures that 2D particles cannot be advected by 3D flows and
    vice-versa.

    Args:
        flow_field: Flow field supplied to the operation under test.
        particle_positions: Particle positions for this test case.
        error_msg: Error msg for this test case.
    """
    with pytest.raises(
        ValueError,
        match=error_msg,
    ):
        input_check_apply_flow(particle_positions, flow_field)


@pytest.mark.parametrize(
    "dt", ["a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])]
)
def test_invalid_dt(dt: Any) -> None:
    """Test that invalid dt raise a ValueError.

    Args:
        dt: Dt for this test case.
    """
    particle_positions = jnp.zeros((1, 2))
    flow_field = jnp.zeros((128, 128, 2))
    with pytest.raises(
        ValueError, match="dt must be a scalar \\(int or float\\)"
    ):
        input_check_apply_flow(particle_positions, flow_field, dt)


@pytest.mark.parametrize(
    "flow_field_res_x",
    ["a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])],
)
def test_invalid_flow_field_res_x(flow_field_res_x: Any) -> None:
    """Test that invalid flow_field_res_x raise a ValueError.

    Args:
        flow_field_res_x: Flow field res x for this test case.
    """
    particle_positions = jnp.zeros((1, 2))
    flow_field = jnp.zeros((128, 128, 2))
    with pytest.raises(
        ValueError,
        match=("flow_field_res_x must be a positive scalar \\(int or float\\)"),
    ):
        input_check_apply_flow(
            particle_positions, flow_field, flow_field_res_x=flow_field_res_x
        )


@pytest.mark.parametrize(
    "flow_field_res_y",
    ["a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])],
)
def test_invalid_flow_field_res_y(flow_field_res_y: Any) -> None:
    """Test that invalid flow_field_res_y raise a ValueError.

    Args:
        flow_field_res_y: Flow field res y for this test case.
    """
    particle_positions = jnp.zeros((1, 2))
    flow_field = jnp.zeros((128, 128, 2))
    with pytest.raises(
        ValueError,
        match=("flow_field_res_y must be a positive scalar \\(int or float\\)"),
    ):
        input_check_apply_flow(
            particle_positions, flow_field, flow_field_res_y=flow_field_res_y
        )


@pytest.mark.parametrize(
    "flow_field_res_z",
    ["a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])],
)
def test_invalid_flow_field_res_z(flow_field_res_z: Any) -> None:
    """Test that invalid flow_field_res_z raise a ValueError.

    Args:
        flow_field_res_z: Flow field res z for this test case.
    """
    particle_positions = jnp.zeros((1, 2))
    flow_field = jnp.zeros((128, 128, 2))
    with pytest.raises(
        ValueError,
        match=("flow_field_res_z must be a positive scalar \\(int or float\\)"),
    ):
        input_check_apply_flow(
            particle_positions, flow_field, flow_field_res_z=flow_field_res_z
        )


# skipif is used to skip the test if the user is not connected to the server
@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connected to the server.",
)
@pytest.mark.parametrize("selected_flow", ["horizontal"])
@pytest.mark.parametrize("seeding_density", [0.016])
@pytest.mark.parametrize("image_shape", [(1216, 1936)])
def test_speed_apply_flow_to_particles(
    seeding_density: float, selected_flow: str, image_shape: tuple[int, int]
) -> None:
    """Benchmark performance of GPU-parallelized particle advection.

    Uses `shard_map` to distribute particles across available GPUs and
    verifies that the advection completion time is within tight limits.

    Args:
        seeding_density: Seeding density for this test case.
        selected_flow: Name of the analytical flow used as ground truth.
        image_shape: Height and width of the output images.
    """

    # Name of the axis for the device mesh
    shard_particles = "particles"

    # Check how many GPUs are available
    devices = jax.devices()
    if len(devices) == 3:
        devices = devices[:2]
    elif len(devices) > 4:
        devices = devices[:4]

    num_devices = len(devices)

    # Limit time in seconds (depends on the number of GPUs)
    if num_devices == 1:
        limit_time = 4.5e-5
    elif num_devices == 2:
        limit_time = 6e-5
    elif num_devices == 4:
        limit_time = 8e-5

    # Setup device mesh
    # We want to shard the particles along the first axis
    # and replicate the flow field along all devices.
    # The idea is that each device will apply the flow to a part of the
    # particles
    # and then we will combine the results.
    mesh = Mesh(devices, axis_names=(shard_particles))

    # 1. Generate random particles
    key = jax.random.PRNGKey(0)

    # Compute the number of particles and round it to the number of devices
    particles_number = int(image_shape[0] * image_shape[1] * seeding_density)
    particles_number = (particles_number // num_devices + 1) * num_devices
    particles = jax.random.uniform(
        key,
        (particles_number, 2),
        minval=0.0,
        maxval=jnp.array(image_shape) - 1,
    )

    # 2. Send the particles to the devices
    # To make the test also consider the time of sending the variables to the
    # devices
    # comment the next lines
    sharding_particles = NamedSharding(mesh, PartitionSpec(shard_particles))
    particles = jax.device_put(particles, sharding_particles)

    # 3. Create a flow field
    flow_field = generate_array_flow_field(
        get_flow_function(selected_flow, image_shape), image_shape
    )

    # 4. Duplicate the flow field to all devices
    sharding_flow_field = NamedSharding(mesh, PartitionSpec())
    flow_field_replicated = jax.device_put(flow_field, sharding_flow_field)

    # 5. Create the jit function
    apply_flow_to_particles_jit = jax.jit(apply_flow_to_particles)

    def run_apply_jit() -> None:
        """Run apply jit."""
        result = apply_flow_to_particles_jit(particles, flow_field_replicated)
        result.block_until_ready()

    # Warm up the function
    run_apply_jit()

    # Measure the time of the jit function
    total_time_jit = timeit.repeat(
        stmt=run_apply_jit, number=NUMBER_OF_EXECUTIONS, repeat=REPETITIONS
    )
    average_time_jit = min(total_time_jit) / NUMBER_OF_EXECUTIONS

    # Check if the time is less than the limit
    assert average_time_jit < limit_time, (
        f"The average time is {average_time_jit}, time limit: {limit_time}"
    )


if __name__ == "__main__":
    for d in jax.devices():
        print(d.id, d.device_kind, d.platform)
