import timeit

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec

# Import existing modules
from src.sym.example_flows import get_flow_function
from src.sym.processing import generate_images_from_flow, input_check_gen_img_from_flow
from src.utils import generate_array_flow_field, load_configuration

config = load_configuration("config/timeit.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["NUMBER_OF_EXECUTIONS"]


@pytest.mark.parametrize(
    "key",
    [
        None,
        42,
        "invalid_key",
        jnp.array([1, 2]),
        jnp.array([1.0, 2.0]),
        jnp.array([1, 2, 3]),
    ],
)
def test_invalid_key(key):
    """Test that invalid PRNG keys raise a TypeError."""
    flow_field = jnp.zeros((128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(
        ValueError,
        match="key must be a jax.array with shape \\(2,\\) and dtype jnp.uint32.",
    ):
        input_check_gen_img_from_flow(
            key, flow_field=flow_field, image_shape=image_shape
        )


@pytest.mark.parametrize("flow_field", [1, jnp.array([1, 2, 3]), [[[10, 20]]]])
def test_invalid_flow_field(flow_field):
    """Test that invalid flow_field raise a ValueError."""
    key = jax.random.PRNGKey(0)
    image_shape = (128, 128)
    with pytest.raises(
        ValueError, match="Flow_field must be a 3D jnp.ndarray with shape \\(H, W, 2\\)"
    ):
        input_check_gen_img_from_flow(
            key, flow_field=flow_field, image_shape=image_shape
        )


@pytest.mark.parametrize(
    "image_shape", [(-1, 128), (128, -1), (0, 128), (128, 0), (128.2, 128.2)]
)
def test_invalid_image_shape(image_shape):
    """Test that invalid image shapes raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((128, 128, 2))
    with pytest.raises(
        ValueError, match="image_shape must be a tuple of two positive integers."
    ):
        input_check_gen_img_from_flow(
            key, flow_field=flow_field, image_shape=image_shape
        )


@pytest.mark.parametrize(
    "position_bounds", [(-1, 128), (128, -1), (0, 128), (128, 0), (128.2, 128.2)]
)
def test_invalid_position_bounds(position_bounds):
    """Test that invalid position_bounds raises a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((128, 128, 2))
    image_shape = (128, 128)
    img_offset = (0, 0)
    with pytest.raises(
        ValueError, match="position_bounds must be a tuple of two positive integers."
    ):
        input_check_gen_img_from_flow(
            key,
            flow_field=flow_field,
            position_bounds=position_bounds,
            image_shape=image_shape,
            img_offset=img_offset,
        )


@pytest.mark.parametrize(
    "img_offset", [(-1, 0), (0, -1), (128.5, 0), (0, 128.5), (1, 2, 3)]
)
def test_invalid_img_offset(img_offset):
    """Test that invalid img_offset raises a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((128, 128, 2))
    position_bounds = (256, 256)
    image_shape = (128, 128)
    with pytest.raises(
        ValueError, match="img_offset must be a tuple of two non-negative integers."
    ):
        input_check_gen_img_from_flow(
            key,
            flow_field=flow_field,
            position_bounds=position_bounds,
            image_shape=image_shape,
            img_offset=img_offset,
        )


@pytest.mark.parametrize("num_particles", [-1, 0, 1.5, 2.5])
def test_invalid_num_particles(num_particles):
    """Test that invalid num_particles raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(ValueError, match="num_particles must be a positive integer."):
        input_check_gen_img_from_flow(
            key,
            flow_field=flow_field,
            image_shape=image_shape,
            num_particles=num_particles,
        )


@pytest.mark.parametrize("num_images", [-1, 0, 1.5, 2.5])
def test_invalid_num_images(num_images):
    """Test that invalid num_images raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(ValueError, match="num_images must be a positive integer."):
        input_check_gen_img_from_flow(
            key, flow_field=flow_field, image_shape=image_shape, num_images=num_images
        )


@pytest.mark.parametrize("p_hide_img1", [-0.1, 1.1, 1.5, 2.5])
def test_invalid_p_hide_img1(p_hide_img1):
    """Test that invalid p_hide_img1 raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(ValueError, match="p_hide_img1 must be between 0 and 1."):
        input_check_gen_img_from_flow(
            key, flow_field=flow_field, image_shape=image_shape, p_hide_img1=p_hide_img1
        )


@pytest.mark.parametrize("p_hide_img2", [-0.1, 1.1, 1.5, 2.5])
def test_invalid_p_hide_img2(p_hide_img2):
    """Test that invalid p_hide_img2 raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(ValueError, match="p_hide_img2 must be between 0 and 1."):
        input_check_gen_img_from_flow(
            key, flow_field=flow_field, image_shape=image_shape, p_hide_img2=p_hide_img2
        )


@pytest.mark.parametrize("diameter_range", [(0, 1), (1, 0), (-1, 1), (1, -1)])
def test_invalid_diameter_range(diameter_range):
    """Test that invalid diameter ranges raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(
        ValueError, match="diameter_range must be a tuple of two positive floats."
    ):
        input_check_gen_img_from_flow(
            key,
            diameter_range=diameter_range,
            flow_field=flow_field,
            image_shape=image_shape,
        )


@pytest.mark.parametrize("intensity_range", [(-1, 200), (50, -1), (1, 1, 1)])
def test_invalid_intensity_range(intensity_range):
    """Test that invalid intensity ranges raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(ValueError):
        input_check_gen_img_from_flow(
            key,
            flow_field=flow_field,
            image_shape=image_shape,
            intensity_range=intensity_range,
        )


@pytest.mark.parametrize("rho_range", [(-1.1, 1), (1, -1.1), (-1, 1.1), (1, 1, 1)])
def test_invalid_rho_range(rho_range):
    """Test that invalid rho ranges raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(
        ValueError, match="rho_range must be a tuple of two floats between -1 and 1."
    ):
        input_check_gen_img_from_flow(
            key, flow_field=flow_field, image_shape=image_shape, rho_range=rho_range
        )


@pytest.mark.parametrize(
    "dt", ["invalid_dt", jnp.array([1]), jnp.array([1.0, 2.0]), jnp.array([1, 2, 3])]
)
def test_invalid_dt(dt):
    """Test that invalid dt raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(ValueError, match="dt must be a scalar \\(int or float\\)"):
        input_check_gen_img_from_flow(
            key, flow_field=flow_field, image_shape=image_shape, dt=dt
        )


@pytest.mark.parametrize("selected_flow", ["horizontal"])
def test_generate_images_from_flow(selected_flow, visualize=True):
    """Test that we can apply a flow field as a jax array to random particles."""

    # 1. setup the image parameters
    key = jax.random.PRNGKey(0)
    position_bounds = (128, 128)
    image_shape = (128, 128)
    num_particles = 1
    p_hide_img1 = 0.01
    p_hide_img2 = 0.5
    diameter_range = (2, 4)
    intensity_range = (50, 250)
    rho_range = (-0.2, 0.2)
    dt = 5.0

    # 2. create a flow field
    flow_field = generate_array_flow_field(
        get_flow_function(selected_flow, image_shape), image_shape
    )

    # 3. apply the flow field to the particles
    img, img_warped = generate_images_from_flow(
        key,
        flow_field,
        position_bounds=position_bounds,
        image_shape=image_shape,
        num_particles=num_particles,
        num_images=1,
        p_hide_img1=p_hide_img1,
        p_hide_img2=p_hide_img2,
        diameter_range=diameter_range,
        intensity_range=intensity_range,
        rho_range=rho_range,
        dt=dt,
    )

    # 4. fix the shape of the images
    img = jnp.squeeze(img)
    img_warped = jnp.squeeze(img_warped)

    if visualize:
        import matplotlib.pyplot as plt
        import numpy as np

        plt.imsave("img.png", np.array(img), cmap="gray")
        plt.imsave("img_warped.png", np.array(img_warped), cmap="gray")


# skipif is used to skip the test if the user is not connected to the server
@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize("selected_flow", ["horizontal"])
@pytest.mark.parametrize("particles_number", [40000])
@pytest.mark.parametrize("num_images", [100])
@pytest.mark.parametrize("image_shape", [(1216, 1936)])
@pytest.mark.parametrize("big_image_shape", [(1536, 2048)])
def test_speed_generate_images_from_flow(
    particles_number, selected_flow, num_images, image_shape, big_image_shape
):
    """Test that generate_images_from_flow is faster than a limit time."""

    # Name of the axis for the device mesh
    shard_keys = "keys"

    # Check how many GPUs are available
    num_devices = len(jax.devices())

    # Limit time in seconds (depends on the number of GPUs)
    if num_devices == 1:
        limit_time = 2e-2
    elif num_devices == 2:
        limit_time = 1e-2
    elif num_devices == 4:
        limit_time = 5e-3

    # Setup device mesh
    # We want to shard a key to each device
    # and duplicate the flow field.
    # The idea is that each device will generate a num_images images
    # and then stack it with the images generated by the other GPUs.
    devices = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(devices, axis_names=("dp",))

    # dimension of the image
    image_shape = (1216, 1936)
    position_bounds = (1536, 2048)

    # 1. Generate key
    key = jax.random.PRNGKey(0)

    # 2. create a flow field
    flow_field = generate_array_flow_field(
        get_flow_function(selected_flow, position_bounds), position_bounds
    )

    # 3. Setup the random keys
    keys = jax.random.split(key, num_devices)
    keys = jnp.stack(keys)

    # 4. Create the jit function
    jit_generate_images = jax.jit(
        shard_map(
            lambda key, flow: generate_images_from_flow(
                key,
                flow,
                position_bounds=position_bounds,
                image_shape=image_shape,
                num_particles=particles_number,
                num_images=num_images,
            ),
            mesh=mesh,
            in_specs=(PartitionSpec(shard_keys), PartitionSpec()),
            out_specs=(PartitionSpec(shard_keys), PartitionSpec(shard_keys)),
        )
    )

    def run_generate_jit():
        imgs1, imgs2 = jit_generate_images(keys, flow_field)
        imgs1.block_until_ready()
        imgs2.block_until_ready()
        return imgs1, imgs2

    # Warm up the function
    run_generate_jit()

    # Measure the time of the jit function
    # We divide by the number of devices because shard_map
    # will return Number of devices results, like this we keep the number of
    # images generated the same as the number of devices changes
    total_time_jit = timeit.repeat(
        stmt=run_generate_jit,
        number=NUMBER_OF_EXECUTIONS // num_devices,
        repeat=REPETITIONS,
    )

    # Average time
    average_time_jit = min(total_time_jit) / NUMBER_OF_EXECUTIONS

    # Check if the time is less than the limit
    assert (
        average_time_jit < limit_time
    ), f"The average time is {average_time_jit}, time limit: {limit_time}"
