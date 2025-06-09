import timeit
from jax import profiler
import csv
import jax
import jax.numpy as jnp
import pytest
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from synthpix.data_generate import (
    generate_images_from_flow,
    input_check_gen_img_from_flow,
)

# Import existing modules
from synthpix.example_flows import get_flow_function
from synthpix.utils import generate_array_flow_field, load_configuration

config = load_configuration("config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_DATA_GEN"]


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
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(
        ValueError,
        match="key must be a jax.array with shape \\(2,\\) and dtype jnp.uint32.",
    ):
        input_check_gen_img_from_flow(
            key, flow_field=flow_field, image_shape=image_shape
        )


@pytest.mark.parametrize("flow_field", [1, jnp.array([1, 2, 3]), [[[[10, 20]]]]])
def test_invalid_flow_field(flow_field):
    """Test that invalid flow_field raise a ValueError."""
    key = jax.random.PRNGKey(0)
    image_shape = (128, 128)
    with pytest.raises(
        ValueError,
        match="Flow_field must be a 4D jnp.ndarray with shape \\(N, H, W, 2\\).",
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
    flow_field = jnp.zeros((1, 128, 128, 2))
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
    flow_field = jnp.zeros((1, 128, 128, 2))
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
def test_invalid_seeding_density_range(seeding_density_range, expected_message):
    """Test that invalid seeding_density_range raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(ValueError, match=expected_message):
        input_check_gen_img_from_flow(
            key,
            flow_field=flow_field,
            image_shape=image_shape,
            seeding_density_range=seeding_density_range,
        )


@pytest.mark.parametrize("num_images", [-1, 0, 1.5, 2.5])
def test_invalid_num_images(num_images):
    """Test that invalid num_images raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(ValueError, match="num_images must be a positive integer."):
        input_check_gen_img_from_flow(
            key, flow_field=flow_field, image_shape=image_shape, num_images=num_images
        )


@pytest.mark.parametrize(
    "img_offset", [(-1, 0), (0, -1), (128.5, 0), (0, 128.5), (1, 2, 3)]
)
def test_invalid_img_offset(img_offset):
    """Test that invalid img_offset raises a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((1, 128, 128, 2))
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


@pytest.mark.parametrize("p_hide_img1", [-0.1, 1.1, 1.5, 2.5])
def test_invalid_p_hide_img1(p_hide_img1):
    """Test that invalid p_hide_img1 raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(ValueError, match="p_hide_img1 must be between 0 and 1."):
        input_check_gen_img_from_flow(
            key, flow_field=flow_field, image_shape=image_shape, p_hide_img1=p_hide_img1
        )


@pytest.mark.parametrize("p_hide_img2", [-0.1, 1.1, 1.5, 2.5])
def test_invalid_p_hide_img2(p_hide_img2):
    """Test that invalid p_hide_img2 raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(ValueError, match="p_hide_img2 must be between 0 and 1."):
        input_check_gen_img_from_flow(
            key, flow_field=flow_field, image_shape=image_shape, p_hide_img2=p_hide_img2
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
def test_invalid_diameter_range(diameter_range, expected_message):
    """Test that invalid diameter ranges raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(ValueError, match=expected_message):
        input_check_gen_img_from_flow(
            key,
            diameter_range=diameter_range,
            flow_field=flow_field,
            image_shape=image_shape,
        )


@pytest.mark.parametrize(
    "intensity_range, expected_message",
    [
        ((-1.0, 1.0), "intensity_range must be a tuple of two positive floats."),
        ((0.0, -1.0), "intensity_range must be a tuple of two positive floats."),
        ((-0.5, -0.5), "intensity_range must be a tuple of two positive floats."),
        ((1.0, 0.5), "intensity_range must be in the form \\(min, max\\)."),
        ((0.5, 0.1), "intensity_range must be in the form \\(min, max\\)."),
    ],
)
def test_invalid_intensity_range(intensity_range, expected_message):
    """Test that invalid intensity ranges raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(ValueError, match=expected_message):
        input_check_gen_img_from_flow(
            key,
            flow_field=flow_field,
            image_shape=image_shape,
            intensity_range=intensity_range,
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
def test_invalid_rho_range(rho_range, expected_message):
    """Test that invalid rho ranges raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(ValueError, match=expected_message):
        input_check_gen_img_from_flow(
            key, flow_field=flow_field, image_shape=image_shape, rho_range=rho_range
        )


@pytest.mark.parametrize(
    "dt", ["invalid_dt", jnp.array([1]), jnp.array([1.0, 2.0]), jnp.array([1, 2, 3])]
)
def test_invalid_dt(dt):
    """Test that invalid dt raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(ValueError, match="dt must be a scalar \\(int or float\\)"):
        input_check_gen_img_from_flow(
            key, flow_field=flow_field, image_shape=image_shape, dt=dt
        )


@pytest.mark.parametrize(
    "flow_field_res_x",
    ["a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])],
)
def test_invalid_flow_field_res_x(flow_field_res_x):
    """Test that invalid flow_field_res_x raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(
        ValueError,
        match="flow_field_res_x must be a positive scalar \\(int or float\\)",
    ):
        input_check_gen_img_from_flow(
            key,
            flow_field=flow_field,
            image_shape=image_shape,
            flow_field_res_x=flow_field_res_x,
        )


@pytest.mark.parametrize(
    "flow_field_res_y",
    ["a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])],
)
def test_invalid_flow_field_res_y(flow_field_res_y):
    """Test that invalid flow_field_res_y raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(
        ValueError,
        match="flow_field_res_y must be a positive scalar \\(int or float\\)",
    ):
        input_check_gen_img_from_flow(
            key,
            flow_field=flow_field,
            image_shape=image_shape,
            flow_field_res_y=flow_field_res_y,
        )


@pytest.mark.parametrize(
    "noise_level", [-1, "a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])]
)
def test_invalid_noise_level(noise_level):
    """Test that invalid noise_level raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(
        ValueError,
        match="noise_level must be a non-negative number.",
    ):
        input_check_gen_img_from_flow(
            key,
            flow_field=flow_field,
            image_shape=image_shape,
            noise_level=noise_level,
        )


@pytest.mark.parametrize(
    "diameter_var", ["a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])]
)
def test_invalid_diameter_var(diameter_var):
    """Test that invalid diameter_var raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(
        ValueError,
        match="diameter_var must be a non-negative number.",
    ):
        input_check_gen_img_from_flow(
            key,
            flow_field=flow_field,
            image_shape=image_shape,
            diameter_var=diameter_var,
        )


@pytest.mark.parametrize(
    "intensity_var", ["a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])]
)
def test_invalid_intensity_var(intensity_var):
    """Test that invalid intensity_var raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(
        ValueError,
        match="intensity_var must be a non-negative number.",
    ):
        input_check_gen_img_from_flow(
            key,
            flow_field=flow_field,
            image_shape=image_shape,
            intensity_var=intensity_var,
        )


@pytest.mark.parametrize(
    "rho_var", ["a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])]
)
def test_invalid_rho_var(rho_var):
    """Test that invalid rho_var raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(
        ValueError,
        match="rho_var must be a non-negative number.",
    ):
        input_check_gen_img_from_flow(
            key,
            flow_field=flow_field,
            image_shape=image_shape,
            rho_var=rho_var,
        )


@pytest.mark.parametrize(
    "image_shape, img_offset, position_bounds, error_message",
    [
        (
            (128, 128),
            (0, 0),
            (64, 128),
            "The height of the position_bounds must be greater "
            "than the height of the image plus the offset.",
        ),
        (
            (128, 128),
            (1, 0),
            (128, 128),
            "The height of the position_bounds must be greater "
            "than the height of the image plus the offset.",
        ),
        (
            (128, 128),
            (0, 0),
            (128, 64),
            "The width of the position_bounds must be greater "
            "than the width of the image plus the offset.",
        ),
        (
            (128, 128),
            (0, 1),
            (128, 128),
            "The width of the position_bounds must be greater "
            "than the width of the image plus the offset.",
        ),
    ],
)
def test_incoherent_image_shape_and_position_bounds(
    image_shape, img_offset, position_bounds, error_message
):
    """Test that incoherent image_shape and position_bounds raise a ValueError."""
    key = jax.random.PRNGKey(0)
    flow_field = jnp.zeros((1, 128, 128, 2))
    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        input_check_gen_img_from_flow(
            key,
            flow_field=flow_field,
            position_bounds=position_bounds,
            image_shape=image_shape,
            img_offset=img_offset,
        )


def test_generate_images_from_flow(visualize=True):
    """Test that we can generate images from a flow field."""

    # 1. setup the image parameters
    key = jax.random.PRNGKey(0)
    selected_flow = "horizontal"
    position_bounds = (128, 128)
    image_shape = (128, 128)
    seeding_density_range = (0.001, 0.01)
    img_offset = (0, 0)
    p_hide_img1 = 0.0
    p_hide_img2 = 0.0
    diameter_range = (1, 2)
    diameter_var = 0
    intensity_range = (50, 250)
    intensity_var = 0
    rho_range = (-0.2, 0.2)  # rho cannot be -1 or 1
    rho_var = 0
    dt = 0.0
    noise_level = 0.0

    # 2. create a flow field
    flow_field = generate_array_flow_field(
        get_flow_function(selected_flow, image_shape), image_shape
    )
    flow_field = jnp.expand_dims(flow_field, axis=0)

    # 3. apply the flow field to the particles
    img, img_warped, _ = generate_images_from_flow(
        key,
        flow_field,
        position_bounds=position_bounds,
        image_shape=image_shape,
        seeding_density_range=seeding_density_range,
        num_images=1,
        img_offset=img_offset,
        p_hide_img1=p_hide_img1,
        p_hide_img2=p_hide_img2,
        diameter_range=diameter_range,
        diameter_var=diameter_var,
        intensity_range=intensity_range,
        intensity_var=intensity_var,
        rho_range=rho_range,
        rho_var=rho_var,
        dt=dt,
        noise_level=noise_level,
    )

    # 4. fix the shape of the images
    img = jnp.squeeze(img)
    img_warped = jnp.squeeze(img_warped)

    if visualize:
        import matplotlib.pyplot as plt
        import numpy as np

        plt.imsave("img.png", np.array(img), cmap="gray")
        plt.imsave("img_warped.png", np.array(img_warped), cmap="gray")

    # 5. check the shape of the images
    assert img.shape == image_shape
    assert img_warped.shape == image_shape


# skipif is used to skip the test if the user is not connected to the server
# @pytest.mark.skipif(
#     not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
#     reason="user not connect to the server.",
# )
@pytest.mark.parametrize("selected_flow", ["horizontal"])
@pytest.mark.parametrize("seeding_density_range", [(0.001, 0.06)])
@pytest.mark.parametrize("num_images", [64])
@pytest.mark.parametrize("image_shape", [(512, 512)])
# @pytest.mark.parametrize("position_bounds", [(1536, 2048)])
@pytest.mark.parametrize("img_offset", [(10, 10)])
@pytest.mark.parametrize("num_flow_fields", [1])
def test_speed_generate_images_from_flow(
    selected_flow,
    seeding_density_range,
    num_images,
    image_shape,
    # position_bounds,
    img_offset,
    num_flow_fields,
):
    """Test that generate_images_from_flow is faster than a limit time."""

    # Set the position bounds to 20 pixels larger than the image shape
    position_bounds = (image_shape[0] + 20, image_shape[1] + 20)

    NUMBER_OF_EXECUTIONS = 10000

    # Name of the axis for the device mesh
    shard_fields = "fields"

    # Check how many GPUs are available
    num_devices = len(jax.devices())

    # Limit time in seconds (depends on the number of GPUs)
    if num_devices == 1:
        limit_time = 0e-2
    elif num_devices == 2:
        limit_time = 0e-3
    elif num_devices == 4:
        limit_time = 0e-3

    # Setup device mesh
    # We want to shard a key to each device
    # and give different flow fields to each device.
    # The idea is that each device will generate a num_images images
    # and then stack it with the images generated by the other GPUs.
    devices = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(devices, axis_names=(shard_fields))

    # 1. Generate key
    key = jax.random.PRNGKey(0)

    # 2. create a flow field
    flow_field = generate_array_flow_field(
        get_flow_function(selected_flow, position_bounds), position_bounds
    )
    flow_field = jnp.expand_dims(flow_field, axis=0)
    flow_field = jnp.repeat(flow_field, num_flow_fields, axis=0)

    # 3. Shard the flow field
    flow_field_sharded = jax.device_put(
        flow_field, NamedSharding(mesh, PartitionSpec(shard_fields))
    )
    jax.block_until_ready(flow_field_sharded)

    # 4. Setup the random keys
    keys = jax.random.split(key, num_devices)
    keys = jnp.stack(keys)
    keys_sharded = jax.device_put(
        keys, NamedSharding(mesh, PartitionSpec(shard_fields))
    )
    jax.block_until_ready(keys_sharded)

    # 5. Create the jit function
    jit_generate_images = jax.jit(
        shard_map(
            lambda key, flow: generate_images_from_flow(
                key=key,
                flow_field=flow,
                position_bounds=position_bounds,
                image_shape=image_shape,
                img_offset=img_offset,
                seeding_density_range=seeding_density_range,
                num_images=num_images,
                p_hide_img1=0.0,
                p_hide_img2=0.0,
                diameter_range=(1, 2),
                diameter_var=0,
                intensity_range=(80, 100),
                intensity_var=0,
                noise_level=0,
                rho_range=(-0.01, 0.01),
                rho_var=0,
            ),
            mesh=mesh,
            in_specs=(PartitionSpec(shard_fields), PartitionSpec(shard_fields)),
            out_specs=(
                PartitionSpec(shard_fields),
                PartitionSpec(shard_fields),
                PartitionSpec(shard_fields),
            ),
        )
    )

    def run_generate_jit():
        imgs1, imgs2, seeding_densities = jit_generate_images(keys_sharded, flow_field_sharded)
        imgs1.block_until_ready()
        imgs2.block_until_ready()
        seeding_densities.block_until_ready()

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
    average_time_jit = jnp.mean(jnp.array(total_time_jit))

    # Check if the time is less than the limit
    assert (
        average_time_jit < limit_time
    ), f"The average time is {average_time_jit}, time limit: {limit_time}"



import numpy as np



# def write_speed_stats_to_csv(filename, seeding_densities, timings_per_density):
#     """
#     Write a CSV with columns: seeding_density, Q1, Q3, Mean, Min, Max, StdDev

#     timings_per_density: list of 1D arrays, each the timings (in seconds) for a given seeding density
#     """
#     with open(filename, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["seeding_density", "Q1", "Q3", "Mean", "Min", "Max", "StdDev"])
#         for density, timings in zip(seeding_densities, timings_per_density):
#             timings = np.asarray(timings)
#             q1 = np.percentile(timings, 25)
#             q3 = np.percentile(timings, 75)
#             mean = np.mean(timings)
#             min_ = np.min(timings)
#             max_ = np.max(timings)
#             std = np.std(timings)
#             writer.writerow([density, q1, q3, mean, min_, max_, std])

# @pytest.mark.skipif(
#     not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
#     reason="user not connected to the server.",
# )
# def test_speed_generate_images_all_seeding_densities():
#     # ---- SETTINGS ----
#     selected_flow = "horizontal"
#     all_seeding_densities = [0.1, 0.01, 0.001, 0.0001]
#     num_images = 100
#     image_shape = (1216, 1936)
#     img_offset = (10, 10)
#     num_flow_fields = 100
#     NUMBER_OF_EXECUTIONS = 100   # Or whatever you use
#     REPETITIONS = 100             # Or whatever you use

#     # ---- DEVICE AND MESH SETUP ----
#     shard_fields = "fields"
#     num_devices = len(jax.devices())

#     # Set a fake time limit (for completeness; not needed for CSV)
#     if num_devices == 1:
#         limit_time = 0e-2
#     elif num_devices == 2:
#         limit_time = 0e-3
#     elif num_devices == 4:
#         limit_time = 0e-3

#     devices = mesh_utils.create_device_mesh((num_devices,))
#     mesh = Mesh(devices, axis_names=(shard_fields,))

#     position_bounds = (image_shape[0] + 20, image_shape[1] + 20)
#     key = jax.random.PRNGKey(0)

#     timings_per_density = []

#     for density in all_seeding_densities:
#         seeding_density_range = (density, density)

#         # 1. Create flow field
#         flow_field = generate_array_flow_field(
#             get_flow_function(selected_flow, position_bounds), position_bounds
#         )
#         flow_field = jnp.expand_dims(flow_field, axis=0)
#         flow_field = jnp.repeat(flow_field, num_flow_fields, axis=0)
#         flow_field_sharded = jax.device_put(
#             flow_field, NamedSharding(mesh, PartitionSpec(shard_fields))
#         )

#         # 2. Setup random keys
#         keys = jax.random.split(key, num_devices)
#         keys = jnp.stack(keys)
#         keys_sharded = jax.device_put(
#             keys, NamedSharding(mesh, PartitionSpec(shard_fields))
#         )
#         jax.block_until_ready(keys_sharded)
#         jax.block_until_ready(flow_field_sharded)

#         # 3. Prepare the jit function
#         jit_generate_images = jax.jit(
#             shard_map(
#                 lambda key, flow: generate_images_from_flow(
#                     key=key,
#                     flow_field=flow,
#                     position_bounds=position_bounds,
#                     image_shape=image_shape,
#                     img_offset=img_offset,
#                     seeding_density_range=seeding_density_range,
#                     num_images=num_images,
#                     p_hide_img1=0.01,
#                     p_hide_img2=0.01,
#                     diameter_range=(1, 2),
#                     diameter_var=1,
#                     intensity_range=(100, 200),
#                     intensity_var=1,
#                     noise_level=1,
#                     rho_range=(-0.2, 0.2),
#                     rho_var=1,
#                 ),
#                 mesh=mesh,
#                 in_specs=(PartitionSpec(shard_fields), PartitionSpec(shard_fields)),
#                 out_specs=(
#                     PartitionSpec(shard_fields),
#                     PartitionSpec(shard_fields),
#                     PartitionSpec(shard_fields),
#                 ),
#             )
#         )

#         def run_generate_jit():
#             imgs1, imgs2, seeding_densities = jit_generate_images(keys_sharded, flow_field_sharded)
#             # imgs1.block_until_ready()
#             # imgs2.block_until_ready()
#             # seeding_densities.block_until_ready()

#         # Warm up
#         run_generate_jit()
#         # Timing
#         total_time_jit = timeit.repeat(
#             stmt=run_generate_jit,
#             number=NUMBER_OF_EXECUTIONS // num_devices,
#             repeat=REPETITIONS,
#         )
#         timings_per_density.append(total_time_jit)

#         # Optionally: you can assert here if you still want
#         # average_time_jit = min(total_time_jit)
#         # assert average_time_jit < limit_time

#     # --- WRITE TO CSV ---
#     # output_csv = "speed_seeding_density_results_4_GPUs.csv"
#     output_csv = "speed_seeding_density_results_4_GPUs_no_block_until_ready.csv"
#     write_speed_stats_to_csv(output_csv, all_seeding_densities, timings_per_density)
#     print(f"Wrote timings to {output_csv}")





# def write_speed_stats_to_csv(filename, image_shapes, timings_per_shape):
#     """
#     Write a CSV with columns: image_shape, Q1, Q3, Mean, Min, Max, StdDev
#     timings_per_shape: list of 1D arrays, each the timings (in seconds) for a given image shape
#     """
#     with open(filename, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["image_shape", "Q1", "Q3", "Mean", "Min", "Max", "StdDev"])
#         for shape, timings in zip(image_shapes, timings_per_shape):
#             timings = np.asarray(timings)
#             q1 = np.percentile(timings, 25)
#             q3 = np.percentile(timings, 75)
#             mean = np.mean(timings)
#             min_ = np.min(timings)
#             max_ = np.max(timings)
#             std = np.std(timings)
#             writer.writerow([f"{shape[0]}x{shape[1]}", q1, q3, mean, min_, max_, std])

# @pytest.mark.skipif(
#     not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
#     reason="user not connected to the server.",
# )
# def test_speed_generate_images_all_image_sizes():
#     # ---- SETTINGS ----
#     selected_flow = "horizontal"
#     seeding_density_range = (0.016, 0.016)
#     num_images = 100
#     img_offset = (10, 10)
#     num_flow_fields = 100
#     NUMBER_OF_EXECUTIONS = 100
#     REPETITIONS = 100

#     # Image shapes: powers of 2 from 32 to 2048
#     powers_of_two = [2 ** i for i in range(5, 12)]  # 32 to 2048
#     image_shapes = [(s, s) for s in powers_of_two]

#     # ---- DEVICE AND MESH SETUP ----
#     shard_fields = "fields"
#     num_devices = len(jax.devices())

#     devices = mesh_utils.create_device_mesh((num_devices,))
#     mesh = Mesh(devices, axis_names=(shard_fields,))

#     key = jax.random.PRNGKey(0)
#     timings_per_shape = []

#     for image_shape in image_shapes:
#         position_bounds = (image_shape[0] + 20, image_shape[1] + 20)

#         # 1. Create flow field
#         flow_field = generate_array_flow_field(
#             get_flow_function(selected_flow, position_bounds), position_bounds
#         )
#         flow_field = jnp.expand_dims(flow_field, axis=0)
#         flow_field = jnp.repeat(flow_field, num_flow_fields, axis=0)
#         flow_field_sharded = jax.device_put(
#             flow_field, NamedSharding(mesh, PartitionSpec(shard_fields))
#         )

#         # 2. Setup random keys
#         keys = jax.random.split(key, num_devices)
#         keys = jnp.stack(keys)
#         keys_sharded = jax.device_put(
#             keys, NamedSharding(mesh, PartitionSpec(shard_fields))
#         )
#         jax.block_until_ready(keys_sharded)
#         jax.block_until_ready(flow_field_sharded)

#         # 3. Prepare the jit function
#         jit_generate_images = jax.jit(
#             shard_map(
#                 lambda key, flow: generate_images_from_flow(
#                     key=key,
#                     flow_field=flow,
#                     position_bounds=position_bounds,
#                     image_shape=image_shape,
#                     img_offset=img_offset,
#                     seeding_density_range=seeding_density_range,
#                     num_images=num_images,
#                     p_hide_img1=0.01,
#                     p_hide_img2=0.01,
#                     diameter_range=(1, 2),
#                     diameter_var=1,
    #                 intensity_range=(100, 200),
    #                 intensity_var=1,
    #                 noise_level=1,
    #                 rho_range=(-0.2, 0.2),
    #                 rho_var=1,
    #             ),
    #             mesh=mesh,
    #             in_specs=(PartitionSpec(shard_fields), PartitionSpec(shard_fields)),
    #             out_specs=(
    #                 PartitionSpec(shard_fields),
    #                 PartitionSpec(shard_fields),
    #                 PartitionSpec(shard_fields),
    #             ),
    #         )
    #     )

    #     def run_generate_jit():
    #         imgs1, imgs2, seeding_densities = jit_generate_images(keys_sharded, flow_field_sharded)
    #         imgs1.block_until_ready()
    #         imgs2.block_until_ready()
    #         seeding_densities.block_until_ready()

    #     # Warm up
    #     run_generate_jit()

    #     # profiler.start_trace(f"profiler_1GPU/generate_images_{image_shape[0]}x{image_shape[1]}")
    #     # Timing
    #     total_time_jit = timeit.repeat(
    #         stmt=run_generate_jit,
    #         number=NUMBER_OF_EXECUTIONS // num_devices,
    #         repeat=REPETITIONS,
    #     )
    #     timings_per_shape.append(total_time_jit)
    #     # profiler.stop_trace()

    # # --- WRITE TO CSV ---
#     output_csv = "speed_image_size_results_4_GPUs.csv"
#     # output_csv = "speed_image_size_results_4_GPUs_no_block_until_ready.csv"
#     write_speed_stats_to_csv(output_csv, image_shapes, timings_per_shape)
#     print(f"Wrote timings to {output_csv}")













import pytest
import numpy as np
import jax
import timeit
import csv
import itertools

def write_speed_stats_to_csv(filename, all_rows):
    """
    Write a CSV with columns:
    batch_size, image_size, flow_fields_per_batch, particles_dim, Q1, Q3, Mean, Min, Max, StdDev
    all_rows: list of dicts
    """
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "GPU_number","batch_size", "image_size", "flow_fields_per_batch", "particles_dim", "seeding_density",
            "Q1", "Q3", "Mean", "Min", "Max", "StdDev"
        ])
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connected to the server.",
)
def test_speed_generate_images_sweep_all():
    output_csv = "new_seeding_densities.csv"
    # ---- PARAMETERS TO SWEEP ----
    batch_sizes = [64]
    image_sizes = [512]
    flow_fields_per_batch = [1]
    particles_dims = [[0.8, 1.2]]
    seeding_densities = [0.001, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060]

    selected_flow = "horizontal"
    img_offset = (10, 10)
    NUMBER_OF_EXECUTIONS = 1000
    REPETITIONS = 10

    # ---- DEVICE AND MESH SETUP ----
    shard_fields = "fields"
    num_devices = len(jax.devices())

    devices = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(devices, axis_names=(shard_fields,))
    key = jax.random.PRNGKey(0)

    all_rows = []

    # Sweep all parameter combinations
    for batch_size, image_size, flows_per_batch, particles_dim, seeding_density in itertools.product(
        batch_sizes, image_sizes, flow_fields_per_batch, particles_dims, seeding_densities
    ):

        image_shape = (image_size, image_size)
        position_bounds = (image_shape[0] + 20, image_shape[1] + 20)

        # 1. Create flow field
        flow_field = generate_array_flow_field(
            get_flow_function(selected_flow, position_bounds), position_bounds
        )
        flow_field = jnp.expand_dims(flow_field, axis=0)
        flow_field = jnp.repeat(flow_field, flows_per_batch, axis=0)
        flow_field_sharded = jax.device_put(
            flow_field, NamedSharding(mesh, PartitionSpec(shard_fields))
        )

        # 2. Setup random keys
        keys = jax.random.split(key, num_devices)
        keys = jnp.stack(keys)
        keys_sharded = jax.device_put(
            keys, NamedSharding(mesh, PartitionSpec(shard_fields))
        )
        jax.block_until_ready(keys_sharded)
        jax.block_until_ready(flow_field_sharded)

        seeding_density_range = (seeding_density, seeding_density)

        # 3. Prepare the jit function
        jit_generate_images = jax.jit(
            shard_map(
                lambda key, flow: generate_images_from_flow(
                    key=key,
                    flow_field=flow,
                    position_bounds=position_bounds,
                    image_shape=image_shape,
                    img_offset=img_offset,
                    seeding_density_range=seeding_density_range,
                    num_images=batch_size,
                    p_hide_img1=0.00,
                    p_hide_img2=0.00,
                    diameter_range=tuple(particles_dim),
                    diameter_var=0,
                    intensity_range=(80, 100),
                    intensity_var=0,
                    noise_level=0,
                    rho_range=(-0.01, 0.01),
                    rho_var=0,
                ),
                mesh=mesh,
                in_specs=(PartitionSpec(shard_fields), PartitionSpec(shard_fields)),
                out_specs=(
                    PartitionSpec(shard_fields),
                    PartitionSpec(shard_fields),
                    PartitionSpec(shard_fields),
                ),
            )
        )

        def run_generate_jit():
            imgs1, imgs2, seeding_densities = jit_generate_images(keys_sharded, flow_field_sharded)
            imgs1.block_until_ready()
            imgs2.block_until_ready()
            seeding_densities.block_until_ready()

        # Warm up
        run_generate_jit()


        # Timing
        total_time_jit = timeit.repeat(
            stmt=run_generate_jit,
            number=NUMBER_OF_EXECUTIONS // num_devices,
            repeat=REPETITIONS,
        )

        timings = np.asarray(total_time_jit)
        timings_per_img = timings / batch_size / NUMBER_OF_EXECUTIONS
        hz_per_img = 1.0 / timings_per_img

        q1 = np.percentile(hz_per_img, 25)
        q3 = np.percentile(hz_per_img, 75)
        mean = np.mean(hz_per_img)
        min_ = np.min(hz_per_img)
        max_ = np.max(hz_per_img)
        std = np.std(hz_per_img)

        all_rows.append(dict(
            GPU_number=num_devices,
            batch_size=batch_size,
            image_size=image_size,
            flow_fields_per_batch=flows_per_batch,
            particles_dim=str(particles_dim),
            seeding_density=seeding_density,
            Q1=q1, Q3=q3, Mean=mean, Min=min_, Max=max_, StdDev=std,
        ))

    # --- WRITE TO CSV ---
    write_speed_stats_to_csv(output_csv, all_rows)
