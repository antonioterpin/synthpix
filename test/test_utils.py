import os
import tempfile
import timeit

import jax
import jax.numpy as jnp
import pytest
import yaml
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from synthpix.example_flows import get_flow_function
from synthpix.sanity import calculate_min_and_max_speeds, update_config_file
from synthpix.utils import (
    bilinear_interpolate,
    discover_leaf_dirs,
    flow_field_adapter,
    generate_array_flow_field,
    input_check_flow_field_adapter,
    is_int,
    load_configuration,
    trilinear_interpolate,
)

config = load_configuration("config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_UTILS"]


@pytest.mark.parametrize(
    "val, expected",
    [
        (1, True),
        (1.0, True),
        (1.000000001, True),
        (1.00000001, True),
        (1.5, False),
        (0, True),
        (-1, True),
        (-1.0, True),
        (-1.000000001, True),
        (-1.00000001, True),
        (1e9, True),
        (1e9 + 0.000000001, True),
        (1e9 + 0.00000001, True),
    ],
)
def test_is_int(val, expected):
    """Test the is_int function with various inputs.

    Args:
        val (Union[int, float]): The value to check.
        expected (bool): The expected result.
    """
    assert is_int(val) == expected


@pytest.mark.parametrize(
    "image, x, y, expected",
    [
        (jnp.array([[0, 1], [2, 3]]), 0.5, 0.5, 1.5),
        (jnp.array([[0, 1], [2, 3]]), 0.25, 0.75, 1.75),
        (jnp.array([[0, 1], [2, 3]]), 0.75, 0.25, 1.25),
        (jnp.array([[0, 1], [2, 3]]), -0.5, -0.5, 0.0),
    ],
)
def test_bilinear_interpolate(image, x, y, expected):
    """Test the bilinear_interpolate function with various inputs

    Args:
        image (jnp.ndarray): The input image.
        x (jnp.ndarray): The x-coordinates for interpolation.
        y (jnp.ndarray): The y-coordinates for interpolation.
        expected (jnp.ndarray): The expected interpolated values.
    """
    res = bilinear_interpolate(image, x, y)
    assert res == expected, f"Expected {expected} but got {res}"


@pytest.mark.parametrize(
    "image, x, y, z, expected",
    [
        (jnp.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]), 0.5, 0.5, 0.5, 3.5),
        (jnp.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]), 0.25, 0.75, 0.75, 4.75),
        (jnp.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]), -0.5, -0.5, -0.5, 0.0),
    ],
)
def test_trilinear_interpolate(image, x, y, z, expected):
    """Test the trilinear_interpolate function with various inputs.

    Args:
        image (jnp.ndarray): The input image.
        x (jnp.ndarray): The x-coordinates for interpolation.
        y (jnp.ndarray): The y-coordinates for interpolation.
        z (jnp.ndarray): The z-coordinates for interpolation.
        expected (jnp.ndarray): The expected interpolated values.
    """
    assert trilinear_interpolate(image, x, y, z) == expected


@pytest.mark.parametrize(
    "shape, flow_field_type, expected",
    [
        (
            (10, 10),
            "vertical",
            jnp.ones((10, 10, 2), dtype=jnp.float32)
            * jnp.array([0, 10], dtype=jnp.float32),
        ),
        (
            (20, 20),
            "horizontal",
            jnp.ones((20, 20, 2), dtype=jnp.float32)
            * jnp.array([10, 0], dtype=jnp.float32),
        ),
    ],
)
def test_generate_array_flow_field(shape, flow_field_type, expected):
    """Test the generate_array_flow_field function with various inputs.

    Args:
        shape (tuple): The shape of the flow field.
        flow_field (jnp.ndarray): The expected flow field.
    """
    # Generate the flow field using the specified type
    flow_field = get_flow_function(flow_field_type)
    generated_flow_field = generate_array_flow_field(flow_field, shape)

    assert generated_flow_field.shape == (shape[0], shape[1], 2)
    assert generated_flow_field.shape == expected.shape
    assert jnp.allclose(generated_flow_field, expected, atol=1e-5)
    assert generated_flow_field.dtype == jnp.float32


def test_update_config_file():
    """Test the update_config_file function."""
    # Create a temporary configuration file based on test_data.yaml
    base_config_path = os.path.join("config", "test_data.yaml")
    tmp_path = tempfile.gettempdir()
    temp_config_path = os.path.join(tmp_path, "temp_config.yaml")
    try:
        base_config = load_configuration(base_config_path)
        with open(temp_config_path, "w") as temp_file:
            yaml.safe_dump(base_config, temp_file)
        # Define the updates to be made
        updates = {
            "max_speed_x": 15.0,
            "max_speed_y": 20.0,
            "min_speed_x": -15.0,
            "min_speed_y": -20.0,
        }
        # Call the function to update the configuration file
        update_config_file(temp_config_path, updates)
        # Reload the updated configuration file
        updated_config = load_configuration(temp_config_path)
        # Assert that the updates were applied correctly
        for key, value in updates.items():
            assert updated_config[key] == value
        # Assert that other keys remain unchanged
        for key in base_config:
            if key not in updates:
                assert updated_config[key] == base_config[key]
    finally:
        # Ensure the temporary file is deleted
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


@pytest.mark.parametrize("mock_hdf5_files", [2], indirect=True)
def test_calculate_min_and_max_speeds(mock_hdf5_files):
    """Test the calculate_min_and_max_speeds function."""
    files, dims = mock_hdf5_files

    # Call the function to calculate speeds
    result = calculate_min_and_max_speeds(files)

    # Assert the results
    assert "min_speed_x" in result
    assert "max_speed_x" in result
    assert "min_speed_y" in result
    assert "max_speed_y" in result

    # Ensure the values are within expected ranges based on the mock data
    assert result["min_speed_x"] <= result["max_speed_x"]
    assert result["min_speed_y"] <= result["max_speed_y"]

    # Test with invalid inputs
    with pytest.raises(ValueError):
        calculate_min_and_max_speeds([])  # Empty list

    with pytest.raises(ValueError):
        calculate_min_and_max_speeds(["nonexistent_file.h5"])  # Nonexistent file


# Mock valid inputs
valid_flow_field = jnp.ones((1, 256, 256, 2))
valid_shape = (256, 256)
valid_offset = (0, 0)
valid_resolution = 1.0
valid_position_bounds = (256, 256)
valid_position_offset = (0, 0)
valid_batch_size = 1
valid_dt = 1.0
valid_zero_padding = (0, 0)


@pytest.mark.parametrize("image_shape", [(256,), "invalid"])
def test_invalid_image_shape_format(image_shape):
    with pytest.raises(
        ValueError, match="image_shape must be a tuple of two positive integers."
    ):
        input_check_flow_field_adapter(
            flow_field=valid_flow_field,
            new_flow_field_shape=valid_shape,
            image_shape=image_shape,
            img_offset=valid_offset,
            resolution=valid_resolution,
            res_x=valid_resolution,
            res_y=valid_resolution,
            position_bounds=valid_position_bounds,
            position_bounds_offset=valid_position_offset,
            batch_size=valid_batch_size,
            output_units="pixels",
            dt=valid_dt,
            zero_padding=valid_zero_padding,
        )


@pytest.mark.parametrize("image_shape", [(0, 256), (-1, 256), (256, "256")])
def test_invalid_image_shape_values(image_shape):
    with pytest.raises(
        ValueError, match="image_shape must contain two positive integers."
    ):
        input_check_flow_field_adapter(
            flow_field=valid_flow_field,
            new_flow_field_shape=valid_shape,
            image_shape=image_shape,
            img_offset=valid_offset,
            resolution=valid_resolution,
            res_x=valid_resolution,
            res_y=valid_resolution,
            position_bounds=valid_position_bounds,
            position_bounds_offset=valid_position_offset,
            batch_size=valid_batch_size,
            output_units="pixels",
            dt=valid_dt,
            zero_padding=valid_zero_padding,
        )


@pytest.mark.parametrize("img_offset", [(256,), "invalid"])
def test_invalid_img_offset_format(img_offset):
    with pytest.raises(
        ValueError, match="img_offset must be a tuple of two non-negative numbers."
    ):
        input_check_flow_field_adapter(
            flow_field=valid_flow_field,
            new_flow_field_shape=valid_shape,
            image_shape=valid_shape,
            img_offset=img_offset,
            resolution=valid_resolution,
            res_x=valid_resolution,
            res_y=valid_resolution,
            position_bounds=valid_position_bounds,
            position_bounds_offset=valid_position_offset,
            batch_size=valid_batch_size,
            output_units="pixels",
            dt=valid_dt,
            zero_padding=valid_zero_padding,
        )


@pytest.mark.parametrize("img_offset", [(256, -1), ("0", 0)])
def test_invalid_img_offset_values(img_offset):
    with pytest.raises(
        ValueError, match="img_offset must contain two non-negative numbers."
    ):
        input_check_flow_field_adapter(
            flow_field=valid_flow_field,
            new_flow_field_shape=valid_shape,
            image_shape=valid_shape,
            img_offset=img_offset,
            resolution=valid_resolution,
            res_x=valid_resolution,
            res_y=valid_resolution,
            position_bounds=valid_position_bounds,
            position_bounds_offset=valid_position_offset,
            batch_size=valid_batch_size,
            output_units="pixels",
            dt=valid_dt,
            zero_padding=valid_zero_padding,
        )


@pytest.mark.parametrize(
    "value,param_name",
    [
        (-1.0, "resolution"),
        ("1.0", "resolution"),
        (0.0, "resolution"),
        (-1.0, "res_x"),
        (None, "res_x"),
        (0, "res_y"),
        (-0.1, "res_y"),
    ],
)
def test_invalid_resolutions(value, param_name):
    args = {
        "flow_field": valid_flow_field,
        "new_flow_field_shape": valid_shape,
        "image_shape": valid_shape,
        "img_offset": valid_offset,
        "resolution": valid_resolution,
        "res_x": valid_resolution,
        "res_y": valid_resolution,
        "position_bounds": valid_position_bounds,
        "position_bounds_offset": valid_position_offset,
        "batch_size": valid_batch_size,
        "output_units": "pixels",
        "dt": valid_dt,
        "zero_padding": valid_zero_padding,
    }
    args[param_name] = value
    with pytest.raises(ValueError, match=f"{param_name} must be a positive number."):
        input_check_flow_field_adapter(**args)


@pytest.mark.parametrize("position_bounds", [(256,), "invalid"])
def test_invalid_position_bounds_format(position_bounds):
    with pytest.raises(
        ValueError, match="position_bounds must be a tuple of two positive numbers."
    ):
        input_check_flow_field_adapter(
            flow_field=valid_flow_field,
            new_flow_field_shape=valid_shape,
            image_shape=valid_shape,
            img_offset=valid_offset,
            resolution=valid_resolution,
            res_x=valid_resolution,
            res_y=valid_resolution,
            position_bounds=position_bounds,
            position_bounds_offset=valid_position_offset,
            batch_size=valid_batch_size,
            output_units="pixels",
            dt=valid_dt,
            zero_padding=valid_zero_padding,
        )


@pytest.mark.parametrize("position_bounds", [(-1, 256), ("a", 256)])
def test_invalid_position_bounds_values(position_bounds):
    with pytest.raises(
        ValueError, match="position_bounds must contain two positive numbers."
    ):
        input_check_flow_field_adapter(
            flow_field=valid_flow_field,
            new_flow_field_shape=valid_shape,
            image_shape=valid_shape,
            img_offset=valid_offset,
            resolution=valid_resolution,
            res_x=valid_resolution,
            res_y=valid_resolution,
            position_bounds=position_bounds,
            position_bounds_offset=valid_position_offset,
            batch_size=valid_batch_size,
            output_units="pixels",
            dt=valid_dt,
            zero_padding=valid_zero_padding,
        )


@pytest.mark.parametrize("position_bounds_offset", [(256,), "invalid"])
def test_invalid_position_bounds_offset_format(position_bounds_offset):
    with pytest.raises(
        ValueError,
        match="position_bounds_offset must be a tuple of two non-negative numbers.",
    ):
        input_check_flow_field_adapter(
            flow_field=valid_flow_field,
            new_flow_field_shape=valid_shape,
            image_shape=valid_shape,
            img_offset=valid_offset,
            resolution=valid_resolution,
            res_x=valid_resolution,
            res_y=valid_resolution,
            position_bounds=valid_position_bounds,
            position_bounds_offset=position_bounds_offset,
            batch_size=valid_batch_size,
            output_units="pixels",
            dt=valid_dt,
            zero_padding=valid_zero_padding,
        )


@pytest.mark.parametrize("position_bounds_offset", [(-1, 0), ("0", 0)])
def test_invalid_position_bounds_offset_values(position_bounds_offset):
    with pytest.raises(
        ValueError,
        match="position_bounds_offset must contain two non-negative numbers.",
    ):
        input_check_flow_field_adapter(
            flow_field=valid_flow_field,
            new_flow_field_shape=valid_shape,
            image_shape=valid_shape,
            img_offset=valid_offset,
            resolution=valid_resolution,
            res_x=valid_resolution,
            res_y=valid_resolution,
            position_bounds=valid_position_bounds,
            position_bounds_offset=position_bounds_offset,
            batch_size=valid_batch_size,
            output_units="pixels",
            dt=valid_dt,
            zero_padding=valid_zero_padding,
        )


@pytest.mark.parametrize("batch_size", [-1, 0, "1", 1.5])
def test_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch_size must be a positive integer."):
        input_check_flow_field_adapter(
            flow_field=valid_flow_field,
            new_flow_field_shape=valid_shape,
            image_shape=valid_shape,
            img_offset=valid_offset,
            resolution=valid_resolution,
            res_x=valid_resolution,
            res_y=valid_resolution,
            position_bounds=valid_position_bounds,
            position_bounds_offset=valid_position_offset,
            batch_size=batch_size,
            output_units="pixels",
            dt=valid_dt,
            zero_padding=valid_zero_padding,
        )


@pytest.mark.parametrize("output_units", [None, "invalid", 1.0])
def test_invalid_output_units(output_units, scheduler):
    with pytest.raises(
        ValueError,
        match="output_units must be either 'pixels' or 'measure units per second'.",
    ):
        input_check_flow_field_adapter(
            flow_field=valid_flow_field,
            new_flow_field_shape=valid_shape,
            image_shape=valid_shape,
            img_offset=valid_offset,
            resolution=valid_resolution,
            res_x=valid_resolution,
            res_y=valid_resolution,
            position_bounds=valid_position_bounds,
            position_bounds_offset=valid_position_offset,
            batch_size=valid_batch_size,
            output_units=output_units,
            dt=valid_dt,
            zero_padding=valid_zero_padding,
        )


@pytest.mark.parametrize("dt", [None, "invalid", -1.0, 0.0])
def test_invalid_dt(dt):
    with pytest.raises(ValueError, match="dt must be a positive number."):
        input_check_flow_field_adapter(
            flow_field=valid_flow_field,
            new_flow_field_shape=valid_shape,
            image_shape=valid_shape,
            img_offset=valid_offset,
            resolution=valid_resolution,
            res_x=valid_resolution,
            res_y=valid_resolution,
            position_bounds=valid_position_bounds,
            position_bounds_offset=valid_position_offset,
            batch_size=valid_batch_size,
            output_units="pixels",
            dt=dt,
            zero_padding=valid_zero_padding,
        )


@pytest.mark.parametrize("zero_padding", [(1,), "invalid", (-1, 0), (0, -1)])
def test_invalid_zero_padding(zero_padding):
    with pytest.raises(
        ValueError, match="zero_padding must be a tuple of two non-negative integers."
    ):
        input_check_flow_field_adapter(
            flow_field=valid_flow_field,
            new_flow_field_shape=valid_shape,
            image_shape=valid_shape,
            img_offset=valid_offset,
            resolution=valid_resolution,
            res_x=valid_resolution,
            res_y=valid_resolution,
            position_bounds=valid_position_bounds,
            position_bounds_offset=valid_position_offset,
            batch_size=valid_batch_size,
            output_units="pixels",
            dt=valid_dt,
            zero_padding=zero_padding,
        )


@pytest.mark.parametrize(
    "flow_field, flow_field_shape, new_flow_field_shape, "
    "expected_shape, expected_first_vector",
    [
        ("horizontal", (1280, 1280), (128, 128), (128, 128, 2), jnp.array([10.0, 0.0])),
        ("vertical", (12, 12), (256, 256), (256, 256, 2), jnp.array([0.0, 10.0])),
        ("diagonal", (1, 1), (2560, 2560), (2560, 2560, 2), jnp.array([10.0, 10.0])),
    ],
)
def test_flow_field_adapter_shape(
    flow_field,
    flow_field_shape,
    new_flow_field_shape,
    expected_shape,
    expected_first_vector,
):
    """Test that flow_field_adapter returns the correct shape and first vector."""
    # Generate a flow field based on the selected flow type
    flow_function = get_flow_function(flow_field)
    flow_field = generate_array_flow_field(flow_function, flow_field_shape)
    num_flows = 4
    flow_fields = jnp.tile(flow_field, (num_flows, 1, 1, 1))

    # Call the adapter function
    new_flow_field = flow_field_adapter(
        flow_fields=flow_fields,
        new_flow_field_shape=new_flow_field_shape,
    )

    # Check the shape of the adapted flow field
    assert new_flow_field[0][0].shape == expected_shape

    # Check the first vector of the adapted flow field
    assert jnp.allclose(new_flow_field[0][0], expected_first_vector)


@pytest.mark.parametrize(
    "flow_field, new_flow_field_shape, expected",
    [
        (
            jnp.array([[[10, 10], [10, 0]], [[0, 0], [0, 0]]]),
            (3, 3),
            jnp.array([5.0, 2.5]),
        ),
        (
            jnp.array([[[10, 5], [0, 5]], [[0, 5], [10, 5]]]),
            (3, 3),
            jnp.array([5.0, 5.0]),
        ),
        (
            jnp.array([[[1, 5], [2, 6]], [[3, 7], [4, 8]]]),
            (3, 3),
            jnp.array([2.5, 6.5]),
        ),
    ],
)
def test_flow_field_adapter(flow_field, new_flow_field_shape, expected):
    """Test that flow_field_adapter returns the correct central vector."""
    num_flows = 1
    flow_fields = jnp.tile(flow_field, (num_flows, 1, 1, 1))

    # Call the adapter function
    new_flow_field = flow_field_adapter(
        flow_fields=flow_fields,
        new_flow_field_shape=new_flow_field_shape,
        image_shape=(3, 3),
        res_x=2 / 3,
        res_y=2 / 3,
    )

    # Check the flow field
    assert jnp.allclose(new_flow_field[0][0][1, 1], expected)


@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize("selected_flow", ["horizontal"])
@pytest.mark.parametrize("flow_field_shape", [(1536, 1024)])
@pytest.mark.parametrize("new_flow_field_shape", [(1216, 1936)])
@pytest.mark.parametrize("batch_size", [128])
def test_speed_flow_fields_adapter(
    selected_flow, flow_field_shape, new_flow_field_shape, batch_size
):
    """Test that flow_field_adapter is faster than a limit time."""

    # Check how many GPUs are available
    num_devices = len(jax.devices())

    # Limit time in seconds (depends on the number of GPUs)
    if num_devices == 1:
        limit_time = 1.5e-2
    elif num_devices == 2:
        limit_time = 1.2e-2
    elif num_devices == 4:
        limit_time = 2.1e-3

    # Name of the axis for the device mesh
    shard_fields = "fields"
    devices = jax.devices()
    num_devices = len(devices)
    mesh = Mesh(devices, axis_names=(shard_fields))

    sharding = NamedSharding(mesh, PartitionSpec(shard_fields))

    # Generate a flow field with shape (N, H, W, 2)
    flow_field = generate_array_flow_field(
        get_flow_function(selected_flow, flow_field_shape), flow_field_shape
    )
    flow_fields = jnp.tile(flow_field, (batch_size // num_devices, 1, 1, 1))

    flow_fields = jax.device_put(flow_fields, sharding)

    flow_field_adapter_jit = jax.jit(
        jax.shard_map(
            lambda flow: flow_field_adapter(
                flow,
                new_flow_field_shape=new_flow_field_shape,
                image_shape=(1216, 1936),
                img_offset=(0, 0),
                resolution=1.0,
                res_x=1.0,
                res_y=1.0,
                batch_size=batch_size // num_devices,
            ),
            mesh=mesh,
            in_specs=(PartitionSpec(shard_fields)),
            out_specs=(PartitionSpec(shard_fields), PartitionSpec(shard_fields)),
        )
    )

    def run_flow_field_adapter_jit():
        result = flow_field_adapter_jit(flow_fields)
        result[0].block_until_ready()
        result[1].block_until_ready()

    # Warm up
    run_flow_field_adapter_jit()

    # Time the JIT-ed function
    total_time_jit = timeit.repeat(
        stmt=run_flow_field_adapter_jit, number=NUMBER_OF_EXECUTIONS, repeat=REPETITIONS
    )
    average_time_jit = min(total_time_jit) / NUMBER_OF_EXECUTIONS

    assert (
        average_time_jit < limit_time
    ), f"The average time is {average_time_jit}, time limit: {limit_time}"


def test__discover_leaf_dirs(tmp_path, generate_mat_file, mat_test_dims):
    """
    tmp_path/
      ├── seq_A/
      │   ├── flow_0000.mat
      │   └── flow_0001.mat
      ├── seq_B/                    # <─ NOT a leaf
      │   ├── flow_0000.mat
      │   └── sub_1/
      │       └── flow_0002.mat
      └── seq_C/                    # <─ empty, should be ignored
    """
    # Build directories
    seq_A = tmp_path / "seq_A"
    seq_A.mkdir()
    seq_B = tmp_path / "seq_B"
    seq_B.mkdir()
    sub_1 = seq_B / "sub_1"
    sub_1.mkdir()
    str(sub_1)
    seq_C = tmp_path / "seq_C"
    seq_C.mkdir()

    # Drop dummy files
    for t in (0, 1):
        generate_mat_file(seq_A, t, mat_test_dims)
    generate_mat_file(seq_B, 0, mat_test_dims)
    generate_mat_file(sub_1, 2, mat_test_dims)

    filepath_seq_A_0 = os.path.join(seq_A, "flow_0000.mat")
    filepath_seq_A_1 = os.path.join(seq_A, "flow_0001.mat")
    filepath_seq_B_0 = os.path.join(seq_B, "flow_0000.mat")
    filepath_sub_1_2 = os.path.join(sub_1, "flow_0002.mat")

    # Turn the path into a string
    paths = [
        str(filepath_seq_A_0),
        str(filepath_seq_A_1),
        str(filepath_seq_B_0),
        str(filepath_sub_1_2),
    ]

    # What does the static helper think are leaves?
    leaves = discover_leaf_dirs(paths)
    leaves = set(map(os.path.abspath, leaves))

    expected = {
        os.path.abspath(seq_A),
        os.path.abspath(sub_1),  # leaf even though parent has child dir
    }
    assert leaves == expected, f"Expected {expected}, got {leaves}"
