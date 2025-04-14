import os
import tempfile
import timeit

import jax
import jax.numpy as jnp
import pytest
import yaml

from synthpix.example_flows import get_flow_function
from synthpix.utils import (
    bilinear_interpolate,
    calculate_min_and_max_speeds,
    compute_image_scaled_height,
    flow_field_adapter,
    generate_array_flow_field,
    input_check_flow_field_adapter,
    is_int,
    load_configuration,
    particles_per_pixel,
    trilinear_interpolate,
    update_config_file,
)

config = load_configuration("config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_UTILS"]


@pytest.mark.parametrize(
    "original_height, original_width, new_width, expected_height",
    [
        (200, 400, 800, 400),
        (100, 400, 800, 200),
        (50, 200, 100, 25),
        (300, 600, 900, 450),
    ],
)
def test_compute_image_scaled_height(
    original_height, original_width, new_width, expected_height
):
    """Test the compute_image_scaled_height function with various inputs.

    Args:
        original_height (int): The original height of the image.
        original_width (int): The original width of the image.
        new_width (int): The new width to scale the image to.
        expected_height (int): The expected height after scaling.
    """
    assert (
        compute_image_scaled_height(original_height, original_width, new_width)
        == expected_height
    )


@pytest.mark.parametrize(
    "original_height, original_width, new_width",
    [
        (0, 400, 800),
        (100, 0, 800),
        (50, 200, 0),
        (-300, 600, 900),
        (300, -600, 900),
        (300, 600, -900),
    ],
)
def test_compute_image_scaled_height_exceptions(
    original_height, original_width, new_width
):
    """Test the compute_image_scaled_height function with invalid inputs.

    Args:
        original_height (int): The original height of the image.
        original_width (int): The original width of the image.
        new_width (int): The new width to scale the image to.
    """
    with pytest.raises(ValueError):
        compute_image_scaled_height(original_height, original_width, new_width)


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
    assert bilinear_interpolate(image, x, y) == expected


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

    print(generated_flow_field)
    print(expected)
    assert generated_flow_field.shape == (shape[0], shape[1], 2)
    assert generated_flow_field.shape == expected.shape
    assert jnp.allclose(generated_flow_field, expected, atol=1e-5)
    assert generated_flow_field.dtype == jnp.float32


@pytest.mark.parametrize(
    "image, threshold, expected",
    [
        (jnp.array([[0, 1], [2, 3]], dtype=jnp.float32), 1.0, 0.5),
        (jnp.array([[0, 1], [2, 3]], dtype=jnp.float32), 2.0, 0.25),
        (jnp.array([[0, 1], [2, 3]], dtype=jnp.float32), 3.0, 0.0),
    ],
)
def test_particles_per_pixel(image, threshold, expected):
    """Test the particles_per_pixel function with various inputs.

    Args:
        image (jnp.ndarray): The input image.
        threshold (float): The threshold to apply to the image.
        expected (float): The expected density.
    """
    # Call the function and check the result
    assert jnp.isclose(particles_per_pixel(image, threshold), expected, atol=1e-5)


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


@pytest.mark.parametrize("new_flow_field_shape", [(-128, 128), (256, 256.0), "invalid"])
def test_invalid_new_flow_field_shape(new_flow_field_shape):
    """Test that invalid new_flow_field_shape raises a ValueError."""
    flow_field = jnp.ones((256, 256, 2))
    with pytest.raises(
        ValueError,
        match="new_flow_field_shape must be a tuple of two positive integers.",
    ):
        input_check_flow_field_adapter(
            flow_field=flow_field,
            new_flow_field_shape=new_flow_field_shape,
        )


@pytest.mark.parametrize(
    "flow_field", [(256, 256), jnp.ones((256, 256)), jnp.ones((256, 256, 3)), "invalid"]
)
def test_invalid_flow_field(flow_field):
    """Test that invalid flow_field raises a ValueError."""
    new_flow_field_shape = (256, 256)
    with pytest.raises(
        ValueError,
        match="Flow_field must be a 3D jnp.ndarray with shape \\(H, W, 2\\).",
    ):
        input_check_flow_field_adapter(
            flow_field=flow_field, new_flow_field_shape=new_flow_field_shape
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

    # Call the adapter function
    new_flow_field = flow_field_adapter(
        flow_field=flow_field,
        new_flow_field_shape=new_flow_field_shape,
    )

    # Check the shape of the adapted flow field
    assert new_flow_field.shape == expected_shape

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

    # Call the adapter function
    new_flow_field = flow_field_adapter(
        flow_field=flow_field,
        new_flow_field_shape=new_flow_field_shape,
    )

    # Check the flow field
    assert jnp.allclose(new_flow_field[1, 1], expected)


# skipif is used to skip the test if the user is not connected to the server
@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize("selected_flow", ["horizontal"])
@pytest.mark.parametrize("flow_field_shape", [(1216, 1936)])
@pytest.mark.parametrize("new_flow_field_shape", [(256, 256)])
def test_speed_flow_field_adapter(
    selected_flow, flow_field_shape, new_flow_field_shape
):
    """Test that flow_field_adapter is faster than a limit time."""

    # Limit time in seconds (depends on the number of GPUs)
    limit_time = 3.5e-5

    # Create a flow field
    flow_field = generate_array_flow_field(
        get_flow_function(selected_flow, flow_field_shape), flow_field_shape
    )

    # Create the jit function
    flow_field_adapter_jit = jax.jit(
        flow_field_adapter, static_argnames=["new_flow_field_shape"]
    )

    def run_flow_field_adapter_jit():
        result = flow_field_adapter_jit(
            flow_field=flow_field, new_flow_field_shape=new_flow_field_shape
        )
        result.block_until_ready()

    # Warm up the function
    run_flow_field_adapter_jit()

    # Measure the time of the jit function
    total_time_jit = timeit.repeat(
        stmt=run_flow_field_adapter_jit, number=NUMBER_OF_EXECUTIONS, repeat=REPETITIONS
    )
    average_time_jit = min(total_time_jit) / NUMBER_OF_EXECUTIONS

    # Check if the time is less than the limit
    assert (
        average_time_jit < limit_time
    ), f"The average time is {average_time_jit}, time limit: {limit_time}"
