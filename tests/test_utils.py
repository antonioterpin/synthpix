"""Tests for low-level utility functions and data manipulation helpers.

These tests cover interpolation logic (bilinear, trilinear), flow field 
adaptation, configuration loading, and directory discovery for dataset 
preprocessing.
"""
import os
import re
import sys
import timeit
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from synthpix.utils import (bilinear_interpolate, discover_leaf_dirs,
                            flow_field_adapter, generate_array_flow_field,
                            input_check_flow_field_adapter, load_configuration,
                            trilinear_interpolate)
from tests.example_flows import get_flow_function

config = load_configuration("config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_UTILS"]


@pytest.mark.parametrize(
    "image, x, y, expected",
    [
        (jnp.array([[0, 1], [2, 3]]), 0.5, 0.5, 1.5),
        (jnp.array([[0, 1], [2, 3]]), 0.25, 0.75, 1.75),
        (jnp.array([[0, 1], [2, 3]]), 0.75, 0.25, 1.25),
        (jnp.array([[0, 1], [2, 3]]), -0.5, -0.5, 0.0),
    ],
)
def test_bilinear_interpolate(
    image: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, expected: jnp.ndarray
):
    """Test the bilinear_interpolate function with various inputs

    Args:
        image: The input image.
        x: The x-coordinates for interpolation.
        y: The y-coordinates for interpolation.
        expected: The expected interpolated values.
    """
    res = bilinear_interpolate(image, x, y)
    assert res == expected, f"Expected {expected} but got {res}"


@pytest.mark.parametrize(
    "image, x, y, z, expected",
    [
        (jnp.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]), 0.5, 0.5, 0.5, 3.5),
        (
            jnp.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]),
            0.25,
            0.75,
            0.75,
            4.75,
        ),
        (
            jnp.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]),
            -0.5,
            -0.5,
            -0.5,
            0.0,
        ),
    ],
)
def test_trilinear_interpolate(
    image: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    expected: jnp.ndarray,
):
    """Test the trilinear_interpolate function with various inputs.

    Args:
        image: The input image.
        x: The x-coordinates for interpolation.
        y: The y-coordinates for interpolation.
        z: The z-coordinates for interpolation.
        expected: The expected interpolated values.
    """
    assert trilinear_interpolate(image, x, y, z) == expected, f"Trilinear interpolation mismatch. Expected {expected}, got {trilinear_interpolate(image, x, y, z)}"


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
def test_generate_array_flow_field(
    shape: tuple[int, int], flow_field_type, expected: jnp.ndarray
):
    """Test generating a discretized flow field array from a functional description.

    Verifies that the resulting array matches the expected shape, dtype, 
    and values for standard flow types (horizontal, vertical).
    """
    # Generate the flow field using the specified type
    flow_field = get_flow_function(flow_field_type)
    generated_flow_field = generate_array_flow_field(flow_field, shape)

    assert generated_flow_field.shape == (shape[0], shape[1], 2), f"Generated flow field shape mismatch. Expected {(shape[0], shape[1], 2)}, got {generated_flow_field.shape}"
    assert generated_flow_field.shape == expected.shape, f"Generated flow field shape {generated_flow_field.shape} does not match expected shape {expected.shape}"
    assert jnp.allclose(generated_flow_field, expected, atol=1e-5), "Generated flow field values do not match expected values"
    assert generated_flow_field.dtype == jnp.float32, f"Expected dtype jnp.float32, got {generated_flow_field.dtype}"


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


@pytest.mark.parametrize("flow_field", [(256,), "invalid", None])
def test_invalid_flow_field(flow_field):
    """Test input_check_flow_field_adapter raises ValueError for invalid flow fields."""
    with pytest.raises(ValueError, match="flow_field must be a jnp.ndarray."):
        input_check_flow_field_adapter(
            flow_field=flow_field,
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
            zero_padding=valid_zero_padding,
        )


@pytest.mark.parametrize(
    "flow_field",
    [
        jnp.zeros((256, 256, 2)),
        jnp.ones((256, 256)),
        jnp.zeros((64, 256, 256, 2, 1)),
    ],
)
def test_invalid_flow_field_dims(flow_field):
    """Test that `input_check_flow_field_adapter` rejects flow fields with incorrect rank.

    Expects a 4D array (N, H, W, 2).
    """
    pattern = (
        r"^flow_field must be a 4D jnp\.ndarray with shape \(N, H, W, 2\), "
        r"got " + re.escape(str(flow_field.shape)) + r"\.$"
    )
    with pytest.raises(ValueError, match=pattern):
        input_check_flow_field_adapter(
            flow_field=flow_field,
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
            zero_padding=valid_zero_padding,
        )


@pytest.mark.parametrize("N", [1, 3, 64, 12, 0])
def test_invalid_flow_field_channels(N):
    """Test that `input_check_flow_field_adapter` rejects flow fields with incorrect channels.

    Expects exactly 2 components (u, v) in the last dimension.
    """
    ff = jnp.zeros((16, 64, 64, N))
    pattern = (
        r"^flow_field must have shape \(N, H, W, 2\), "
        r"got " + re.escape(str(ff.shape)) + r"\.$"
    )
    with pytest.raises(ValueError, match=pattern):
        input_check_flow_field_adapter(
            flow_field=ff,
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
            zero_padding=valid_zero_padding,
        )


@pytest.mark.parametrize(
    "new_flow_field_shape",
    [(256,), (256, 1, 3), "invalid", None, (234.1, 243.3), (-256, 245)],
)
def test_invalid_new_flow_field_shape(new_flow_field_shape):
    """Test that specifying an invalid target shape raises a ValueError.

    Expects a tuple of two positive integers for (H, W).
    """
    pattern = (
        r"new_flow_field_shape must be a tuple of two positive integers, "
        r"got " + re.escape(str(new_flow_field_shape)) + r"\.$"
    )
    with pytest.raises(ValueError, match=pattern):
        input_check_flow_field_adapter(
            flow_field=valid_flow_field,
            new_flow_field_shape=new_flow_field_shape,
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
            zero_padding=valid_zero_padding,
        )


@pytest.mark.parametrize("image_shape", [(256,), "invalid"])
def test_invalid_image_shape_format(image_shape):
    with pytest.raises(
        ValueError,
        match="image_shape must be a tuple of two positive integers.",
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
        ValueError,
        match="img_offset must be a tuple of two non-negative numbers.",
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
    with pytest.raises(
        ValueError, match=f"{param_name} must be a positive number."
    ):
        input_check_flow_field_adapter(**args)


@pytest.mark.parametrize("position_bounds", [(256,), "invalid"])
def test_invalid_position_bounds_format(position_bounds):
    with pytest.raises(
        ValueError,
        match="position_bounds must be a tuple of two positive numbers.",
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
    with pytest.raises(
        ValueError, match="batch_size must be a positive integer."
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
            batch_size=batch_size,
            output_units="pixels",
            dt=valid_dt,
            zero_padding=valid_zero_padding,
        )


@pytest.mark.parametrize("output_units", [None, "invalid", 1.0])
def test_invalid_output_units_adapter(output_units, scheduler):
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
        ValueError,
        match="zero_padding must be a tuple of two non-negative integers.",
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
        (
            "horizontal",
            (1280, 1280),
            (128, 128),
            (128, 128, 2),
            jnp.array([10.0, 0.0]),
        ),
        (
            "vertical",
            (12, 12),
            (256, 256),
            (256, 256, 2),
            jnp.array([0.0, 10.0]),
        ),
        (
            "diagonal",
            (1, 1),
            (2560, 2560),
            (2560, 2560, 2),
            jnp.array([10.0, 10.0]),
        ),
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
    assert new_flow_field[0][0].shape == expected_shape, f"Adapted flow field shape mismatch. Expected {expected_shape}, got {new_flow_field[0][0].shape}"

    # Check the first vector of the adapted flow field
    assert jnp.allclose(new_flow_field[0][0], expected_first_vector), "Adapted flow field values do not match expected first vector"


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
    assert jnp.allclose(new_flow_field[0][0][1, 1], expected), f"Central vector mismatch. Expected {expected}, got {new_flow_field[0][0][1, 1]}"


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
    """Benchmark performance of GPU-parallelized flow field adaptation.

    Uses `shard_map` to distribute interpolation tasks across GPUs and 
    verifies that the latency stays within acceptable limits for 
    real-time or batch processing.
    """

    # Check how many GPUs are available
    devices = jax.devices()
    if len(devices) == 3:
        devices = devices[:2]
    elif len(devices) > 4:
        devices = devices[:4]
    num_devices = len(devices)

    # Limit time in seconds (depends on the number of GPUs)
    if num_devices == 1:
        limit_time = 1.5e-2
    elif num_devices == 2:
        limit_time = 1.2e-2
    elif num_devices == 4:
        limit_time = 2.1e-3

    # Name of the axis for the device mesh
    shard_fields = "fields"
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
            out_specs=(
                PartitionSpec(shard_fields),
                PartitionSpec(shard_fields),
            ),
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
        stmt=run_flow_field_adapter_jit,
        number=NUMBER_OF_EXECUTIONS,
        repeat=REPETITIONS,
    )
    average_time_jit = min(total_time_jit) / NUMBER_OF_EXECUTIONS

    assert average_time_jit < limit_time, (
        f"The average time is {average_time_jit}, time limit: {limit_time}"
    )


def _abspath(p):  # small helper to compare reliably
    return os.path.abspath(str(p))


def test_discover_leaf_dirs(tmp_path):
    """Test the automatic discovery of leaf directories containing `.mat` files.

    Verifies that the function correctly identifies directories at the end 
    of the hierarchy that actually contain relevant data files, 
    ignoring parent or empty branches.
    """
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
    (tmp_path / "seq_C").mkdir()  # empty, ignored

    # Drop dummy files
    for t in (0, 1):
        (seq_A / f"flow_{t:04d}.mat").touch()
    (seq_B / "flow_0000.mat").touch()
    (sub_1 / "flow_0002.mat").touch()

    # Collect file paths as strings (API contract), then resolve results for
    # comparison
    paths = [str(p) for p in tmp_path.rglob("*.mat")]

    leaves = {Path(p).resolve() for p in discover_leaf_dirs(paths)}
    expect = {seq_A.resolve(), sub_1.resolve()}

    assert leaves == expect, f"Expected {expect}, got {leaves}"


def test_discover_leaf_dirs_skips_missing_dirs(tmp_path, monkeypatch):
    """Simulate FileNotFoundError for one directory during discovery."""
    gone = tmp_path / "gone_seq"
    gone.mkdir()
    (gone / "flow_0000.mat").touch()

    keep = tmp_path / "keep_seq"
    keep.mkdir()
    (keep / "flow_0000.mat").touch()

    paths = [str(p) for p in tmp_path.rglob("*.mat")]

    real_scandir = os.scandir

    def fake_scandir(path):
        if os.path.abspath(path) == _abspath(gone):
            raise FileNotFoundError(path)
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", fake_scandir)

    leaves = set(map(os.path.abspath, discover_leaf_dirs(paths)))
    assert _abspath(keep) in leaves, f"Expected {_abspath(keep)} to be in leaves: {leaves}"
    assert _abspath(gone) not in leaves, f"Expected {_abspath(gone)} to be excluded from leaves"


def test_discover_leaf_dirs_skips_not_a_directory(tmp_path, monkeypatch):
    """Simulate NotADirectoryError for a path that used to be a directory."""
    will_be_file = tmp_path / "turns_into_file"
    will_be_file.mkdir()
    (will_be_file / "flow_0000.mat").touch()

    ok = tmp_path / "ok_seq"
    ok.mkdir()
    (ok / "flow_0000.mat").touch()

    paths = [str(p) for p in tmp_path.rglob("*.mat")]

    real_scandir = os.scandir

    def fake_scandir(path):
        if os.path.abspath(path) == _abspath(will_be_file):
            raise NotADirectoryError(path)
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", fake_scandir)

    leaves = set(map(os.path.abspath, discover_leaf_dirs(paths)))
    assert _abspath(ok) in leaves, f"Expected {_abspath(ok)} to be in leaves, got {leaves}"
    assert _abspath(will_be_file) not in leaves, f"Expected {_abspath(will_be_file)} (not a directory) to be excluded from leaves"


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="permission semantics differ on Windows",
)
def test_discover_leaf_dirs_skips_permission_denied(tmp_path, monkeypatch):
    """Simulate PermissionError without touching actual filesystem permissions."""
    denied = tmp_path / "no_perm_seq"
    denied.mkdir()
    (denied / "flow_0000.mat").touch()

    ok = tmp_path / "ok_seq"
    ok.mkdir()
    (ok / "flow_0000.mat").touch()

    paths = [str(p) for p in tmp_path.rglob("*.mat")]

    real_scandir = os.scandir

    def fake_scandir(path):
        if os.path.abspath(path) == _abspath(denied):
            raise PermissionError(path)
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", fake_scandir)

    leaves = set(map(os.path.abspath, discover_leaf_dirs(paths)))
    assert _abspath(ok) in leaves, f"Expected {_abspath(ok)} to be in leaves, got {leaves}"
    assert _abspath(denied) not in leaves, f"Expected {_abspath(denied)} (permission denied) to be excluded from leaves"
