"""Tests for the high-level image generation functions and their input
validation.

This module contains tests for `generate_images_from_flow`,
verifying that particle-based synthetic images are correctly
rendered across various parameter spaces (densities, diameters,
intensities, noise, etc.). It also includes exhaustive input
validation checks and performance benchmarks for the rendering
pipeline.
"""

from __future__ import annotations

import os
import re
import timeit
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import synthpix.data_generate as dg
from synthpix.data_generate import (
    generate_images_from_flow,
    input_check_gen_img_from_flow,
)
from synthpix.types import ImageGenerationSpecification

# Import existing modules
from synthpix.utils import generate_array_flow_field, load_configuration
from tests.example_flows import get_flow_function

config = load_configuration("config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_DATA_GEN"]


@pytest.mark.parametrize("flow_field", [None, "invalid_flow", 42, [1, 2]])
def test_invalid_flow_field(flow_field: Any) -> None:
    """Test that provide non-array flow_field objects raise a ValueError.

    Validates that the input is a jnp.ndarray, which is required for JAX
    accelerated generation.

    Args:
        flow_field: Flow field supplied to the operation under test.
    """
    with pytest.raises(
        ValueError,
        match=(f"flow_field must be a jnp.ndarray, got {type(flow_field)}."),
    ):
        input_check_gen_img_from_flow(
            flow_field=flow_field, parameters=ImageGenerationSpecification()
        )


@pytest.mark.parametrize(
    "flow_field", [jnp.zeros((1, 128, 128, 3)), jnp.zeros((128, 128, 2))]
)
def test_invalid_flow_field_shape(flow_field: Any) -> None:
    """Test that flow fields with incorrect rank or dimensions raise a
    ValueError.

    Expects a 4D array (N, H, W, 2) where N is batch, H/W are spatial
    dimensions, and 2 represents (u, v) components.

    Args:
        flow_field: Flow field supplied to the operation under test.
    """
    expected_message = (
        "flow_field must be a 4D jnp.ndarray with shape (N, H, W, 2), "
        f"got shape {flow_field.shape}."
    )
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        input_check_gen_img_from_flow(
            flow_field=flow_field, parameters=ImageGenerationSpecification()
        )


@pytest.mark.parametrize(
    "image_shape", [(-1, 128), (128, -1), (0, 128), (128, 0), (128.2, 128.2)]
)
def test_invalid_image_shape(image_shape: Any) -> None:
    """Test that non-positive or non-integer image shapes raise a
    ValueError.

    Validation ensures that the output image dimensions are physically
    valid positive integers.

    Args:
        image_shape: Height and width of the output images.
    """
    flow_field = jnp.zeros((1, 128, 128, 2))
    with pytest.raises(
        ValueError,
        match=r"image_shape must be a tuple of two positive integers.",
    ):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            parameters=ImageGenerationSpecification(image_shape=image_shape),
        )


@pytest.mark.parametrize(
    "position_bounds",
    [(-1, 128), (128, -1), (0, 128), (128, 0), (128.2, 128.2)],
)
def test_invalid_position_bounds(position_bounds: Any) -> None:
    """Test that invalid position_bounds raises a ValueError.

    Position bounds define the absolute spatial domain of the flow field
    and must be positive integers.

    Args:
        position_bounds: Position bounds for this test case.
    """
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    img_offset = (0, 0)
    with pytest.raises(
        ValueError,
        match=(
            r"position_bounds must be a tuple of two positive " r"integers."
        ),
    ):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            parameters=ImageGenerationSpecification(
                image_shape=image_shape,
                img_offset=img_offset,
            ),
            position_bounds=position_bounds,
        )


@pytest.mark.parametrize(
    "seeding_density_range, expected_message",
    [
        (
            (-1.0, 1.0),
            (
                "seeding_density_range must be a tuple of two "
                "non-negative numbers."
            ),
        ),
        (
            (0.0, -1.0),
            (
                "seeding_density_range must be a tuple of two "
                "non-negative numbers."
            ),
        ),
        (
            (-0.5, -0.5),
            (
                "seeding_density_range must be a tuple of two "
                "non-negative numbers."
            ),
        ),
        (
            (1.0, 0.5),
            ("seeding_density_range must be in the form \\(min, max\\)."),
        ),
        (
            (0.5, 0.1),
            ("seeding_density_range must be in the form \\(min, max\\)."),
        ),
    ],
)
def test_invalid_seeding_density_range(
    seeding_density_range: Any, expected_message: Any
) -> None:
    """Test various invalid seeding_density_range configurations.

    Verifies that ranges are non-negative, correctly ordered (min <=
    max), and raise descriptive errors when these conditions are
    violated.

    Args:
        seeding_density_range: Seeding density range for this test case.
        expected_message: Expected message.
    """
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(ValueError, match=expected_message):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            parameters=ImageGenerationSpecification(
                image_shape=image_shape,
                seeding_density_range=seeding_density_range,
            ),
        )


@pytest.mark.parametrize("num_images", [-1, 0, 1.5, 2.5])
def test_invalid_num_images(num_images: Any) -> None:
    """Test that non-positive or non-integer batch sizes raise a
    ValueError.

    Validates that the requested number of images to generate is a
    positive integer.

    Args:
        num_images: Num images for this test case.
    """
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(
        ValueError, match=r"batch_size must be a positive integer."
    ):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            parameters=ImageGenerationSpecification(
                image_shape=image_shape,
                batch_size=num_images,
            ),
        )


@pytest.mark.parametrize("img_offset", [(-1, 0), (0, -1), (1, 2, 3)])
def test_invalid_img_offset(img_offset: Any) -> None:
    """Test that invalid image offsets (negative or wrong dimension) raise
    a ValueError.

    The offset must be a non-negative 2-tuple (y, x).

    Args:
        img_offset: Img offset for this test case.
    """
    flow_field = jnp.zeros((1, 128, 128, 2))
    position_bounds = (256, 256)
    image_shape = (128, 128)
    with pytest.raises(
        ValueError,
        match=r"img_offset must be a tuple of two non-negative numbers.",
    ):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            position_bounds=position_bounds,
            parameters=ImageGenerationSpecification(
                image_shape=image_shape,
                img_offset=img_offset,
            ),
        )


@pytest.mark.parametrize("p_hide_img1", [-0.1, 1.1, 1.5, 2.5])
def test_invalid_p_hide_img1(p_hide_img1: Any) -> None:
    """Test that p_hide_img1 values outside [0, 1] raise a ValueError.

    Ensures that the probability of hiding the first frame is a valid
    scalar between 0 and 1.

    Args:
        p_hide_img1: P hide img1 for this test case.
    """
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(
        ValueError, match=r"p_hide_img1 must be between 0 and 1."
    ):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            parameters=ImageGenerationSpecification(
                image_shape=image_shape, p_hide_img1=p_hide_img1
            ),
        )


@pytest.mark.parametrize("p_hide_img2", [-0.1, 1.1, 1.5, 2.5])
def test_invalid_p_hide_img2(p_hide_img2: Any) -> None:
    """Test that p_hide_img2 values outside [0, 1] raise a ValueError.

    Ensures that the probability of hiding the second frame is a valid
    scalar between 0 and 1.

    Args:
        p_hide_img2: P hide img2 for this test case.
    """
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(
        ValueError, match=r"p_hide_img2 must be between 0 and 1."
    ):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            parameters=ImageGenerationSpecification(
                image_shape=image_shape, p_hide_img2=p_hide_img2
            ),
        )


@pytest.mark.parametrize(
    "diameter_ranges, expected_message",
    [
        ([(-1.0, 1.0)], "Each diameter_range must satisfy 0 < min <= max."),
        ([(0.0, -1.0)], "Each diameter_range must satisfy 0 < min <= max."),
        ([(0.5, 0.1)], "Each diameter_range must satisfy 0 < min <= max."),
        (None, "diameter_ranges must be a list of (min, max) tuples."),
        ("invalid", "diameter_ranges must be a list of (min, max) tuples."),
        (
            jnp.array([[1], [2]]),
            "diameter_ranges must be a list of (min, max) tuples.",
        ),
    ],
)
def test_invalid_diameter_range(
    diameter_ranges: Any, expected_message: Any
) -> None:
    """Test that invalid particle diameter ranges raise a ValueError.

    Ensures that diameters are positive, ranges are correctly ordered,
    and the input format is a list of tuples.

    Args:
        diameter_ranges: Diameter ranges for this test case.
        expected_message: Expected message.
    """
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            parameters=ImageGenerationSpecification(
                diameter_ranges=diameter_ranges,
                image_shape=image_shape,
            ),
        )


@pytest.mark.parametrize(
    "intensity_ranges, expected_message",
    [
        ([(-1.0, 1.0)], "Each intensity_range must satisfy 0 < min <= max."),
        ([(0.0, -1.0)], "Each intensity_range must satisfy 0 < min <= max."),
        ([(-0.5, -0.5)], "Each intensity_range must satisfy 0 < min <= max."),
        ([(1.0, 0.5)], "Each intensity_range must satisfy 0 < min <= max."),
        ([(0.5, 0.1)], "Each intensity_range must satisfy 0 < min <= max."),
        (
            jnp.array([[0.5], [0.4]]),
            "intensity_ranges must be a list of (min, max) tuples.",
        ),
    ],
)
def test_invalid_intensity_range(
    intensity_ranges: Any, expected_message: Any
) -> None:
    """Test that invalid particle intensity ranges raise a ValueError.

    Intensity values must be positive and ranges must be correctly
    ordered min <= max.

    Args:
        intensity_ranges: Intensity ranges for this test case.
        expected_message: Expected message.
    """
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            parameters=ImageGenerationSpecification(
                image_shape=image_shape,
                intensity_ranges=intensity_ranges,
            ),
        )


@pytest.mark.parametrize(
    "rho_ranges, expected_message",
    [
        (
            [(-1.1, 1.0)],
            "Each rho_range must satisfy -1 < min <= max < 1.",
        ),
        (
            [(0.0, 1.1)],
            "Each rho_range must satisfy -1 < min <= max < 1.",
        ),
        ([(0.9, 0.5)], "Each rho_range must satisfy -1 < min <= max < 1."),
        ([(0.5, 0.1)], "Each rho_range must satisfy -1 < min <= max < 1."),
        (
            [[0.5], [0.4]],
            "rho_ranges must be a list of (min, max) tuples.",
        ),
    ],
)
def test_invalid_rho_range(rho_ranges: Any, expected_message: Any) -> None:
    """Test that invalid rho ranges (particle correlations) raise a
    ValueError.

    Rho must satisfy -1 < min <= max < 1 to be physically valid for
    Gaussian particle distributions.

    Args:
        rho_ranges: Rho ranges for this test case.
        expected_message: Expected message.
    """
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            parameters=ImageGenerationSpecification(
                image_shape=image_shape,
                rho_ranges=rho_ranges,
            ),
        )


@pytest.mark.parametrize(
    "dt",
    ["invalid_dt", jnp.array([1]), jnp.array([1.0, 2.0]), jnp.array([1, 2, 3])],
)
def test_invalid_dt(dt: Any) -> None:
    """Test that invalid timestep values raise a ValueError.

    The dt parameter must be a positive number to define the temporal
    separation between frames.

    Args:
        dt: Dt for this test case.
    """
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(ValueError, match=r"dt must be a positive number."):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            parameters=ImageGenerationSpecification(
                image_shape=image_shape,
                dt=dt,
            ),
        )


@pytest.mark.parametrize(
    "flow_field_res_x",
    ["a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])],
)
def test_invalid_flow_field_res_x(flow_field_res_x: Any) -> None:
    """Test that invalid spatial resolution in X raises a ValueError.

    Spatial resolution must be a positive scalar to correctly map flow
    field units to pixels.

    Args:
        flow_field_res_x: Flow field res x for this test case.
    """
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(
        ValueError,
        match=("flow_field_res_x must be a positive scalar \\(int or float\\)"),
    ):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            flow_field_res_x=flow_field_res_x,
            parameters=ImageGenerationSpecification(
                image_shape=image_shape,
            ),
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
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(
        ValueError,
        match=("flow_field_res_y must be a positive scalar \\(int or float\\)"),
    ):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            flow_field_res_y=flow_field_res_y,
            parameters=ImageGenerationSpecification(
                image_shape=image_shape,
            ),
        )


@pytest.mark.parametrize(
    "noise_uniform", [-1, "a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])]
)
def test_invalid_noise_uniform(noise_uniform: Any) -> None:
    """Test that invalid noise_uniform raise a ValueError.

    Args:
        noise_uniform: Noise uniform for this test case.
    """
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(
        ValueError,
        match=r"noise_uniform must be a non-negative number.",
    ):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            parameters=ImageGenerationSpecification(
                image_shape=image_shape,
                noise_uniform=noise_uniform,
            ),
        )


@pytest.mark.parametrize(
    "noise_gaussian_mean", ["a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])]
)
def test_invalid_noise_gaussian_mean(noise_gaussian_mean: Any) -> None:
    """Test that invalid noise_gaussian_mean raise a ValueError.

    Args:
        noise_gaussian_mean: Noise gaussian mean for this test case.
    """
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(
        ValueError,
        match=r"noise_gaussian_mean must be a non-negative number.",
    ):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            parameters=ImageGenerationSpecification(
                image_shape=image_shape,
                noise_gaussian_mean=noise_gaussian_mean,
            ),
        )


@pytest.mark.parametrize(
    "noise_gaussian_std", ["a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])]
)
def test_invalid_noise_gaussian_std(noise_gaussian_std: Any) -> None:
    """Test that invalid noise_gaussian_std raise a ValueError.

    Args:
        noise_gaussian_std: Noise gaussian std for this test case.
    """
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(
        ValueError,
        match=r"noise_gaussian_std must be a non-negative number.",
    ):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            parameters=ImageGenerationSpecification(
                image_shape=image_shape,
                noise_gaussian_std=noise_gaussian_std,
            ),
        )


@pytest.mark.parametrize(
    "diameter_var", ["a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])]
)
def test_invalid_diameter_var(diameter_var: Any) -> None:
    """Test that invalid diameter_var raise a ValueError.

    Args:
        diameter_var: Diameter var for this test case.
    """
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(
        ValueError,
        match=r"diameter_var must be a non-negative number.",
    ):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            parameters=ImageGenerationSpecification(
                image_shape=image_shape,
                diameter_var=diameter_var,
            ),
        )


@pytest.mark.parametrize(
    "intensity_var", ["a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])]
)
def test_invalid_intensity_var(intensity_var: Any) -> None:
    """Test that invalid intensity_var raise a ValueError.

    Args:
        intensity_var: Intensity var for this test case.
    """
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(
        ValueError,
        match=r"intensity_var must be a non-negative number.",
    ):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            parameters=ImageGenerationSpecification(
                image_shape=image_shape,
                intensity_var=intensity_var,
            ),
        )


@pytest.mark.parametrize(
    "rho_var", ["a", [1, 2], jnp.array([1, 2]), jnp.array([[1, 2]])]
)
def test_invalid_rho_var(rho_var: Any) -> None:
    """Test that invalid rho_var raise a ValueError.

    Args:
        rho_var: Rho var for this test case.
    """
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(
        ValueError,
        match=r"rho_var must be a non-negative number.",
    ):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            parameters=ImageGenerationSpecification(
                image_shape=image_shape,
                rho_var=rho_var,
            ),
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
    image_shape: tuple[int, int],
    img_offset: tuple[float, float],
    position_bounds: tuple[int, int],
    error_message: str,
) -> None:
    """Test that the image cannot exceed the flow field spatial bounds.

    The requested image size plus its offset must fit within the total
    defined `position_bounds` of the flow field.

    Args:
        image_shape: Height and width of the output images.
        img_offset: Img offset for this test case.
        position_bounds: Position bounds for this test case.
        error_message: Error message for this test case.
    """
    flow_field = jnp.zeros((1, 128, 128, 2))
    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            position_bounds=position_bounds,
            parameters=ImageGenerationSpecification(
                image_shape=image_shape,
                img_offset=img_offset,
            ),
        )


@pytest.mark.parametrize("debug_flag", [False, True])
def test_generate_images_from_flow(
    monkeypatch: pytest.MonkeyPatch, debug_flag: bool
) -> None:
    """Test end-to-end synthetic image generation from a given flow field.

    Verifies that the generated images match the requested shape and
    that the parameter specification is respected. Supports testing both
    the JIT-compiled and debug paths.

    Args:
        monkeypatch: Fixture that restores patched state after the test.
        debug_flag: Debug flag for this test case.
    """

    if debug_flag:
        monkeypatch.setattr(dg, "DEBUG_JIT", debug_flag, raising=True)

    # 1. setup the image parameters
    key = jax.random.PRNGKey(0)
    selected_flow = "horizontal"
    position_bounds = (128, 128)
    image_shape = (128, 128)
    seeding_density_range = (0.001, 0.01)
    img_offset = (0, 0)
    p_hide_img1 = 0.0
    p_hide_img2 = 0.0
    diameter_ranges = [(1, 2)]
    diameter_var = 0
    intensity_ranges = [(50, 250)]
    intensity_var = 0
    rho_ranges = [(-0.2, 0.2)]  # rho cannot be -1 or 1
    rho_var = 0
    dt = 0.1
    noise_uniform = 0.0
    noise_gaussian_mean = 0.0
    noise_gaussian_std = 0.0

    # 2. create a flow field
    flow_field = generate_array_flow_field(
        get_flow_function(selected_flow, image_shape), image_shape
    )
    flow_field = jnp.expand_dims(flow_field, axis=0)

    # 3. apply the flow field to the particles
    images1, images2, _ = dg.generate_images_from_flow(
        key=key,
        flow_field=flow_field,
        position_bounds=position_bounds,
        parameters=ImageGenerationSpecification(
            image_shape=image_shape,
            seeding_density_range=seeding_density_range,
            batch_size=1,
            img_offset=img_offset,
            p_hide_img1=p_hide_img1,
            p_hide_img2=p_hide_img2,
            diameter_ranges=diameter_ranges,
            diameter_var=diameter_var,
            intensity_ranges=intensity_ranges,
            intensity_var=intensity_var,
            rho_ranges=rho_ranges,
            rho_var=rho_var,
            dt=dt,
            noise_uniform=noise_uniform,
            noise_gaussian_mean=noise_gaussian_mean,
            noise_gaussian_std=noise_gaussian_std,
        ),
    )
    img = images1
    img_warped = images2

    # 4. fix the shape of the images
    img = jnp.squeeze(img)
    img_warped = jnp.squeeze(img_warped)

    # 5. check the shape of the images
    assert img.shape == image_shape, (
        f"Image shape mismatch. Expected {image_shape}, got {img.shape}"
    )
    assert img_warped.shape == image_shape, (
        f"Warped image shape mismatch. Expected {image_shape}"
        f", got {img_warped.shape}"
    )

    # an invalid argument should raise a ValueError (the check is in the
    # function)
    if debug_flag:
        with pytest.raises(
            ValueError,
            match=r"image_shape must be a tuple of two positive integers.",
        ):
            dg.generate_images_from_flow(
                key=key,
                flow_field=flow_field,
                position_bounds=position_bounds,
                parameters=ImageGenerationSpecification(
                    image_shape=(-1, -1),
                    seeding_density_range=seeding_density_range,
                    batch_size=1,
                    img_offset=img_offset,
                    p_hide_img1=p_hide_img1,
                    p_hide_img2=p_hide_img2,
                    diameter_ranges=diameter_ranges,
                    diameter_var=diameter_var,
                    intensity_ranges=intensity_ranges,
                    intensity_var=intensity_var,
                    rho_ranges=rho_ranges,
                    rho_var=rho_var,
                    dt=dt,
                    noise_uniform=noise_uniform,
                ),
            )


# skipif is used to skip the test if the user is not connected to the server
@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connected to the server.",
)
@pytest.mark.parametrize("selected_flow", ["horizontal"])
@pytest.mark.parametrize("seeding_density_range", [(0.01, 0.1)])
@pytest.mark.parametrize("max_seeding_density", [0.02])
@pytest.mark.parametrize("num_images", [100])
@pytest.mark.parametrize("image_shape", [(1216, 1936)])
@pytest.mark.parametrize("position_bounds", [(1536, 2048)])
@pytest.mark.parametrize("img_offset", [(160, 56)])
@pytest.mark.parametrize("num_flow_fields", [100])
@pytest.mark.benchmark
def test_speed_generate_images_from_flow(
    selected_flow: str,
    seeding_density_range: tuple[float, float],
    max_seeding_density: float,
    num_images: int,
    image_shape: tuple[int, int],
    position_bounds: tuple[int, int],
    img_offset: tuple[float, float],
    num_flow_fields: int,
) -> None:
    """Benchmark performance of GPU-accelerated image generation using
    `shard_map`.

    Tests generation speed on available NVIDIA devices, ensuring that
    parallelizing flow fields across GPUs yields performance within safe
    timing limits.

    Args:
        selected_flow: Name of the analytical flow used as ground truth.
        seeding_density_range: Seeding density range for this test case.
        max_seeding_density: Max seeding density for this test case.
        num_images: Num images for this test case.
        image_shape: Height and width of the output images.
        position_bounds: Position bounds for this test case.
        img_offset: Img offset for this test case.
        num_flow_fields: Num flow fields for this test case.
    """

    # Name of the axis for the device mesh
    shard_fields = "fields"

    # Check how many GPUs are available
    devices = jax.devices()
    if len(devices) == 3:
        devices = devices[:2]
    elif len(devices) > 4:
        devices = devices[:4]

    num_devices = len(devices)

    # Limit time in seconds (depends on the number of GPUs)
    if num_devices == 1:
        limit_time = 8.5e-2
    elif num_devices == 2:
        limit_time = 4.5e-2
    elif num_devices == 4:
        limit_time = 2.5e-2

    # Setup device mesh
    # We want to shard a key to each device
    # and give different flow fields to each device.
    # The idea is that each device will generate a num_images images
    # and then stack it with the images generated by the other GPUs.
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

    out_specs = (
        PartitionSpec(shard_fields),
        PartitionSpec(shard_fields),
        PartitionSpec(shard_fields),
    )

    # 5. Create the jit function
    jit_generate_images = jax.jit(
        jax.shard_map(
            lambda key, flow: generate_images_from_flow(
                key=key,
                flow_field=flow,
                position_bounds=position_bounds,
                parameters=ImageGenerationSpecification(
                    image_shape=image_shape,
                    img_offset=img_offset,
                    seeding_density_range=seeding_density_range,
                    batch_size=num_images,
                    p_hide_img1=0.0,
                    p_hide_img2=0.0,
                    diameter_ranges=[(1, 2)],
                    diameter_var=0,
                    intensity_ranges=[(80, 100)],
                    intensity_var=0,
                    noise_uniform=0,
                    noise_gaussian_mean=0.0,
                    noise_gaussian_std=0.0,
                    rho_ranges=[(-0.01, 0.01)],
                    rho_var=0,
                ),
            ),
            mesh=mesh,
            in_specs=(PartitionSpec(shard_fields), PartitionSpec(shard_fields)),
            out_specs=out_specs,
        )
    )

    def run_generate_jit() -> None:
        """Run generate jit."""
        imgs1, imgs2, params = jit_generate_images(
            keys_sharded, flow_field_sharded
        )
        seeding_densities = params.seeding_densities
        diameter_ranges = params.diameter_ranges
        intensity_ranges = params.intensity_ranges
        rho_ranges = params.rho_ranges
        imgs1.block_until_ready()
        imgs2.block_until_ready()
        seeding_densities.block_until_ready()
        diameter_ranges.block_until_ready()
        intensity_ranges.block_until_ready()
        rho_ranges.block_until_ready()

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
    assert average_time_jit < limit_time, (
        f"The average time is {average_time_jit}, time limit: {limit_time}"
    )


@pytest.mark.run_explicitly
@pytest.mark.parametrize("warped", [True])
@pytest.mark.parametrize("selected_flow", ["horizontal"])
@pytest.mark.parametrize("seeding_density", [0.06])
@pytest.mark.parametrize("image_shape", [(256, 256), (512, 512)])
@pytest.mark.parametrize("diameter_ranges", [[(1, 2)]])
@pytest.mark.parametrize("diameter_var", [0])
@pytest.mark.parametrize("intensity_ranges", [[(80, 100)]])
@pytest.mark.parametrize("intensity_var", [0])
@pytest.mark.parametrize("dt", [0.1])
@pytest.mark.parametrize(
    "rho_ranges", [[(-0.01, 0.01)]]
)  # rho cannot be -1 or 1
@pytest.mark.parametrize("rho_var", [0])
@pytest.mark.parametrize("noise_uniform", [0.0])
@pytest.mark.parametrize("noise_gaussian_mean", [0.0])
@pytest.mark.parametrize("noise_gaussian_std", [0.0])
@pytest.mark.parametrize("img_offset", [(0, 0)])
def test_img_parameter_combinations(
    warped: jax.Array,
    selected_flow: str,
    seeding_density: float,
    image_shape: tuple[int, int],
    diameter_ranges: list[tuple[float, float]],
    diameter_var: float,
    intensity_ranges: list[tuple[float, float]],
    intensity_var: float,
    dt: float,
    rho_ranges: list[tuple[float, float]],
    rho_var: float,
    noise_uniform: float,
    noise_gaussian_mean: float,
    noise_gaussian_std: float,
    img_offset: tuple[float, float],
) -> None:
    """Test various combinations of parameters by generating and saving
    images.

    This test is intended to be run explicitly (e.g., during visual
    debugging) to verify that specific configurations produce visually
    sane results. It saves the generated frames to the `results/`
    directory.

    Args:
        warped: Warped for this test case.
        selected_flow: Name of the analytical flow used as ground truth.
        seeding_density: Seeding density for this test case.
        image_shape: Height and width of the output images.
        diameter_ranges: Diameter ranges for this test case.
        diameter_var: Diameter var for this test case.
        intensity_ranges: Intensity ranges for this test case.
        intensity_var: Intensity var for this test case.
        dt: Dt for this test case.
        rho_ranges: Rho ranges for this test case.
        rho_var: Rho var for this test case.
        noise_uniform: Noise uniform for this test case.
        noise_gaussian_mean: Noise gaussian mean for this test case.
        noise_gaussian_std: Noise gaussian std for this test case.
        img_offset: Img offset for this test case.
    """

    # 1. setup the image parameters
    key = jax.random.PRNGKey(0)

    os.makedirs("results/images_generated", exist_ok=True)
    file_description = (
        "results/images_generated/"
        + image_shape[0].__str__()
        + "_"
        + seeding_density.__str__()
        + "_"
        + diameter_ranges[0][0].__str__()
        + "-"
        + diameter_ranges[0][1].__str__()
        + "_"
        + intensity_ranges[0][0].__str__()
        + "-"
        + intensity_ranges[0][1].__str__()
        + "_"
        + noise_uniform.__str__()
        + "_"
        + noise_gaussian_mean.__str__()
        + "_"
        + noise_gaussian_std.__str__()
    )

    # 2. create a flow field
    flow_field = generate_array_flow_field(
        get_flow_function(selected_flow, image_shape), image_shape
    )
    flow_field = jnp.expand_dims(flow_field, axis=0)

    # 3. apply the flow field to the particles
    jit_gen = jax.jit(
        lambda key, flow_field: generate_images_from_flow(
            key,
            flow_field,
            position_bounds=image_shape,
            parameters=ImageGenerationSpecification(
                image_shape=image_shape,
                seeding_density_range=(seeding_density, seeding_density),
                batch_size=1,
                img_offset=img_offset,
                p_hide_img1=0,
                p_hide_img2=0,
                diameter_ranges=diameter_ranges,
                diameter_var=diameter_var,
                intensity_ranges=intensity_ranges,
                intensity_var=intensity_var,
                rho_ranges=rho_ranges,
                rho_var=rho_var,
                dt=dt,
                noise_uniform=noise_uniform,
                noise_gaussian_mean=noise_gaussian_mean,
                noise_gaussian_std=noise_gaussian_std,
            ),
        )
    )

    img, img_warped, _ = jit_gen(key, flow_field)

    import matplotlib.pyplot as plt  # noqa: PLC0415 - optional or intentionally lazy import

    # 4. fix the shape of the images
    img = np.squeeze(img)
    img_warped = np.squeeze(img_warped)

    plt.imsave(file_description + "img.png", np.array(img), cmap="gray")
    if warped:
        plt.imsave(
            file_description + "img_warped.png",
            np.array(img_warped),
            cmap="gray",
        )

    # 5. check the shape of the images
    assert img.shape == image_shape, (
        f"Image shape mismatch. Expected {image_shape}, got {img.shape}"
    )
    assert img_warped.shape == image_shape, (
        f"Warped image shape mismatch. Expected {image_shape}"
        f", got {img_warped.shape}"
    )


@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connected to the server.",
)
@pytest.mark.run_explicitly
@pytest.mark.parametrize("selected_flow", ["horizontal"])
@pytest.mark.parametrize("seeding_density_range", [(0.1, 0.1)])
@pytest.mark.parametrize("num_images", [1, 100, 500, 1000, 5000, 10000])
@pytest.mark.parametrize(
    "image_shape",
    [(128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)],
)
@pytest.mark.parametrize("img_offset", [(0, 0)])
@pytest.mark.parametrize("num_flow_fields", [100])
@pytest.mark.parametrize("diameter_ranges", [[(1, 2)]])
@pytest.mark.parametrize("position_bounds", [(128, 128)])
@pytest.mark.parametrize("intensity_ranges", [[(100, 200)]])
@pytest.mark.parametrize("rho_ranges", [[(-0.2, 0.2)]])
@pytest.mark.benchmark
def test_speed_parameter_combinations(
    selected_flow: str,
    seeding_density_range: tuple[float, float],
    num_images: int,
    image_shape: tuple[int, int],
    img_offset: tuple[float, float],
    num_flow_fields: int,
    position_bounds: tuple[int, int],
    diameter_ranges: list[tuple[float, float]],
    intensity_ranges: list[tuple[float, float]],
    rho_ranges: list[tuple[float, float]],
) -> None:
    """Deep benchmark of generation speed across wide parameter ranges.

    Systematically measures throughput for various image sizes and batch
    sizes. Intended for profiling and hardware comparison; contains a
    hard `assert False` to output timing results during test execution.

    Args:
        selected_flow: Name of the analytical flow used as ground truth.
        seeding_density_range: Seeding density range for this test case.
        num_images: Num images for this test case.
        image_shape: Height and width of the output images.
        img_offset: Img offset for this test case.
        num_flow_fields: Num flow fields for this test case.
        position_bounds: Position bounds for this test case.
        diameter_ranges: Diameter ranges for this test case.
        intensity_ranges: Intensity ranges for this test case.
        rho_ranges: Rho ranges for this test case.

    Raises:
        AssertionError: When the simulated failure is triggered.
    """

    # Name of the axis for the device mesh
    shard_fields = "fields"

    # Check how many GPUs are available
    devices = jax.devices()
    if len(devices) == 3:
        devices = devices[:2]
    elif len(devices) > 4:
        devices = devices[:4]

    num_devices = len(devices)

    # Setup device mesh
    # We want to shard a key to each device
    # and give different flow fields to each device.
    # The idea is that each device will generate a num_images images
    # and then stack it with the images generated by the other GPUs.
    mesh = Mesh(devices, axis_names=(shard_fields))

    # 1. Generate key
    key = jax.random.PRNGKey(0)

    # 2. create a flow field
    flow_field = generate_array_flow_field(
        get_flow_function(selected_flow, image_shape), image_shape
    )
    flow_field = jnp.expand_dims(flow_field, axis=0)
    flow_field = jnp.repeat(flow_field, num_flow_fields, axis=0)

    # 3. Shard the flow field
    flow_field_sharded = jax.device_put(
        flow_field, NamedSharding(mesh, PartitionSpec(shard_fields))
    )

    # 4. Setup the random keys
    keys = jax.random.split(key, num_devices)
    keys = jnp.stack(keys)
    keys_sharded = jax.device_put(
        keys, NamedSharding(mesh, PartitionSpec(shard_fields))
    )
    jax.block_until_ready(keys_sharded)

    # 5. Create the jit function
    jit_generate_images = jax.jit(
        jax.shard_map(
            lambda key, flow: generate_images_from_flow(
                key=key,
                flow_field=flow,
                position_bounds=position_bounds,
                parameters=ImageGenerationSpecification(
                    image_shape=image_shape,
                    img_offset=img_offset,
                    seeding_density_range=seeding_density_range,
                    batch_size=num_images,
                    diameter_ranges=diameter_ranges,
                    intensity_ranges=intensity_ranges,
                    rho_ranges=rho_ranges,
                ),
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

    def run_generate_jit() -> None:
        """Run generate jit."""
        imgs1, imgs2, params = jit_generate_images(
            keys_sharded, flow_field_sharded
        )
        seeding_densities = params.seeding_densities
        diameter_ranges = params.diameter_ranges
        intensity_ranges = params.intensity_ranges
        rho_ranges = params.rho_ranges
        imgs1.block_until_ready()
        imgs2.block_until_ready()
        seeding_densities.block_until_ready()
        diameter_ranges.block_until_ready()
        intensity_ranges.block_until_ready()
        rho_ranges.block_until_ready()

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

    # Print the average time for evaluation purposes
    raise AssertionError(f"The average time is {average_time_jit}")
