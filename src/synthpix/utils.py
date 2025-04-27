"""Utility functions for the vision module."""

import logging
import os
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from ruamel.yaml import YAML

DEBUG = False
DEBUG_JIT = False


# Create a logger instance
logger = logging.getLogger(__name__)

# Configure the logging format
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="[%(levelname)s][%(asctime)s][%(filename)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def is_int(val: Union[int, float]) -> bool:
    """Check if a value is an integer.

    Args:
        val (Union[int, float]): The value to check.

    Returns:
        bool: True if the value is an integer, False otherwise.
    """
    if isinstance(val, int):
        return True
    if isinstance(val, float):
        if abs(val - int(val)) < 1e-6:
            return True
    return False


def load_configuration(file_path: str):
    """Load YAML configuration from file."""
    yaml = YAML(typ="safe", pure=True)
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.load(file)


def bilinear_interpolate(
    image: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray
) -> jnp.ndarray:
    """Perform bilinear interpolation of `image` at floating-point pixel coordinates.

    Args:
        image: jnp.ndarray
            2D image to sample from, of shape (H, W).
        x: jnp.ndarray
            2D array of floating-point x-coordinates
        y: jnp.ndarray
            2D array of floating-point y-coordinates

    Returns:
        jnp.ndarray: Interpolated intensities at each (y, x) location, of shape (H, W).
    """
    H, W = image.shape

    # Floor of x, y
    x0 = jnp.floor(x).astype(int)
    y0 = jnp.floor(y).astype(int)
    # Ceiling (neighbor) of x, y
    x1 = jnp.ceil(x).astype(int)
    y1 = jnp.ceil(y).astype(int)

    # Clamp to image boundaries
    # Note: in this way, the positions need to be within the image boundaries
    x0 = jnp.clip(x0, 0, W - 1)
    x1 = jnp.clip(x1, 0, W - 1)
    y0 = jnp.clip(y0, 0, H - 1)
    y1 = jnp.clip(y1, 0, H - 1)

    # Compute interpolation weights
    alpha_x = x - jnp.floor(x)
    alpha_y = y - jnp.floor(y)

    # Gather intensities from the four corners
    Ia = image[y0, x0]  # top-left
    Ib = image[y0, x1]  # top-right
    Ic = image[y1, x0]  # bottom-left
    Id = image[y1, x1]  # bottom-right

    # Bilinear interpolation formula
    wa = (1.0 - alpha_x) * (1.0 - alpha_y)
    wb = alpha_x * (1.0 - alpha_y)
    wc = (1.0 - alpha_x) * alpha_y
    wd = alpha_x * alpha_y

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def trilinear_interpolate(
    volume: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray
) -> jnp.ndarray:
    """Perform trilinear interpolation of `volume` at floating-point pixel coordinates.

    Args:
        volume: jnp.ndarray
            3D volume to sample from, of shape (D, H, W).
        x: jnp.ndarray
            Array of floating-point x-coordinates.
        y: jnp.ndarray
            Array of floating-point y-coordinates.
        z: jnp.ndarray
            Array of floating-point z-coordinates.

    Returns:
        jnp.ndarray: Interpolated intensities at each (z, y, x) location.
    """
    D, H, W = volume.shape

    # Floor and ceil indices for each coordinate
    x0 = jnp.floor(x).astype(int)
    y0 = jnp.floor(y).astype(int)
    z0 = jnp.floor(z).astype(int)

    x1 = jnp.ceil(x).astype(int)
    y1 = jnp.ceil(y).astype(int)
    z1 = jnp.ceil(z).astype(int)

    # Clamp indices to be within volume boundaries
    x0 = jnp.clip(x0, 0, W - 1)
    x1 = jnp.clip(x1, 0, W - 1)
    y0 = jnp.clip(y0, 0, H - 1)
    y1 = jnp.clip(y1, 0, H - 1)
    z0 = jnp.clip(z0, 0, D - 1)
    z1 = jnp.clip(z1, 0, D - 1)

    # Compute interpolation weights for each axis
    alpha_x = x - jnp.floor(x)
    alpha_y = y - jnp.floor(y)
    alpha_z = z - jnp.floor(z)

    # Retrieve intensities from the eight corners of the cube
    Ia = volume[z0, y0, x0]
    Ib = volume[z0, y0, x1]
    Ic = volume[z0, y1, x0]
    Id = volume[z0, y1, x1]
    Ie = volume[z1, y0, x0]
    If = volume[z1, y0, x1]
    Ig = volume[z1, y1, x0]
    Ih = volume[z1, y1, x1]

    # Compute weights for each corner
    wa = (1.0 - alpha_x) * (1.0 - alpha_y) * (1.0 - alpha_z)
    wb = alpha_x * (1.0 - alpha_y) * (1.0 - alpha_z)
    wc = (1.0 - alpha_x) * alpha_y * (1.0 - alpha_z)
    wd = alpha_x * alpha_y * (1.0 - alpha_z)
    we = (1.0 - alpha_x) * (1.0 - alpha_y) * alpha_z
    wf = alpha_x * (1.0 - alpha_y) * alpha_z
    wg = (1.0 - alpha_x) * alpha_y * alpha_z
    wh = alpha_x * alpha_y * alpha_z

    # Compute the weighted sum of the corner intensities
    return Ia * wa + Ib * wb + Ic * wc + Id * wd + Ie * we + If * wf + Ig * wg + Ih * wh


def generate_array_flow_field(
    flow_f, grid_shape: tuple[int, int] = (128, 128)
) -> jnp.ndarray:
    """Generate a array flow field from a flow field function.

    Args:
        flow_f:
            The flow field function.
        grid_shape: tuple[int, int]
            The shape of the grid.

    Returns:
        arr: jnp.ndarray
            The array flow field.
    """
    # Get the image shape
    H, W = grid_shape
    # Create pixel coordinate grids: y in [0..H-1], x in [0..W-1]
    rows = jnp.arange(H)
    cols = jnp.arange(W)

    # vmap over both axes, and apply the flow function at time t=1
    arr = jax.vmap(lambda i: jax.vmap(lambda j: jnp.array(flow_f(1, i, j)))(cols))(rows)

    return arr


def visualize_and_save(name, image1, image2, flow_field, output_dir="output_images"):
    """Visualizes and saves a specified number of images from a batch.

    Args:
        name (str): The name of the batch.
        image1 (jnp.ndarray): The first image to visualize.
        image2 (jnp.ndarray): The second image to visualize.
        flow_field (jnp.ndarray): The flow field to visualize.
        output_dir (str): Directory to save the images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract flow field
    flow_x = flow_field[..., 0]
    flow_y = flow_field[..., 1]
    # Create a grid for the quiver plot
    y, x = np.mgrid[0 : flow_x.shape[0], 0 : flow_x.shape[1]]

    # Save individual images and flow field
    plt.imsave(os.path.join(output_dir, f"{name}_image1.png"), image1, cmap="gray")
    plt.imsave(os.path.join(output_dir, f"{name}_image2.png"), image2, cmap="gray")

    # Save the quiver plot as a separate image
    quiver_fig, quiver_ax = plt.subplots(figsize=(7, 7))
    step = 1
    quiver_ax.quiver(
        x[::step, ::step],
        y[::step, ::step],
        flow_x[::step, ::step],
        flow_y[::step, ::step],
        pivot="mid",
        color="blue",
    )
    quiver_ax.set_aspect("equal")
    quiver_fig.savefig(os.path.join(output_dir, f"{name}_quiver.png"))
    plt.close(quiver_fig)

    logger.info(f"Saved images for {name} to {output_dir}.")


def interp_channel(
    channel, row_floor, row_ceil, col_floor, col_ceil, row_lerp, col_lerp
):
    """Perform bilinear interpolation on a single 2D channel."""
    I00 = channel[row_floor[:, None], col_floor[None, :]]
    I01 = channel[row_floor[:, None], col_ceil[None, :]]
    I10 = channel[row_ceil[:, None], col_floor[None, :]]
    I11 = channel[row_ceil[:, None], col_ceil[None, :]]

    top = I00 * (1 - col_lerp) + I01 * col_lerp
    bottom = I10 * (1 - col_lerp) + I11 * col_lerp
    return top * (1 - row_lerp) + bottom * row_lerp


def flow_field_adapter(
    flow_fields: jnp.ndarray,
    new_flow_field_shape: Tuple[int, int] = (256, 256),
    image_shape: Tuple[int, int] = (256, 256),
    img_offset: Tuple[int, int] = (0, 0),
    resolution: float = 1.0,
    res_x: float = 1.0,
    res_y: float = 1.0,
    position_bounds: Tuple[int, int] = (256, 256),
    position_bounds_offset: Tuple[int, int] = (0, 0),
    batch_size: int = 1,
    output_units: str = "pixels",
    dt: float = 1.0,
    zero_padding: Tuple[int, int] = (0, 0),
):
    """Adapts a batch of flow fields to a new shape and resolution.

    Args:
        flow_field: jnp.ndarray
            The original flow field batch to be adapted.
        new_flow_field_shape: Tuple[int, int]
            The desired shape of the new flow fields.
        image_shape: Tuple[int, int]
            The shape of the images.
        img_offset: Tuple[int, int]
            The offset of the images from the position bounds in pixels.
        resolution: float
            Resolution of the images in pixels per unit length.
        res_x: float
            Flow field resolution in the x direction [grid steps/length measure units].
        res_y: float
            Flow field resolution in the y direction [grid steps/length measure units].
        position_bounds: Tuple[int, int]
            The bounds of the flow field in the x and y directions.
        position_bounds_offset: Tuple[int, int]
            The offset of the position bounds in length measure units.
        batch_size: int
            The desired batch size of the output flow fields.
        output_units: str
            The units of the output flow fields.
            Can be "pixels" or "measure units per second".
        dt: float
            The time step for the flow field adaptation.
        zero_padding: Tuple[int, int]
            The amount of zero-padding to apply to the
            top and left edges of the flow field.

    Returns:
        jnp.ndarray: The adapted flow fields of shape (batch_size, new_h, new_w, 2).
        jnp.ndarray: The cropped flow field region of position bounds.
    """
    new_h, new_w = new_flow_field_shape

    def process_single(flow):
        # Apply zero-padding
        pad_y, pad_x = zero_padding
        flow = jnp.pad(
            flow,
            pad_width=((pad_y, 0), (pad_x, 0), (0, 0)),
            mode="edge",
        )

        # Crop by position bounds
        y_start = int(position_bounds_offset[0] * res_y)
        y_end = y_start + int(position_bounds[0] / resolution * res_y)
        x_start = int(position_bounds_offset[1] * res_x)
        x_end = x_start + int(position_bounds[1] / resolution * res_x)
        flow_position_bounds = flow[y_start:y_end, x_start:x_end, :]

        # Resize position bounds to new shape
        alpha1 = new_h / image_shape[0]
        alpha2 = new_w / image_shape[1]
        position_bounds_resized = (
            position_bounds[0] * alpha1,
            position_bounds[1] * alpha2,
        )
        flow_position_bounds_resized = jax.image.resize(
            flow_position_bounds,
            shape=(int(position_bounds_resized[0]), int(position_bounds_resized[1]), 2),
            method="linear",
        )

        # Crop to image offset
        y_img_start = int(img_offset[0] * alpha1)
        y_img_end = y_img_start + new_h
        x_img_start = int(img_offset[1] * alpha2)
        x_img_end = x_img_start + new_w
        flow_resized = flow_position_bounds_resized[
            y_img_start:y_img_end, x_img_start:x_img_end, :
        ]

        if output_units == "pixels":
            flow_resized = flow_resized.at[..., 0].multiply(resolution * dt)
            flow_resized = flow_resized.at[..., 1].multiply(resolution * dt)

        return flow_resized, flow_position_bounds

    adapted_flows, flow_bounds = jax.vmap(process_single)(flow_fields)

    n = adapted_flows.shape[0]
    repeats = (batch_size + n - 1) // n
    tiled = jnp.tile(adapted_flows, (repeats, 1, 1, 1))

    return tiled[:batch_size, ...], flow_bounds


def input_check_flow_field_adapter(
    flow_field: jnp.ndarray,
    new_flow_field_shape: Tuple[int, int],
    image_shape: Tuple[int, int],
    img_offset: Tuple[int, int],
    resolution: float,
    res_x: float,
    res_y: float,
    position_bounds: Tuple[int, int],
    position_bounds_offset: Tuple[int, int],
    batch_size: int,
    output_units: str,
    dt: float,
    zero_padding: Tuple[int, int],
):
    """Checks the input arguments of the flow field adapter function.

    Args:
        flow_field: jnp.ndarray
            The original flow field batch to be adapted.
        new_flow_field_shape: Tuple[int, int]
            The desired shape of the new flow fields.
        image_shape: Tuple[int, int]
            The shape of the images.
        img_offset: Tuple[int, int]
            The offset of the images.
        resolution: float
            Resolution of the images in pixels per unit length.
        res_x: float
            Flow field resolution in the x direction [grid steps/length measure units].
        res_y: float
            Flow field resolution in the y direction [grid steps/length measure units].
        position_bounds: Tuple[int, int]
            The bounds of the flow field in the x and y directions.
        position_bounds_offset: Tuple[int, int]
            The offset of the flow field in the x and y directions.
        batch_size: int
            The desired batch size of the output flow fields.
        output_units: str
            The units of the output flow fields.
            Can be "pixels" or "measure units per second".
        dt: float
            The time step for the flow field adaptation.
        zero_padding: Tuple[int, int]
            The amount of zero-padding to apply to the
            top and left edges of the flow field.
    """
    if not isinstance(flow_field, jnp.ndarray):
        raise ValueError("flow_field must be a jnp.ndarray.")
    if flow_field.ndim != 4:
        raise ValueError("flow_field must be a 4D jnp.ndarray with shape (N, H, W, 2).")
    if flow_field.shape[-1] != 2:
        raise ValueError(
            "flow_field must have shape (N, H, W, 2) in the last dimension."
        )

    if not isinstance(new_flow_field_shape, tuple) or len(new_flow_field_shape) != 2:
        raise ValueError(
            "new_flow_field_shape must be a tuple of two positive integers."
        )
    if not all(isinstance(s, int) and s > 0 for s in new_flow_field_shape):
        raise ValueError("new_flow_field_shape must contain two positive integers.")

    if not isinstance(image_shape, tuple) or len(image_shape) != 2:
        raise ValueError("image_shape must be a tuple of two positive integers.")
    if not all(isinstance(s, int) and s > 0 for s in image_shape):
        raise ValueError("image_shape must contain two positive integers.")

    if not isinstance(img_offset, tuple) or len(img_offset) != 2:
        raise ValueError("img_offset must be a tuple of two non-negative numbers.")
    if not all(isinstance(s, (int, float)) and s >= 0 for s in img_offset):
        raise ValueError("img_offset must contain two non-negative numbers.")

    if not isinstance(resolution, (int, float)) or resolution <= 0:
        raise ValueError("resolution must be a positive number.")

    if not isinstance(res_x, (int, float)) or res_x <= 0:
        raise ValueError("res_x must be a positive number.")

    if not isinstance(res_y, (int, float)) or res_y <= 0:
        raise ValueError("res_y must be a positive number.")

    if not isinstance(position_bounds, tuple) or len(position_bounds) != 2:
        raise ValueError("position_bounds must be a tuple of two positive numbers.")
    if not all(isinstance(s, (int, float)) and s > 0 for s in position_bounds):
        raise ValueError("position_bounds must contain two positive numbers.")

    if (
        not isinstance(position_bounds_offset, tuple)
        or len(position_bounds_offset) != 2
    ):
        raise ValueError(
            "position_bounds_offset must be a tuple of two non-negative numbers."
        )
    if not all(isinstance(s, (int, float)) and s >= 0 for s in position_bounds_offset):
        raise ValueError(
            "position_bounds_offset must contain two non-negative numbers."
        )

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    if not isinstance(output_units, str) or output_units not in [
        "pixels",
        "measure units per second",
    ]:
        raise ValueError(
            "output_units must be either 'pixels' or 'measure units per second'."
        )

    if not isinstance(dt, (int, float)) or dt <= 0:
        raise ValueError("dt must be a positive number.")

    if (
        not isinstance(zero_padding, tuple)
        or len(zero_padding) != 2
        or not all(isinstance(s, int) and s >= 0 for s in zero_padding)
    ):
        raise ValueError("zero_padding must be a tuple of two non-negative integers.")
