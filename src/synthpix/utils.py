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


def flow_field_adapter(
    flow_field: jnp.ndarray, new_flow_field_shape: Tuple[int, int] = (256, 256)
):
    """Adapter to convert flow field to one with a different resolution.

    Args:
        flow_field: jnp.ndarray
            The original flow field to be adapted.
        new_flow_field_shape: Tuple[int, int]
            The desired shape of the new flow field.

    Returns:
        jnp.ndarray: The adapted flow field with the new shape.
    """
    original_shape = flow_field.shape[:2]

    # Create a 2D grid of coordinates for the new shape
    x = jnp.linspace(0, original_shape[1] - 1, new_flow_field_shape[1])
    y = jnp.linspace(0, original_shape[0] - 1, new_flow_field_shape[0])
    x_new, y_new = jnp.meshgrid(x, y)

    # Vectorize over the columns
    interp_over_cols_x = jax.vmap(
        lambda x_coord, y_coord: bilinear_interpolate(
            flow_field[..., 0],
            x_coord,
            y_coord,
        ),
        in_axes=(0, 0),
    )

    # Now vectorize over the rows
    new_flow_field_x = jax.vmap(
        lambda xs, ys: interp_over_cols_x(xs, ys), in_axes=(0, 0)
    )(x_new, y_new)

    # Repeat for the second channel
    interp_over_cols_y = jax.vmap(
        lambda x_coord, y_coord: bilinear_interpolate(
            flow_field[..., 1],
            x_coord,
            y_coord,
        ),
        in_axes=(0, 0),
    )
    new_flow_field_y = jax.vmap(
        lambda xs, ys: interp_over_cols_y(xs, ys), in_axes=(0, 0)
    )(x_new, y_new)

    # Stack the two interpolated channels along the last dimension
    new_flow_field = jnp.stack([new_flow_field_x, new_flow_field_y], axis=-1)
    return new_flow_field


def input_check_flow_field_adapter(
    flow_field: jnp.ndarray, new_flow_field_shape: Tuple[int, int] = (256, 256)
):
    """Checks the input arguments of the flow field adapter function.

    Args:
        flow_field: jnp.ndarray
            The original flow field to be adapted.
        new_flow_field_shape: Tuple[int, int]
            The desired shape of the new flow field.
    """
    if (
        not isinstance(flow_field, jnp.ndarray)
        or len(flow_field.shape) != 3
        or flow_field.shape[2] != 2
    ):
        raise ValueError("Flow_field must be a 3D jnp.ndarray with shape (H, W, 2).")
    if (
        not isinstance(new_flow_field_shape, tuple)
        or len(new_flow_field_shape) != 2
        or not all(isinstance(s, int) and s > 0 for s in new_flow_field_shape)
    ):
        raise ValueError(
            "new_flow_field_shape must be a tuple of two positive integers."
        )
