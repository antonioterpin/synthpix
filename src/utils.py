"""Utility functions for the vision module."""

import logging
import os
import signal
from typing import Union

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from tqdm import tqdm

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
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def compute_image_scaled_height(
    target_width: int, image_width: int, image_height: int
) -> int:
    """Computes the height of an image given a target width keeping the aspect ratio.

    Args:
        target_width (int): The target width.
        image_width (int): The width of the image.
        image_height (int): The height of the image.

    Returns:
        int: The scaled height.
    """
    if target_width <= 0 or image_width <= 0 or image_height <= 0:
        raise ValueError("All dimensions must be positive integers.")
    return int(image_height * target_width / image_width)


def particles_per_pixel(image: jnp.ndarray, threshold: float = 0.1) -> float:
    """Estimates the number of particles per pixel in the image.

    Args:
        image: jnp.ndarray
            The input image of shape (H, W, 1).
        threshold: float
            The threshold to apply to the image.

    Returns:
        float: The estimated density.
    """
    # Simple, fast, metric for particle density is the fraction of pixels
    # above a threshold
    if not (0 <= threshold <= 255):
        raise ValueError("threshold must be a float in the range [0, 1].")
    return float(jnp.sum(image > threshold) / jnp.prod(image.size))


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


class GracefulShutdown:
    """A context manager for graceful shutdowns."""

    stop = False

    def __enter__(self):
        """Register the signal handler."""

        def handle_signal(signum, frame):
            self.stop = True

        signal.signal(signal.SIGINT, handle_signal)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Unregister the signal handler."""
        pass


def generate_array_flow_field(flow_f, grid_shape: tuple[int, int]) -> jnp.ndarray:
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


def calculate_min_and_max_speeds(file_list: list[str]) -> dict[str, float]:
    """Calculate the missing speeds for a list of files.

    Args:
        file_list: list[str]
            The list of files.

    Returns:
        dict[str, float]: A dictionary containing the minimum and maximum speeds
            in the x and y directions with keys:
            - "min_speed_x"
            - "max_speed_x"
            - "min_speed_y"
            - "max_speed_y"
    """
    # Input validation
    if not file_list:
        raise ValueError("The file_list must not be empty.")

    for file_path in file_list:
        if not isinstance(file_path, str):
            raise ValueError("All file paths must be strings.")
        if not os.path.isfile(file_path):
            raise ValueError(f"File {file_path} does not exist.")
        if not file_path.endswith(".h5"):
            raise ValueError(f"File {file_path} is not a .h5 file.")

    # Initialize lists to store the min and max speeds for each file
    max_speeds_x = []
    max_speeds_y = []
    min_speeds_x = []
    min_speeds_y = []
    # Wrap the file list with tqdm for a loading bar
    for file in tqdm(file_list, desc="Processing files"):
        with h5py.File(file, "r") as f:
            # Read the file
            dataset_name = list(f.keys())[0]
            data = f[dataset_name][:]

            # Find the min and max speeds along each axis
            max_speeds_x.append(np.max(data[:, :, :, 0]))
            max_speeds_y.append(
                np.max(data[:, :, :, 2])
            )  # TODO: set to 1, left to 2 for test
            min_speeds_x.append(np.min(data[:, :, :, 0]))
            min_speeds_y.append(np.min(data[:, :, :, 2]))

    min_speed_x = np.min(min_speeds_x)
    max_speed_x = np.max(max_speeds_x)
    min_speed_y = np.min(min_speeds_y)
    max_speed_y = np.max(max_speeds_y)

    return {
        "min_speed_x": min_speed_x,
        "max_speed_x": max_speed_x,
        "min_speed_y": min_speed_y,
        "max_speed_y": max_speed_y,
    }


def update_config_file(config_path: str, updated_values: dict):
    """Update the YAML configuration file with new values.

    Args:
        config_path: str
            Path to the configuration file.
        updated_values: dict
            Dictionary containing the updated values.
    """
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    # Convert all values in updated_values to standard Python types
    updated_values = {key: float(value) for key, value in updated_values.items()}

    config_data.update(updated_values)

    with open(config_path, "w") as file:
        yaml.safe_dump(config_data, file)
