"""Utility functions for the vision module."""

import logging
import signal
from typing import Union

import jax.numpy as jnp
import yaml

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s][%(filename)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # filename='app.log',
)

logger = logging


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
        image (jnp.ndarray): The input image of shape (H, W, 1).
        threshold (float): The threshold to apply to the image.

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
        image (jnp.ndarray): 2D image to sample from, of shape (H, W).
        x (jnp.ndarray): 2D array of floating-point x-coordinates
        y (jnp.ndarray): 2D array of floating-point y-coordinates

    Returns:
        jnp.ndarray: Interpolated intensities at each (y, x) location, of shape (H, W).
    """
    H, W = image.shape

    # Floor of x, y
    x0 = jnp.floor(x).astype(int)
    y0 = jnp.floor(y).astype(int)
    # Ceiling (neighbor) of x, y
    x1 = x0 + 1
    y1 = y0 + 1

    # Clamp to image boundaries
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
