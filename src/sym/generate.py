"""Module to generate synthetic images for testing and debugging."""

from typing import Tuple

import jax
import jax.numpy as jnp

from src.utils import is_int


def gaussian_2d(
    x: jnp.ndarray, y: jnp.ndarray, x0: float, y0: float, sigma: float, amplitude: float
) -> jnp.ndarray:
    """Generate a 2D Gaussian function.

    Args:
        x (jnp.ndarray): 2D coordinate grid for x-axis.
        y (jnp.ndarray): 2D coordinate grid for y-axis.
        x0 (float): Center position of the Gaussian on the x-axis.
        y0 (float): Center position of the Gaussian on the y-axis.
        sigma (float): Standard deviation of the Gaussian.
        amplitude (float): Peak intensity (I0).

    Returns:
        jnp.ndarray: 2D array representing the Gaussian function.
    """
    return amplitude * jnp.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))


def add_noise_to_image(
    key: jax.random.PRNGKey, image: jnp.ndarray, background_level: float = 5.0
):
    """Add noise to an image.

    Args:
        key (jax.random.PRNGKey): Random key for reproducibility.
        image (jnp.ndarray): Input image.
        background_level (float): Constant background level added to the image.

    Returns:
        jnp.ndarray: Noisy image.
    """
    return jnp.clip(
        image
        + jax.random.uniform(key, shape=image.shape, minval=0, maxval=background_level),
        min=0,
        max=255,
    )


def generate_synthetic_particle_image(
    key: jax.random.PRNGKey,
    image_shape: Tuple[int, int] = (256, 256),
    seeding_density: float = 0.05,
    diameter_range: Tuple[float, float] = (0.1, 1.0),
    intensity_range: Tuple[float, float] = (50, 200),
) -> jnp.ndarray:
    """Generate a synthetic particle image using 2D Gaussians.

    Args:
        key: jax.random.PRNGKey
            Random key for reproducibility.
        image_shape: Tuple[int, int]
            (height, width) of the output image.
        seeding_density: float
            Number of particles per pixel. E.g., 0.05 means ~ 5% of the total
            pixels will be the number of particles.
        diameter_range: Tuple[float, float]
            Minimum and maximum particle diameter in pixels.
        intensity_range: Tuple[float, float]
            Minimum and maximum peak intensity (I0).

    Returns:
        jnp.ndarray: Synthetic particle image of shape `image_shape`.
    """
    # Argument checks using exceptions instead of asserts
    if (
        len(image_shape) != 2
        or not all(s > 0 for s in image_shape)
        or not all(is_int(s) for s in image_shape)
    ):
        raise ValueError("image_shape must be a tuple of two positive integers.")
    if seeding_density <= 0 or seeding_density > 1:
        raise ValueError("seeding_density must be positive.")
    if len(diameter_range) != 2 or not all(d > 0 for d in diameter_range):
        raise ValueError("diameter_range must be a tuple of two positive floats.")
    if len(intensity_range) != 2 or not all(0 <= i <= 255 for i in intensity_range):
        raise ValueError(
            "intensity_range must be a tuple of two floats in the range [0, 255]."
        )

    # 1. Determine how many particles we expect
    height, width = image_shape
    num_pixels = height * width
    num_particles = int(num_pixels * seeding_density)

    # 2. Create coordinate grids
    y_grid = jnp.arange(height)
    x_grid = jnp.arange(width)
    X, Y = jnp.meshgrid(x_grid, y_grid)

    # 3. Sample random parameters
    key_x, key_y, key_d, key_i = jax.random.split(key, 4)

    # Particle center positions
    x0s = jax.random.uniform(key_x, shape=(num_particles,), minval=0, maxval=width)
    y0s = jax.random.uniform(key_y, shape=(num_particles,), minval=0, maxval=height)

    # Diameters in the specified range, then convert to sigma = diameter / 2
    diameters = jax.random.uniform(
        key_d,
        shape=(num_particles,),
        minval=diameter_range[0],
        maxval=diameter_range[1],
    )
    sigmas = diameters / 2.0

    # Peak intensities
    intensities = jax.random.uniform(
        key_i,
        shape=(num_particles,),
        minval=intensity_range[0],
        maxval=intensity_range[1],
    )

    # 4. Accumulate each particle's contribution using jax.vmap for parallel computation
    image = jnp.zeros(image_shape)

    def particle_image(x0, y0, sigma, amp):
        return gaussian_2d(X, Y, x0, y0, sigma, amp)

    particles = jax.vmap(particle_image)(x0s, y0s, sigmas, intensities)
    image = image + jnp.sum(particles, axis=0)

    return jnp.clip(image, min=0, max=255)
