"""Module to generate synthetic images for testing and debugging."""

from typing import Tuple

import jax
import jax.numpy as jnp

from .utils import is_int


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


def gaussian_2d_correlated(
    x: jnp.ndarray,
    y: jnp.ndarray,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    rho: float,
    amplitude: float,
) -> jnp.ndarray:
    """Generate a 2D Gaussian function.

    Args:
        x (jnp.ndarray): 2D coordinate grid for x-axis.
        y (jnp.ndarray): 2D coordinate grid for y-axis.
        x0 (float): Center position of the Gaussian on the x-axis.
        y0 (float): Center position of the Gaussian on the y-axis.
        sigma_x (float): Standard deviation of the Gaussian on the x-axis.
        sigma_y (float): Standard deviation of the Gaussian on the y-axis.
        rho (float): Correlation coefficient between x and y.
        amplitude (float): Peak intensity (I0).

    Returns:
        jnp.ndarray: 2D array representing the Gaussian function.
    """
    x_shifted = x - x0
    y_shifted = y - y0

    # Inverse of the covariance matrix
    one_minus_rho2 = 1.0 - rho**2
    z = (
        (x_shifted**2 / sigma_x**2)
        + (y_shifted**2 / sigma_y**2)
        - (2 * rho * x_shifted * y_shifted) / (sigma_x * sigma_y)
    )

    exponent = -z / (2 * one_minus_rho2)

    return amplitude * jnp.exp(exponent)


def add_noise_to_image(
    key: jax.random.PRNGKey,
    image: jnp.ndarray,
    noise_uniform: float = 0.0,
    noise_gaussian_mean: float = 0.0,
    noise_gaussian_std: float = 0.0,
):
    """Add noise to an image.

    Args:
        key (jax.random.PRNGKey): Random key for reproducibility.
        image (jnp.ndarray): Input image.
        noise_uniform (float): Maximum amplitude of the uniform noise to add.
        noise_gaussian_mean (float): Mean of the Gaussian noise to add.
        noise_gaussian_std (float): Standard deviation of the Gaussian noise to add.

    Returns:
        jnp.ndarray: Noisy image.
    """
    uniform_key, gaussian_key = jax.random.split(key)

    # Add uniform noise
    image = jnp.clip(
        image
        + jax.random.uniform(
            uniform_key, shape=image.shape, minval=0, maxval=noise_uniform
        ),
        min=0,
        max=255,
    )
    # Add Gaussian noise
    image += noise_gaussian_mean
    image += jax.random.normal(gaussian_key, shape=image.shape) * noise_gaussian_std

    # Clip the final image to valid range
    return jnp.clip(image, 0, 255)


def img_gen_from_density(
    key: jax.random.PRNGKey,
    image_shape: Tuple[int, int] = (256, 256),
    seeding_density: float = 0.05,
    diameter_range: Tuple[float, float] = (0.1, 1.0),
    intensity_range: Tuple[float, float] = (50, 200),
    particle_positions: jnp.ndarray = None,
) -> jnp.ndarray:
    """Generate a synthetic particle image using 2D Gaussians from seeding density.

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
        particle_positions: jnp.ndarray
            Optional array of particle positions (x, y) in pixels.

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
    if seeding_density <= 0 or seeding_density >= 1:
        raise ValueError("seeding_density must be a float between 0 and 1.")
    if len(diameter_range) != 2 or not all(d > 0 for d in diameter_range):
        raise ValueError("diameter_range must be a tuple of two positive floats.")
    if len(intensity_range) != 2 or not all(0 <= i for i in intensity_range):
        raise ValueError("intensity_range must be a tuple of two positive floats")

    # 1. Determine how many particles we expect
    height, width = image_shape
    num_pixels = height * width
    num_particles = int(num_pixels * seeding_density)

    # 2. Create coordinate grids
    y_grid = jnp.arange(height)
    x_grid = jnp.arange(width)
    X, Y = jnp.meshgrid(x_grid, y_grid)

    # 3. Sample random parameters
    key_pos, key_d, key_i = jax.random.split(key, 3)

    if particle_positions is not None:
        # Use the provided particle positions
        x0s, y0s = particle_positions.T
        num_particles = len(x0s)
    else:
        # Particle center positions
        particle_positions = jax.random.uniform(
            key_pos, (num_particles, 2), minval=0.0, maxval=1.0
        ) * jnp.array([width, height])

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

    particles = jax.vmap(particle_image)(
        particle_positions[:, 0], particle_positions[:, 1], sigmas, intensities
    )
    image = image + jnp.sum(particles, axis=0)

    return jnp.clip(image, min=0, max=255)


def input_check_img_gen_from_data(
    image_shape: Tuple[int, int] = (256, 256),
    particle_positions: jnp.ndarray = None,
    max_diameter: float = 1.0,
    diameters_x: jnp.ndarray = None,
    diameters_y: jnp.ndarray = None,
    intensities: jnp.ndarray = None,
    rho: jnp.ndarray = None,
    clip: bool = True,
):
    """Check the input arguments for img_gen_from_data.

    Args:
        image_shape: Tuple[int, int]
            (height, width) of the output image.
        particle_positions: jnp.ndarray
            Array of particle positions (y, x) in pixels.
        max_diameter: float
            Maximum particle diameter in pixels.
        diameters_x: jnp.ndarray
            Array of particle diameters in the x-direction.
        diameters_y: jnp.ndarray
            Array of particle diameters in the y-direction.
        intensities: jnp.ndarray
            Array of peak intensities (I0).
        rho: jnp.ndarray
            Array of correlation coefficients (rho).
        clip: bool
            If True, clip the image values to [0, 255].
    """
    # Argument checks using exceptions instead of asserts
    if (
        len(image_shape) != 2
        or not all(s > 0 for s in image_shape)
        or not all(is_int(s) for s in image_shape)
    ):
        raise ValueError("image_shape must be a tuple of two positive integers.")
    if particle_positions is not None and (
        not isinstance(particle_positions, jnp.ndarray)
        or particle_positions.ndim != 2
        or particle_positions.shape[1] != 2
    ):
        raise ValueError("Particle positions must be a 2D array with shape (N, 2)")

    if not isinstance(max_diameter, (int, float)) or max_diameter <= 0:
        raise ValueError("max_diameter must be a positive number.")

    if (
        not isinstance(diameters_x, jnp.ndarray)
        or diameters_x.ndim != 1
        or diameters_x.shape[0] != particle_positions.shape[0]
    ):
        raise ValueError(
            "diameters_x must be a 1D array with the same length as particle_positions."
        )
    if (
        not isinstance(diameters_y, jnp.ndarray)
        or diameters_y.ndim != 1
        or diameters_y.shape[0] != particle_positions.shape[0]
    ):
        raise ValueError(
            "diameters_y must be a 1D array with the same length as particle_positions."
        )
    if (
        not isinstance(intensities, jnp.ndarray)
        or intensities.ndim != 1
        or intensities.shape[0] != particle_positions.shape[0]
    ):
        raise ValueError(
            "intensities must be a 1D array with the same length as particle_positions."
        )
    if (
        not isinstance(rho, jnp.ndarray)
        or rho.ndim != 1
        or rho.shape[0] != particle_positions.shape[0]
    ):
        raise ValueError(
            "rho must be a 1D array with the same length as particle_positions."
        )

    if not isinstance(clip, bool):
        raise ValueError("clip must be a boolean value.")


def img_gen_from_data(
    image_shape: Tuple[int, int] = (256, 256),
    particle_positions: jnp.ndarray = None,
    max_diameter: float = 1.0,
    diameters_x: jnp.ndarray = None,
    diameters_y: jnp.ndarray = None,
    intensities: jnp.ndarray = None,
    rho: jnp.ndarray = None,
    clip: bool = True,
) -> jnp.ndarray:
    """Generate a synthetic particle image from particles positions.

    This function creates an image where each particle
    is rendered as a 2D Gaussian kernel.

    Notes:
        - Particle positions are rounded to the nearest integer pixel locations.
        - Out-of-bounds particles are clipped to ensure valid rendering.

    Args:
        image_shape: Tuple[int, int]
            (height, width) of the output image.
        particle_positions: jnp.ndarray
            Array of particle positions (y, x) in pixels.
        max_diameter: float
            Maximum particle diameter in pixels.
        diameters_x: jnp.ndarray
            Array of particle diameters in the x-direction.
        diameters_y: jnp.ndarray
            Array of particle diameters in the y-direction.
        intensities: jnp.ndarray
            Array of peak intensities (I0).
        rho: jnp.ndarray
            Array of correlation coefficients (rho).
        clip: bool
            If True, clip the image values to [0, 255].

    Returns:
        jnp.ndarray: Synthetic particle image of shape `image_shape`.
    """
    H, W = image_shape

    # The radius of the patch that contains the particle
    patch_radius = int(3 * max_diameter / 2)
    patch_size = 2 * patch_radius + 1

    # Precompute a (patch_size x patch_size) Gaussian kernel centered at (0,0)
    y = jnp.arange(-patch_radius, patch_radius + 1)
    x = jnp.arange(-patch_radius, patch_radius + 1)
    Y, X = jnp.meshgrid(y, x, indexing="ij")

    # Flatten positions
    float_pos = jnp.reshape(particle_positions, (-1, 2))

    def single_particle_scatter(pos, diameter_x, diameter_y, rho, amp):
        y0, x0 = pos[0], pos[1]

        # Clip to image bounds
        y0 = jnp.clip(y0, patch_radius, H - patch_radius - 1e-6)
        x0 = jnp.clip(x0, patch_radius, W - patch_radius - 1e-6)

        # Compute top-left corner of the patch
        top_left = (
            jnp.floor(y0 - patch_radius).astype(int),
            jnp.floor(x0 - patch_radius).astype(int),
        )

        # fractional offset of the true center within that patch
        frac_y = y0 - (top_left[0] + patch_radius)
        frac_x = x0 - (top_left[1] + patch_radius)

        # Create indices for scatter_add
        yy_patch = jnp.arange(patch_size) + top_left[0]
        xx_patch = jnp.arange(patch_size) + top_left[1]

        coords = (
            jnp.array(jnp.meshgrid(yy_patch, xx_patch, indexing="ij")).reshape(2, -1).T
        )

        # Flatten kernel and scatter
        sigma_x = diameter_x / 2.0
        sigma_y = diameter_y / 2.0
        kernel = gaussian_2d_correlated(
            X - frac_x, Y - frac_y, 0, 0, sigma_x, sigma_y, rho, amp
        )
        updates = kernel.flatten()
        return coords, updates

    # Vectorized scatter prep
    coords_updates = jax.vmap(single_particle_scatter)(
        float_pos, diameters_x, diameters_y, rho, intensities
    )
    all_coords = coords_updates[0].reshape(-1, 2)
    all_updates = coords_updates[1].reshape(-1)

    # Scatter into final image
    image = jnp.zeros((H, W))
    image = image.at[tuple(all_coords.T)].add(all_updates)

    # Clip to image bounds
    if clip:
        image = jnp.clip(image, 0, 255)

    return image
