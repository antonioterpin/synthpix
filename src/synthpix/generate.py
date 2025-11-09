"""Module to generate synthetic images for testing and debugging."""

import jax
import jax.numpy as jnp

from synthpix.types import PRNGKey


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
        x: 2D coordinate grid for x-axis.
        y: 2D coordinate grid for y-axis.
        x0: Center position of the Gaussian on the x-axis.
        y0: Center position of the Gaussian on the y-axis.
        sigma_x: Standard deviation of the Gaussian on the x-axis.
        sigma_y: Standard deviation of the Gaussian on the y-axis.
        rho: Correlation coefficient between x and y.
        amplitude: Peak intensity (I0).

    Returns:
        2D array representing the Gaussian function.
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
    key: PRNGKey,
    image: jnp.ndarray,
    noise_uniform: float = 0.0,
    noise_gaussian_mean: float = 0.0,
    noise_gaussian_std: float = 0.0,
):
    """Add noise to an image.

    Args:
        key: Random key for reproducibility.
        image: Input image.
        noise_uniform: Maximum amplitude of the uniform noise to add.
        noise_gaussian_mean: Mean of the Gaussian noise to add.
        noise_gaussian_std: Standard deviation of the Gaussian noise to add.

    Returns:
        The noisy image.
    """
    uniform_key, gaussian_key = jax.random.split(key)

    # Add uniform noise
    image += jax.random.uniform(
        uniform_key, shape=image.shape, minval=0, maxval=noise_uniform
    )
    # Add Gaussian noise
    image += noise_gaussian_mean
    image += jax.random.normal(gaussian_key, shape=image.shape) * noise_gaussian_std

    # Clip the final image to valid range
    return jnp.clip(image, 0, 255)


def img_gen_from_data(
    particle_positions: jnp.ndarray,
    image_shape: tuple[int, int] = (256, 256),
    max_diameter: float = 1.0,
    diameters_x: jnp.ndarray | None = None,
    diameters_y: jnp.ndarray | None = None,
    intensities: jnp.ndarray | None = None,
    rho: jnp.ndarray | None = None,
    clip: bool = True,
) -> jnp.ndarray:
    """Generate a synthetic particle image from particles positions.

    This function creates an image where each particle
    is rendered as a 2D Gaussian kernel.

    Notes:
        - Particle positions are rounded to the nearest integer pixel locations.
        - Out-of-bounds particles are clipped to ensure valid rendering.

    Args:
        particle_positions: Array of particle positions (y, x) in pixels.
        image_shape: (height, width) of the output image.
        max_diameter: Maximum particle diameter in pixels.
        diameters_x: Array of particle diameters in the x-direction.
        diameters_y: Array of particle diameters in the y-direction.
        intensities: Array of peak intensities (I0).
        rho: Array of correlation coefficients (rho).
        clip: If True, clip the image values to [0, 255].

    Returns:
        Synthetic particle image of shape `image_shape`.
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
