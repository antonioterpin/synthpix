"""Processing module for generating images from flow fields."""

import jax
import jax.numpy as jnp

import goggles as gg

from .apply import apply_flow_to_particles
from synthpix.types import PRNGKey

# Import existing modules
from synthpix.generate import add_noise_to_image, img_gen_from_data
from synthpix.utils import DEBUG_JIT, match_histogram, SYNTHPIX_SCOPE
from synthpix.types import ImageGenerationParameters, ImageGenerationSpecification

logger = gg.get_logger(__name__, scope=SYNTHPIX_SCOPE)


def generate_images_from_flow(
    key: PRNGKey,
    flow_field: jnp.ndarray,
    parameters: ImageGenerationSpecification,
    position_bounds: tuple[int, int] = (512, 512),
    flow_field_res_x: float = 1.0,
    flow_field_res_y: float = 1.0,
    mask: jnp.ndarray | None = None,
    histogram: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, ImageGenerationParameters]:
    """Generates a batch of grey scale image pairs from a batch of flow fields.

    This function generates pairs of images from a given flow field by simulating
    the motion of particles in the flow. A single flow can be used to generate
    multiple pairs of images, each with different particle positions and parameters.

    Args:
        key: Random key for reproducibility.
        flow_field: Array of shape (N, H, W, 2) containing N velocity fields
            with velocities in length measure unit per second.
        parameters: ImageGenerationSpecification dataclass containing all the
            parameters for image generation.
        position_bounds: (height, width) bounds on the positions of
            the particles in pixels.
        flow_field_res_x: Resolution of the flow field in the x direction
            in grid steps per length measure unit.
        flow_field_res_y: Resolution of the flow field in the y direction
            in grid steps per length measure unit.
        mask: Optional mask to apply to the generated images.
        histogram: Optional histogram to match the images to.
            NOTE: Histogram equalization is very slow!

    Returns:
        Tuple containing:
        - images1: Array of shape (num_images, image_height, image_width),
                the first images in the pairs.
        - images2: Array of shape (num_images, image_height, image_width),
                the second images in the pairs.
        - params: ImageGenerationParameters dataclass containing the
            sampled parameters used for generation.
    """
    if DEBUG_JIT:
        input_check_gen_img_from_flow(
            flow_field=flow_field, parameters=parameters, mask=mask, histogram=histogram
        )

    # Compute maximum diameter and maximum seeding density
    max_seeding_density = parameters.seeding_density_range[1]
    max_diameter = max(d[1] for d in parameters.diameter_ranges)

    # Fix the key shape
    key = jnp.reshape(key, (-1, key.shape[-1]))[0]

    # scale factors for particle positions
    # position bounds are in pixels, flow field is in grid steps
    # doing so, our position bounds cover the whole flow field
    alpha1 = flow_field.shape[1] / position_bounds[0]
    alpha2 = flow_field.shape[2] / position_bounds[1]

    # Calculate the number of particles based on the max density
    # Density is given in particles per pixel, so we use
    # the number of pixels in position bounds
    num_particles = int(position_bounds[0] * position_bounds[1] * max_seeding_density)

    # Number of flow fields
    num_flow_fields = flow_field.shape[0]

    # Pre-sample seeding densities
    key, density_key = jax.random.split(key)
    seeding_densities = jax.random.uniform(
        density_key,
        shape=(parameters.batch_size,),
        minval=parameters.seeding_density_range[0],
        maxval=parameters.seeding_density_range[1],
    )
    seeding_densities = jnp.clip(seeding_densities, 0, max_seeding_density)
    diameter_ranges = jnp.array(parameters.diameter_ranges)
    intensity_ranges = jnp.array(parameters.intensity_ranges)
    rho_ranges = jnp.array(parameters.rho_ranges)

    def scan_body(carry, inputs):
        (key,) = carry
        i, seeding_density = inputs

        # Split the key for randomness
        key_i = jax.random.fold_in(key, i)
        subkeys = jax.random.split(key_i, 9)
        (
            subkey1,
            subkey2,
            subkey3,
            subkey4,
            subkey5,
            subkey6,
            subkey7,
            subkey8,
            subkey9,
        ) = subkeys

        # Randomly select a range for this image for each property
        diameter_idx = jax.random.randint(subkey7, (), 0, len(diameter_ranges))
        intensity_idx = jax.random.randint(subkey8, (), 0, len(intensity_ranges))
        rho_idx = jax.random.randint(subkey9, (), 0, len(rho_ranges))

        diameter_range = diameter_ranges[diameter_idx]
        intensity_range = intensity_ranges[intensity_idx]
        rho_range = rho_ranges[rho_idx]

        # Calculate the number of particles for this couple of images
        current_num_particles = jnp.floor(
            position_bounds[0] * position_bounds[1] * seeding_density
        )

        # Make visible only the particles that are in the current image
        mixed = jax.lax.iota(jnp.int32, num_particles) < current_num_particles

        # Get the flow field for the current iteration
        flow_field_i = flow_field[i % num_flow_fields]

        # Generate random masks
        mask_img1 = jax.random.bernoulli(
            subkey1, 1.0 - parameters.p_hide_img1, shape=(num_particles,)
        )
        mask_img2 = jax.random.bernoulli(
            subkey2, 1.0 - parameters.p_hide_img2, shape=(num_particles,)
        )

        # Generate random particle positions
        H, W = position_bounds
        particle_positions = jax.random.uniform(
            subkey3, (num_particles, 2)
        ) * jnp.array([H, W])

        (
            key_dx,
            key_dy,
            key_rho,
            key_in,
            key_noise_dx,
            key_noise_dy,
            key_noise_rho,
            key_noise_in,
        ) = jax.random.split(subkey4, 8)

        # Sample random parameters for the particles
        # Diameters in the specified range, then convert to sigma = diameter / 2
        diameters_x1 = jax.random.uniform(
            key_dx,
            shape=(num_particles,),
            minval=diameter_range[0],
            maxval=diameter_range[1],
        )
        diameters_y1 = diameters_x1 + jax.random.uniform(
            key_dy,
            shape=(num_particles,),
            minval=-1,
            maxval=+1,
        )
        diameters_y1 = jnp.clip(diameters_y1, diameter_range[0], diameter_range[1])

        # Sample theta in the specified range
        rho1 = jax.random.uniform(
            key_rho,
            shape=(num_particles,),
            minval=rho_range[0],
            maxval=rho_range[1],
        )

        # Peak intensities
        intensities1 = jax.random.uniform(
            key_in,
            shape=(num_particles,),
            minval=intensity_range[0],
            maxval=intensity_range[1],
        )

        # Generate Gaussian noise with mean 0 and standard deviation = sqrt(variance)
        noise_dx = jax.random.normal(key_noise_dx, shape=(num_particles,)) * jnp.sqrt(
            parameters.diameter_var
        )
        noise_dy = jax.random.normal(key_noise_dy, shape=(num_particles,)) * jnp.sqrt(
            parameters.diameter_var
        )
        noise_rho = jax.random.normal(key_noise_rho, shape=(num_particles,)) * jnp.sqrt(
            parameters.rho_var
        )
        noise_i = jax.random.normal(key_noise_in, shape=(num_particles,)) * jnp.sqrt(
            parameters.intensity_var
        )

        # Add noise to the original values
        diameters_x2 = diameters_x1 + noise_dx
        diameters_y2 = diameters_y1 + noise_dy
        rho2 = rho1 + noise_rho
        intensities2 = intensities1 + noise_i

        # Clip the noisy values to their respective ranges
        diameters_x2 = jnp.clip(diameters_x2, diameter_range[0], diameter_range[1])
        diameters_y2 = jnp.clip(diameters_y2, diameter_range[0], diameter_range[1])
        rho2 = jnp.clip(rho2, rho_range[0], rho_range[1])
        intensities2 = jnp.clip(intensities2, intensity_range[0], intensity_range[1])

        # First image generation
        first_img = img_gen_from_data(
            particle_positions=particle_positions,
            image_shape=position_bounds,
            max_diameter=max_diameter,
            diameters_x=diameters_x1,
            diameters_y=diameters_y1,
            intensities=intensities1 * mask_img1 * mixed,
            rho=rho1,
            clip=False,
        )

        # Rescale the particle positions to match the flow field resolution
        particle_positions = jnp.array(
            [
                particle_positions[:, 0] * alpha1,
                particle_positions[:, 1] * alpha2,
            ]
        ).T

        # Apply flow field to particle positions
        final_positions = apply_flow_to_particles(
            particle_positions=particle_positions,
            flow_field=flow_field_i,
            dt=parameters.dt,
            flow_field_res_x=flow_field_res_x,
            flow_field_res_y=flow_field_res_y,
        )

        # Rescale the coordinates back to the original scale
        final_positions = jnp.array(
            [
                final_positions[:, 0] / alpha1,
                final_positions[:, 1] / alpha2,
            ]
        ).T

        # Second image generation
        second_img = img_gen_from_data(
            particle_positions=final_positions,
            image_shape=position_bounds,
            max_diameter=max_diameter,
            diameters_x=diameters_x2,
            diameters_y=diameters_y2,
            intensities=intensities2 * mask_img2 * mixed,
            rho=rho2,
            clip=False,
        )

        # Crop the images to image_shape
        first_img = first_img[
            parameters.img_offset[0] : position_bounds[0] + parameters.img_offset[0],
            parameters.img_offset[1] : position_bounds[1] + parameters.img_offset[1],
        ]
        second_img = second_img[
            parameters.img_offset[0] : position_bounds[0] + parameters.img_offset[0],
            parameters.img_offset[1] : position_bounds[1] + parameters.img_offset[1],
        ]

        # Add noise to the images
        first_img = add_noise_to_image(
            image=first_img,
            key=subkey5,
            noise_uniform=parameters.noise_uniform,
            noise_gaussian_mean=parameters.noise_gaussian_mean,
            noise_gaussian_std=parameters.noise_gaussian_std,
        )
        second_img = add_noise_to_image(
            image=second_img,
            key=subkey6,
            noise_uniform=parameters.noise_uniform,
            noise_gaussian_mean=parameters.noise_gaussian_mean,
            noise_gaussian_std=parameters.noise_gaussian_std,
        )

        # If a mask is provided, apply it to the images
        if mask is not None:
            first_img = first_img * mask
            second_img = second_img * mask

        # If histogram is provided, equalize the images
        if histogram is not None:
            first_img = match_histogram(first_img, histogram)
            second_img = match_histogram(second_img, histogram)

        outputs = (first_img, second_img, diameter_idx, intensity_idx, rho_idx)
        new_carry = (key,)
        return new_carry, outputs

    # Prepare scan inputs
    indices = jnp.arange(parameters.batch_size)
    scan_inputs = (indices, seeding_densities)

    # Generate images using a lax.scan loop
    # For some reason, even if the different indices are independent, vmap is slower
    _, outs = jax.lax.scan(
        scan_body,
        (key,),
        scan_inputs,
    )
    images1, images2, diameter_indices, intensity_indices, rho_indices = outs

    # Optionally, map indices back to actual tuples for reporting
    used_diameter_ranges = jnp.array(parameters.diameter_ranges)[diameter_indices]
    used_intensity_ranges = jnp.array(parameters.intensity_ranges)[intensity_indices]
    used_rho_ranges = jnp.array(parameters.rho_ranges)[rho_indices]

    return (
        images1,
        images2,
        ImageGenerationParameters(
            seeding_densities=seeding_densities,
            diameter_ranges=used_diameter_ranges,
            intensity_ranges=used_intensity_ranges,
            rho_ranges=used_rho_ranges,
        ),
    )


def input_check_gen_img_from_flow(
    flow_field: jnp.ndarray,
    parameters: ImageGenerationSpecification,
    position_bounds: tuple[int, int] = (512, 512),
    flow_field_res_x: float = 1.0,
    flow_field_res_y: float = 1.0,
    mask: jnp.ndarray | None = None,
    histogram: jnp.ndarray | None = None,
):
    """Check the input arguments for generate_images_from_flow.

    Args:
        flow_field: Array of shape (N, H, W, 2) containing N velocity fields
            with velocities in length measure unit per second.
        parameters: ImageGenerationSpecification dataclass containing all the
            parameters for image generation.
        position_bounds: (height, width) bounds on the positions of
            the particles in pixels.
        flow_field_res_x: Resolution of the flow field in the x direction
            in grid steps per length measure unit.
        flow_field_res_y: Resolution of the flow field in the y direction
            in grid steps per length measure unit.
        mask: Optional mask to apply to the generated images.
        histogram: Optional histogram to match the images to.
            NOTE: Histogram equalization is very slow!
    """
    if not isinstance(flow_field, jnp.ndarray):
        raise ValueError(f"flow_field must be a jnp.ndarray, got {type(flow_field)}.")
    if flow_field.ndim != 4 or flow_field.shape[3] != 2:
        raise ValueError(
            "flow_field must be a 4D jnp.ndarray with shape (N, H, W, 2), "
            f"got shape {flow_field.shape}."
        )
    if (
        len(position_bounds) != 2
        or not all(s > 0 for s in position_bounds)
        or not all(isinstance(s, int) for s in position_bounds)
    ):
        raise ValueError("position_bounds must be a tuple of two positive integers.")

    if not (isinstance(flow_field_res_x, (int, float)) and flow_field_res_x > 0):
        raise ValueError("flow_field_res_x must be a positive scalar (int or float)")
    if not (isinstance(flow_field_res_y, (int, float)) and 0 < flow_field_res_y):
        raise ValueError("flow_field_res_y must be a positive scalar (int or float)")
    if position_bounds[0] < parameters.image_shape[0] + parameters.img_offset[0]:
        raise ValueError(
            "The height of the position_bounds must be greater "
            "than the height of the image plus the offset."
        )
    if position_bounds[1] < parameters.image_shape[1] + parameters.img_offset[1]:
        raise ValueError(
            "The width of the position_bounds must be greater "
            "than the width of the image plus the offset."
        )

    if mask is not None and not isinstance(mask, jnp.ndarray):
        raise ValueError("mask must be a jnp.ndarray or None.")
    if mask is not None and mask.shape != parameters.image_shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match image_shape {parameters.image_shape}."
        )
    if histogram is not None and not isinstance(histogram, jnp.ndarray):
        raise ValueError("histogram must be a jnp.ndarray or None.")
    if histogram is not None and histogram.ndim != 1:
        raise ValueError("histogram must be a 1D jnp.ndarray.")
    if histogram is not None and histogram.shape[0] != 256:
        raise ValueError("histogram must have 256 bins (shape (256,)).")
    if histogram is not None and not (
        jnp.isclose(
            jnp.sum(histogram), parameters.image_shape[0] * parameters.image_shape[1]
        )
    ):
        raise ValueError(
            "Histogram must sum to the number of pixels in the image shape."
        )

    num_particles = int(
        position_bounds[0] * position_bounds[1] * parameters.seeding_density_range[1]
    )
    logger.debug("Input arguments of generate_images_from_flow are valid.")
    logger.debug(f"Flow field shape: {flow_field.shape}")
    logger.debug(f"Image shape: {parameters.image_shape}")
    logger.debug(f"Position bounds shape: {position_bounds}")
    logger.debug(f"Number of images: {parameters.batch_size}")
    logger.debug(f"Particles density range: {parameters.seeding_density_range}")
    logger.debug(f"Number of particles: {num_particles}")
    logger.debug(
        f"Probability of hiding particles in image 1: {parameters.p_hide_img1}"
    )
    logger.debug(
        f"Probability of hiding particles in image 2: {parameters.p_hide_img2}"
    )
    logger.debug(f"Particle diameter ranges: {parameters.diameter_ranges}")
    logger.debug(f"Diameter variance: {parameters.diameter_var}")
    logger.debug(f"Intensity ranges: {parameters.intensity_ranges}")
    logger.debug(f"Intensity variance: {parameters.intensity_var}")
    logger.debug(f"Correlation coefficient ranges: {parameters.rho_ranges}")
    logger.debug(f"Correlation coefficient variance: {parameters.rho_var}")
    logger.debug(f"Time step (dt): {parameters.dt}")
    logger.debug(f"Flow field resolution (x): {flow_field_res_x}")
    logger.debug(f"Flow field resolution (y): {flow_field_res_y}")
    logger.debug(f"Noise level: {parameters.noise_uniform}")
    logger.debug(f"Gaussian noise mean: {parameters.noise_gaussian_mean}")
    logger.debug(f"Gaussian noise std: {parameters.noise_gaussian_std}")
    if mask is not None:
        debug_msg = (
            f"Masking out {parameters.image_shape[0] * parameters.image_shape[1] - jnp.sum(mask)} pixels "
            "in the images."
        )
        logger.debug(debug_msg)
    if histogram is not None:
        logger.debug("Histogram equalization will be applied to the images.")
