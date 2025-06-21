"""Processing module for generating images from flow fields."""
from typing import Tuple

import jax
import jax.numpy as jnp

from .apply import apply_flow_to_particles, input_check_apply_flow

# Import existing modules
from .generate import (
    add_noise_to_image,
    img_gen_from_data,
    input_check_img_gen_from_data,
)
from .utils import DEBUG_JIT, is_int, logger


def generate_images_from_flow(
    key: jax.random.PRNGKey,
    flow_field: jnp.ndarray,
    position_bounds: Tuple[int, int] = (512, 512),
    image_shape: Tuple[int, int] = (256, 256),
    num_images: int = 300,
    img_offset: Tuple[int, int] = (128, 128),
    seeding_density_range: Tuple[float, float] = (0.01, 0.02),
    p_hide_img1: float = 0.01,
    p_hide_img2: float = 0.01,
    diameter_ranges: jnp.ndarray = jnp.array([[0.1, 1.0]]),
    diameter_var: float = 1.0,
    max_diameter: float = 1.0,
    intensity_ranges: jnp.ndarray = jnp.array([[50, 200]]),
    intensity_var: float = 1.0,
    rho_ranges: jnp.ndarray = jnp.array([[-0.99, 0.99]]),
    rho_var: float = 1.0,
    dt: float = 1.0,
    flow_field_res_x: float = 1.0,
    flow_field_res_y: float = 1.0,
    noise_level: float = 0.0,
):
    """Generates a batch of grey scale image pairs from a batch of flow fields.

    This function generates pairs of images from a given flow field by simulating
    the motion of particles in the flow. A single flow can be used to generate
    multiple pairs of images, each with different particle positions and parameters.

    Args:
        key: jax.random.PRNGKey
            Random key for reproducibility.
        flow_field: jnp.ndarray
            Array of shape (N, H, W, 2) containing N velocity fields
            with velocities in length measure unit per second.
        position_bounds: Tuple[int, int]
            (height, width) bounds on the positions of the particles in pixels.
        image_shape: Tuple[int, int]
            (height, width) of the output image in pixels.
        num_images: int
            Number of image pairs to generate.
        img_offset: Tuple[int, int]
            Offset to apply to the generated images.
        seeding_density_range: Tuple[float, float]
            Range of density of particles in the images.
        p_hide_img1: float
            Probability of hiding particles in the first image.
        p_hide_img2: float
            Probability of hiding particles in the second image.
        diameter_ranges: jnp.ndarray
            Array of shape (N, 2) containing the minimum and maximum
            particle diameter in pixels.
        diameter_var: float
            Variance of the particle diameter.
        max_diameter: float
            Maximum diameter.
        intensity_ranges: jnp.ndarray
            Array of shape (N, 2) containing the minimum and maximum
            peak intensity (I0).
        intensity_var: float
            Variance of the particle intensity.
        rho_ranges: jnp.ndarray
            Array of shape (N, 2) containing the minimum and maximum
            correlation coefficient (rho).
        rho_var: float
            Variance of the correlation coefficient.
        dt: float
            Time step for the simulation, used to scale the velocity
            to compute the displacement.
        flow_field_res_x: float
            Resolution of the flow field in the x direction
            in grid steps per length measure unit.
        flow_field_res_y: float
            Resolution of the flow field in the y direction
            in grid steps per length measure unit
        noise_level: float
            Maximum amplitude of the uniform noise to add.

    Returns:
        tuple: Two image batches (num_images, H, W) each.
    """
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
    num_particles = int(
        position_bounds[0] * position_bounds[1] * seeding_density_range[1]
    )

    # Number of flow fields
    num_flow_fields = flow_field.shape[0]

    # Pre-sample seeding densities
    key, density_key = jax.random.split(key)
    seeding_densities = jax.random.uniform(
        density_key,
        shape=(num_images,),
        minval=seeding_density_range[0],
        maxval=seeding_density_range[1],
    )

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
            subkey1, 1.0 - p_hide_img1, shape=(num_particles,)
        )
        mask_img2 = jax.random.bernoulli(
            subkey2, 1.0 - p_hide_img2, shape=(num_particles,)
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
        diameters_y1 = jax.random.uniform(
            key_dy,
            shape=(num_particles,),
            minval=diameter_range[0],
            maxval=diameter_range[1],
        )

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
            diameter_var
        )
        noise_dy = jax.random.normal(key_noise_dy, shape=(num_particles,)) * jnp.sqrt(
            diameter_var
        )
        noise_rho = jax.random.normal(key_noise_rho, shape=(num_particles,)) * jnp.sqrt(
            rho_var
        )
        noise_i = jax.random.normal(key_noise_in, shape=(num_particles,)) * jnp.sqrt(
            intensity_var
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

        if DEBUG_JIT:
            input_check_img_gen_from_data(
                particle_positions=particle_positions,
                image_shape=position_bounds,
                max_diameter=max_diameter,
                diameters_x=diameters_x1,
                diameters_y=diameters_y1,
                intensities=intensities1 * mask_img1 * mixed,
                rho=rho1,
                clip=False,
            )

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

        if DEBUG_JIT:
            input_check_apply_flow(
                particle_positions=particle_positions,
                flow_field=flow_field_i,
                dt=dt,
                flow_field_res_x=flow_field_res_x,
                flow_field_res_y=flow_field_res_y,
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
            dt=dt,
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

        if DEBUG_JIT:
            input_check_img_gen_from_data(
                particle_positions=final_positions,
                image_shape=position_bounds,
                max_diameter=max_diameter,
                diameters_x=diameters_x2,
                diameters_y=diameters_y2,
                intensities=intensities2 * mask_img2 * mixed,
                rho=rho2,
                clip=False,
            )

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
            img_offset[0] : image_shape[0] + img_offset[0],
            img_offset[1] : image_shape[1] + img_offset[1],
        ]
        second_img = second_img[
            img_offset[0] : image_shape[0] + img_offset[0],
            img_offset[1] : image_shape[1] + img_offset[1],
        ]

        # Add noise to the images
        first_img = add_noise_to_image(
            image=first_img, key=subkey5, noise_level=noise_level
        )
        second_img = add_noise_to_image(
            image=second_img, key=subkey6, noise_level=noise_level
        )

        outputs = (first_img, second_img, diameter_idx, intensity_idx, rho_idx)
        new_carry = (key,)
        return new_carry, outputs

    # Prepare scan inputs
    indices = jnp.arange(num_images)
    scan_inputs = (indices, seeding_densities)

    # Generate images using a lax.scan loop
    # For some reason, even if the different indices are independent, vmap is slower
    _, outs = jax.lax.scan(
        scan_body,
        (key,),
        scan_inputs,
    )
    first_imgs, second_imgs, diameter_indices, intensity_indices, rho_indices = outs

    # Optionally, map indices back to actual tuples for reporting
    used_diameter_ranges = jnp.array(diameter_ranges)[diameter_indices]
    used_intensity_ranges = jnp.array(intensity_ranges)[intensity_indices]
    used_rho_ranges = jnp.array(rho_ranges)[rho_indices]

    return {
        "first_images": first_imgs,
        "second_images": second_imgs,
        "params": {
            "seeding_densities": seeding_densities,
            "diameter_ranges": used_diameter_ranges,
            "intensity_ranges": used_intensity_ranges,
            "rho_ranges": used_rho_ranges,
        },
    }


def input_check_gen_img_from_flow(
    key: jax.random.PRNGKey,
    flow_field: jnp.ndarray,
    position_bounds: Tuple[int, int] = (512, 512),
    image_shape: Tuple[int, int] = (256, 256),
    num_images: int = 300,
    img_offset: Tuple[int, int] = (128, 128),
    seeding_density_range: Tuple[float, float] = (0.01, 0.02),
    p_hide_img1: float = 0.01,
    p_hide_img2: float = 0.01,
    diameter_ranges: jnp.ndarray = jnp.array([[0.1, 1.0]]),
    diameter_var: float = 1.0,
    max_diameter: float = 1.0,
    intensity_ranges: jnp.ndarray = jnp.array([[50, 200]]),
    intensity_var: float = 1.0,
    rho_ranges: jnp.ndarray = jnp.array([[-0.99, 0.99]]),
    rho_var: float = 1.0,
    dt: float = 1.0,
    flow_field_res_x: float = 1.0,
    flow_field_res_y: float = 1.0,
    noise_level: float = 0.0,
):
    """Check the input arguments for generate_images_from_flow.

    Args:
        key: jax.random.PRNGKey
            Random key for reproducibility.
        flow_field: jnp.ndarray
            Array of shape (N, H, W, 2) containing N velocity fields
            with velocities in length measure unit per second.
        position_bounds: Tuple[int, int]
            (height, width) bounds on the positions of the particles in pixels.
        image_shape: Tuple[int, int]
            (height, width) of the output image in pixels.
        num_images: int
            Number of image pairs to generate.
        img_offset: Tuple[int, int]
            Offset to apply to the generated images.
        seeding_density_range: Tuple[float, float]
            Range of density of particles in the images.
        p_hide_img1: float
            Probability of hiding particles in the first image.
        p_hide_img2: float
            Probability of hiding particles in the second image.
        diameter_ranges: jnp.ndarray
            Array of shape (N, 2) containing the minimum and maximum
            particle diameter in pixels.
        diameter_var: float
            Variance of the particle diameter.
        max_diameter: float
            Maximum diameter.
        intensity_ranges: jnp.ndarray
            Array of shape (N, 2) containing the minimum and maximum
            peak intensity (I0).
        intensity_var: float
            Variance of the particle intensity.
        rho_ranges: jnp.ndarray
            Array of shape (N, 2) containing the minimum and maximum
            correlation coefficient (rho).
        rho_var: float
            Variance of the correlation coefficient.
        dt: float
            Time step for the simulation, used to scale the velocity
            to compute the displacement.
        flow_field_res_x: float
            Resolution of the flow field in the x direction
            in grid steps per length measure unit.
        flow_field_res_y: float
            Resolution of the flow field in the y direction
            in grid steps per length measure unit
        noise_level: float
            Maximum amplitude of the uniform noise to add.
    """
    # Argument checks using exceptions instead of asserts
    if not isinstance(key, jax.Array) or key.shape != (2,) or key.dtype != jnp.uint32:
        raise ValueError(
            "key must be a jax.array with shape (2,) and dtype jnp.uint32."
        )
    if (
        len(image_shape) != 2
        or not all(s > 0 for s in image_shape)
        or not all(is_int(s) for s in image_shape)
    ):
        raise ValueError("image_shape must be a tuple of two positive integers.")
    if (
        len(position_bounds) != 2
        or not all(s > 0 for s in position_bounds)
        or not all(is_int(s) for s in position_bounds)
    ):
        raise ValueError("position_bounds must be a tuple of two positive integers.")
    if (
        len(img_offset) != 2
        or not all(is_int(s) for s in img_offset)
        or not all(s >= 0 for s in img_offset)
    ):
        raise ValueError("img_offset must be a tuple of two non-negative integers.")
    if (
        not isinstance(flow_field, jnp.ndarray)
        or flow_field.ndim != 4
        or flow_field.shape[3] != 2
    ):
        raise ValueError("Flow_field must be a 4D jnp.ndarray with shape (N, H, W, 2).")

    # Check diameter_ranges
    if not (
        isinstance(diameter_ranges, jnp.ndarray)
        and diameter_ranges.ndim == 2
        and diameter_ranges.shape[1] == 2
    ):
        raise ValueError("diameter_ranges must be a 2D jnp.ndarray with shape (N, 2).")

    if not jnp.all(diameter_ranges > 0):
        raise ValueError("All values in diameter_ranges must be > 0.")

    if not jnp.all(diameter_ranges[:, 0] <= diameter_ranges[:, 1]):
        raise ValueError("Each diameter_range must satisfy min <= max.")

    # Check intensity_ranges
    if not (
        isinstance(intensity_ranges, jnp.ndarray)
        and intensity_ranges.ndim == 2
        and intensity_ranges.shape[1] == 2
    ):
        raise ValueError("intensity_ranges must be a 2D jnp.ndarray with shape (N, 2).")

    if not jnp.all(intensity_ranges >= 0):
        raise ValueError("All values in intensity_ranges must be >= 0.")

    if not jnp.all(intensity_ranges[:, 0] <= intensity_ranges[:, 1]):
        raise ValueError("Each intensity_range must satisfy min <= max.")

    # Check rho_ranges
    if not (
        isinstance(rho_ranges, jnp.ndarray)
        and rho_ranges.ndim == 2
        and rho_ranges.shape[1] == 2
    ):
        raise ValueError("rho_ranges must be a 2D jnp.ndarray with shape (N, 2).")

    if not jnp.all((-1 < rho_ranges) & (rho_ranges < 1)):
        raise ValueError(
            "All values in rho_ranges must be in the open interval (-1, 1)."
        )

    if not jnp.all(rho_ranges[:, 0] <= rho_ranges[:, 1]):
        raise ValueError("Each rho_range must satisfy min <= max.")

    if not isinstance(num_images, int) or num_images <= 0:
        raise ValueError("num_images must be a positive integer.")
    if not (0 <= p_hide_img1 <= 1):
        raise ValueError("p_hide_img1 must be between 0 and 1.")
    if not (0 <= p_hide_img2 <= 1):
        raise ValueError("p_hide_img2 must be between 0 and 1.")
    if not isinstance(dt, (int, float)):
        raise ValueError("dt must be a scalar (int or float)")
    if not isinstance(flow_field_res_x, (int, float)) or flow_field_res_x <= 0:
        raise ValueError("flow_field_res_x must be a positive scalar (int or float)")
    if not isinstance(flow_field_res_y, (int, float)) or flow_field_res_y <= 0:
        raise ValueError("flow_field_res_y must be a positive scalar (int or float)")
    if position_bounds[0] < image_shape[0] + img_offset[0]:
        raise ValueError(
            "The height of the position_bounds must be greater "
            "than the height of the image plus the offset."
        )
    if position_bounds[1] < image_shape[1] + img_offset[1]:
        raise ValueError(
            "The width of the position_bounds must be greater "
            "than the width of the image plus the offset."
        )
    if len(seeding_density_range) != 2 or not all(
        isinstance(s, (int, float)) and s >= 0 for s in seeding_density_range
    ):
        raise ValueError(
            "seeding_density_range must be a tuple of two non-negative numbers."
        )
    if seeding_density_range[0] > seeding_density_range[1]:
        raise ValueError("seeding_density_range must be in the form (min, max).")
    if not isinstance(noise_level, (int, float)) or noise_level < 0:
        raise ValueError("noise_level must be a non-negative number.")

    if not isinstance(diameter_var, (int, float)) or diameter_var < 0:
        raise ValueError("diameter_var must be a non-negative number.")
    if not isinstance(max_diameter, (int, float)) or max_diameter <= 0:
        raise ValueError("max_diameter must be a positive number.")
    if not isinstance(intensity_var, (int, float)) or intensity_var < 0:
        raise ValueError("intensity_var must be a non-negative number.")
    if not isinstance(rho_var, (int, float)) or rho_var < 0:
        raise ValueError("rho_var must be a non-negative number.")

    num_particles = int(
        position_bounds[0] * position_bounds[1] * seeding_density_range[1]
    )
    logger.debug("Input arguments of generate_images_from_flow are valid.")
    logger.debug(f"Flow field shape: {flow_field.shape}")
    logger.debug(f"Image shape: {image_shape}")
    logger.debug(f"Position bounds shape: {position_bounds}")
    logger.debug(f"Number of images: {num_images}")
    logger.debug(f"Particles density range: {seeding_density_range}")
    logger.debug(f"Number of particles: {num_particles}")
    logger.debug(f"Probability of hiding particles in image 1: {p_hide_img1}")
    logger.debug(f"Probability of hiding particles in image 2: {p_hide_img2}")
    logger.debug(f"Particle diameter ranges: {diameter_ranges}")
    logger.debug(f"Intensity ranges: {intensity_ranges}")
    logger.debug(f"Correlation coefficient ranges: {rho_ranges}")
    logger.debug(f"Time step (dt): {dt}")
    logger.debug(f"Flow field resolution (x): {flow_field_res_x}")
    logger.debug(f"Flow field resolution (y): {flow_field_res_y}")
    logger.debug(f"Noise level: {noise_level}")
