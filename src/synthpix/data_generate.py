"""Processing module for generating images from flow fields."""
from typing import Tuple

import jax
import jax.numpy as jnp

from synthpix.apply import apply_flow_to_particles, input_check_apply_flow

# Import existing modules
from synthpix.generate import img_gen_from_data, input_check_img_gen_from_data
from synthpix.utils import DEBUG_JIT, is_int, logger


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
    diameter_range: Tuple[float, float] = (0.1, 1.0),
    intensity_range: Tuple[float, float] = (50, 200),
    rho_range: Tuple[float, float] = (-0.99, 0.99),
    dt: float = 1.0,
    flow_field_res_x: float = 1.0,
    flow_field_res_y: float = 1.0,
):
    """Generates a batch of image pairs from a flow field.

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
        diameter_range: Tuple[float, float]
            Minimum and maximum particle diameter in pixels.
        intensity_range: Tuple[float, float]
            Minimum and maximum peak intensity (I0).
        rho_range: Tuple[float, float]
            Minimum and maximum correlation coefficient (rho).
        dt: float
            Time step for the simulation, used to scale the velocity
            to compute the displacement.
        flow_field_res_x: float
            Resolution of the flow field in the x direction
            in grid steps per length measure unit.
        flow_field_res_y: float
            Resolution of the flow field in the y direction
            in grid steps per length measure unit

    Returns:
        tuple: Two image batches (num_images, H, W) each.
    """
    # Fix the key shape
    key = jnp.reshape(key, (-1, key.shape[-1]))[0]

    # scale factors for particle positions
    alpha1 = flow_field.shape[1] / position_bounds[0]
    alpha2 = flow_field.shape[2] / position_bounds[1]

    # Calculate the number of particles based on the max density
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
        subkey1, subkey2, subkey3, subkey4, subkey5 = jax.random.split(key_i, 5)

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

        if DEBUG_JIT:
            input_check_img_gen_from_data(
                key=subkey4,
                particle_positions=particle_positions
                * mask_img1[:, None]
                * mixed[:, None],
                image_shape=position_bounds,
                diameter_range=diameter_range,
                intensity_range=intensity_range,
                rho_range=rho_range,
            )

        # First image generation
        first_img = img_gen_from_data(
            key=subkey4,
            particle_positions=particle_positions * mask_img1[:, None] * mixed[:, None],
            image_shape=position_bounds,
            diameter_range=diameter_range,
            intensity_range=intensity_range,
            rho_range=rho_range,
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
                key=subkey5,
                particle_positions=final_positions
                * mask_img2[:, None]
                * mixed[:, None],
                image_shape=position_bounds,
                diameter_range=diameter_range,
                intensity_range=intensity_range,
                rho_range=rho_range,
            )

        # Second image generation
        second_img = img_gen_from_data(
            key=subkey5,
            particle_positions=final_positions * mask_img2[:, None] * mixed[:, None],
            image_shape=position_bounds,
            diameter_range=diameter_range,
            intensity_range=intensity_range,
            rho_range=rho_range,
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

        outputs = (first_img, second_img)
        new_carry = (key,)
        return new_carry, outputs

    # Prepare scan inputs
    indices = jnp.arange(num_images)
    scan_inputs = (indices, seeding_densities)

    # Generate images using a lax.scan loop
    # For some reason, even if the different indices are independent, vmap is slower
    _, (first_imgs, second_imgs) = jax.lax.scan(
        scan_body,
        (key,),
        scan_inputs,
    )

    return first_imgs, second_imgs, seeding_densities


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
    diameter_range: Tuple[float, float] = (0.1, 1.0),
    intensity_range: Tuple[float, float] = (50, 200),
    rho_range: Tuple[float, float] = (-0.99, 0.99),
    dt: float = 1.0,
    flow_field_res_x: float = 1.0,
    flow_field_res_y: float = 1.0,
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
        diameter_range: Tuple[float, float]
            Minimum and maximum particle diameter in pixels.
        intensity_range: Tuple[float, float]
            Minimum and maximum peak intensity (I0).
        rho_range: Tuple[float, float]
            Minimum and maximum correlation coefficient (rho).
        dt: float
            Time step for the simulation, used to scale the velocity
            to compute the displacement.
        flow_field_res_x: float
            Resolution of the flow field in the x direction
            in grid steps per length measure unit.
        flow_field_res_y: float
            Resolution of the flow field in the y direction
            in grid steps per length measure unit
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
    if len(diameter_range) != 2 or not all(d > 0 for d in diameter_range):
        raise ValueError("diameter_range must be a tuple of two positive floats.")
    if diameter_range[0] > diameter_range[1]:
        raise ValueError("diameter_range must be in the form (min, max).")
    if len(intensity_range) != 2 or not all(i >= 0 for i in intensity_range):
        raise ValueError("intensity_range must be a tuple of two positive floats.")
    if intensity_range[0] > intensity_range[1]:
        raise ValueError("intensity_range must be in the form (min, max).")
    if len(rho_range) != 2 or not all(-1 <= i <= 1 for i in rho_range):
        raise ValueError("rho_range must be a tuple of two floats between -1 and 1.")
    if rho_range[0] > rho_range[1]:
        raise ValueError("rho_range must be in the form (min, max).")
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
    logger.debug(f"Particle diameter range: {diameter_range}")
    logger.debug(f"Intensity range: {intensity_range}")
    logger.debug(f"Correlation coefficient range: {rho_range}")
    logger.debug(f"Time step (dt): {dt}")
    logger.debug(f"Flow field resolution (x): {flow_field_res_x}")
    logger.debug(f"Flow field resolution (y): {flow_field_res_y}")
