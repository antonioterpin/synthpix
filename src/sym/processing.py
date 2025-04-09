"""Processing module for generating images from flow fields."""
from typing import Tuple

import jax
import jax.numpy as jnp

from src.sym.apply import apply_flow_to_particles, input_check_apply_flow

# Import existing modules
from src.sym.generate import img_gen_from_data, input_check_img_gen_from_data
from src.utils import is_int, logger

DEBUG = False


def generate_images_from_flow(
    key: jax.random.PRNGKey,
    flow_field: jnp.ndarray,
    position_bounds: Tuple[int, int] = (512, 512),
    image_shape: Tuple[int, int] = (256, 256),
    num_images: int = 300,
    img_offset: Tuple[int, int] = (128, 128),
    num_particles: int = 10000,
    p_hide_img1: float = 0.01,
    p_hide_img2: float = 0.01,
    diameter_range: Tuple[float, float] = (0.1, 1.0),
    intensity_range: Tuple[float, float] = (50, 200),
    rho_range: Tuple[float, float] = (-0.99, 0.99),
    dt: float = 1.0,
):
    """Generates a batch of image pairs from a flow field.

    Args:
        key: jax.random.PRNGKey
            Random key for reproducibility.
        flow_field: jnp.ndarray
            Array of shape (H, W, 2) containing the velocity field
            at each grid_step, in grid_steps per second.
        position_bounds: Tuple[int, int]
            (height, width) of the output big image in pixels.
        image_shape: Tuple[int, int]
            (height, width) of the output image in pixels.
        num_images: int
            Number of image pairs to generate.
        img_offset: Tuple[int, int]
            Offset to apply to the generated images.
        num_particles: int
            Number of particles to generate in each image.
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
        DEBUG: bool
            If True, does input validation and prints debug information.

    Returns:
        tuple: Two image batches (num_images, H, W) each.
    """
    # scale factors for particle positions
    alpha1 = flow_field.shape[0] / position_bounds[0]
    alpha2 = flow_field.shape[1] / position_bounds[1]

    def bodyfun(i, state):
        first_imgs, second_imgs, key = state

        # Split the key for randomness
        key_i = jax.random.fold_in(key, i)
        subkey1, subkey2, subkey3, subkey4, subkey5 = jax.random.split(key_i, 5)

        # generate random masks
        mask_img1 = jax.random.bernoulli(
            subkey1, 1.0 - p_hide_img1, shape=(num_particles,)
        )
        mask_img2 = jax.random.bernoulli(
            subkey2, 1.0 - p_hide_img2, shape=(num_particles,)
        )

        H, W = position_bounds
        # Generate random particle positions
        particle_positions = jax.random.uniform(
            subkey3, (num_particles, 2), minval=0.0, maxval=1.0
        ) * jnp.array([H, W])

        if DEBUG:
            input_check_img_gen_from_data(
                particle_positions=particle_positions,
                image_shape=position_bounds,
                diameter_range=diameter_range,
                intensity_range=intensity_range,
                rho_range=rho_range,
            )

        # First image generation
        first_img = img_gen_from_data(
            key=subkey4,
            particle_positions=particle_positions * mask_img1[:, None],
            image_shape=position_bounds,
            diameter_range=diameter_range,
            intensity_range=intensity_range,
            rho_range=rho_range,
        )

        if DEBUG:
            input_check_apply_flow(
                particle_positions=particle_positions, flow_field=flow_field, dt=dt
            )

        # Divide the x coordinates by 2 to match the flow field
        particle_positions = jnp.array(
            [
                particle_positions[:, 0] * alpha1,
                particle_positions[:, 1] * alpha2,
            ]
        ).T

        # Apply flow field to particle positions
        final_positions = apply_flow_to_particles(
            particle_positions=particle_positions, flow_field=flow_field, dt=dt
        )

        # Rescale the x coordinates back to the original scale
        final_positions = jnp.array(
            [
                final_positions[:, 0] / alpha1,
                final_positions[:, 1] / alpha2,
            ]
        ).T

        if DEBUG:
            input_check_img_gen_from_data(
                particle_positions=final_positions,
                image_shape=position_bounds,
                diameter_range=diameter_range,
                intensity_range=intensity_range,
                rho_range=rho_range,
            )

        # Second image generation
        second_img = img_gen_from_data(
            key=subkey5,
            particle_positions=final_positions * mask_img2[:, None],
            image_shape=position_bounds,
            diameter_range=diameter_range,
            intensity_range=intensity_range,
            rho_range=rho_range,
        )

        # Update the images
        first_imgs = first_imgs.at[i].set(first_img)
        second_imgs = second_imgs.at[i].set(second_img)

        return first_imgs, second_imgs, key

    # Initialize state: empty arrays to collect images and the RNG key
    first_imgs = jnp.zeros((num_images, *position_bounds))
    second_imgs = jnp.zeros((num_images, *position_bounds))

    # fix the key shape
    key = jnp.reshape(key, (-1, key.shape[-1]))[0]

    init_state = (first_imgs, second_imgs, key)
    final_imgs, final_imgs2, _ = jax.lax.fori_loop(0, num_images, bodyfun, init_state)

    # Crop the images to the desired shape
    final_imgs = final_imgs[
        :,
        img_offset[0] : image_shape[0] + img_offset[0],
        img_offset[1] : image_shape[1] + img_offset[1],
    ]

    final_imgs2 = final_imgs2[
        :,
        img_offset[0] : image_shape[0] + img_offset[0],
        img_offset[1] : image_shape[1] + img_offset[1],
    ]

    return final_imgs, final_imgs2


def input_check_gen_img_from_flow(
    key: jax.random.PRNGKey,
    flow_field: jnp.ndarray,
    position_bounds: Tuple[int, int] = (512, 512),
    image_shape: Tuple[int, int] = (256, 256),
    img_offset: Tuple[int, int] = (20, 20),
    num_images: int = 300,
    num_particles: int = 10000,
    p_hide_img1: float = 0.01,
    p_hide_img2: float = 0.01,
    diameter_range: Tuple[float, float] = (0.1, 1.0),
    intensity_range: Tuple[float, float] = (50, 200),
    rho_range: Tuple[float, float] = (-0.99, 0.99),
    dt: float = 1.0,
):
    """Check the input arguments for generate_images_from_flow.

    Args:
        key: jax.random.PRNGKey
            Random key for reproducibility.
        flow_field: jnp.ndarray
            Array of shape (H, W, 2) containing the velocity field
            at each grid_step.
        position_bounds: Tuple[int, int]
            (height, width) of the output big image.
        image_shape: Tuple[int, int]
            (height, width) of the output image.
        num_images: int
            Number of image pairs to generate.
        img_offset: Tuple[int, int]
            Offset to apply to the generated images.
        num_particles: int
            Number of particles to generate in each image.
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
        or flow_field.ndim != 3
        or flow_field.shape[2] != 2
    ):
        raise ValueError("Flow_field must be a 3D jnp.ndarray with shape (H, W, 2).")
    if len(diameter_range) != 2 or not all(d > 0 for d in diameter_range):
        raise ValueError("diameter_range must be a tuple of two positive floats.")
    if len(intensity_range) != 2 or not all(i >= 0 for i in intensity_range):
        raise ValueError("intensity_range must be a tuple of two positive floats.")
    if len(rho_range) != 2 or not all(-1 <= i <= 1 for i in rho_range):
        raise ValueError("rho_range must be a tuple of two floats between -1 and 1.")
    if not isinstance(num_images, int) or num_images <= 0:
        raise ValueError("num_images must be a positive integer.")
    if not isinstance(num_particles, int) or num_particles <= 0:
        raise ValueError("num_particles must be a positive integer.")
    if not (0 <= p_hide_img1 <= 1):
        raise ValueError("p_hide_img1 must be between 0 and 1.")
    if not (0 <= p_hide_img2 <= 1):
        raise ValueError("p_hide_img2 must be between 0 and 1.")
    if not isinstance(dt, (int, float)):
        raise ValueError("dt must be a scalar (int or float)")

    logger.debug("Input arguments of generate_images_from_flow are valid.")
    logger.debug(f"Flow field shape: {flow_field.shape}")
    logger.debug(f"Image shape: {image_shape}")
    logger.debug(f"Big image shape: {position_bounds}")
    logger.debug(f"Number of images: {num_images}")
    logger.debug(f"Number of particles: {num_particles}")
    logger.debug(f"Probability of hiding particles in image 1: {p_hide_img1}")
    logger.debug(f"Probability of hiding particles in image 2: {p_hide_img2}")
    logger.debug(f"Particle diameter range: {diameter_range}")
    logger.debug(f"Intensity range: {intensity_range}")
    logger.debug(f"Correlation coefficient range: {rho_range}")
    logger.debug(f"Time step (dt): {dt}")
