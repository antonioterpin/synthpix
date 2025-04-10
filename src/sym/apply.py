"""Apply a flow field to an image of particles or directly to the particles."""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from src.utils import bilinear_interpolate, trilinear_interpolate


def apply_flow_to_image(
    image: jnp.ndarray,
    flow_field: Callable[[float, float, float], Tuple[float, float]],
    t: float = 0.0,
    dt: float = 1.0,
) -> jnp.ndarray:
    """Warp a 2D image of particles according to a given flow field.

    For each pixel (y, x) in the output image, we compute a velocity (u, v)
    from `flow_field(t, x, y)`, then sample from the input image at
    (y_s, x_s) = (y - v * dt, x - u * dt) via bilinear interpolation.

    Args:
        image (jnp.ndarray): 2D array (H, W) representing the input particle image.
        flow_field (Callable[[float, float, float], Tuple[float, float]]):
            Function that takes (x, y, t) and returns (u, v) velocity.
                - x, y: coordinates
                - t: time parameter (or any scalar)
        t (float, optional): Time parameter passed to flow_field. Defaults to 0.0.
        dt (float, optional): Time step for the backward mapping. Defaults to 1.0.

    Returns:
        jnp.ndarray: A new 2D array of shape (H, W) with the particles displaced.
    """
    H, W = image.shape

    # Create pixel coordinate grids: y in [0..H-1], x in [0..W-1]
    # shapes: (H, W)
    y_grid, x_grid = jnp.indices((H, W))

    # vmap over both axes (first over rows, then over columns)
    flow_field_vmap = jax.vmap(
        jax.vmap(lambda y, x: jnp.array(flow_field(t, x, y)), in_axes=(0, 0)),
        in_axes=(0, 0),
    )
    # shape (H, W, 2)
    uv = flow_field_vmap(y_grid, x_grid)

    # Extract displacements
    u = uv[..., 0]
    v = uv[..., 1]

    # Backward mapping: (x_s, y_s) = (x - u * dt, y - v * dt)
    # x_grid, y_grid are (H, W)
    x_s = x_grid - u * dt
    y_s = y_grid - v * dt

    # Interpolate from the original image at these source coords
    warped = bilinear_interpolate(image, x_s, y_s)
    return warped


def input_check_apply_flow(
    particle_positions: jnp.ndarray,
    flow_field: jnp.ndarray,
    dt: float = 1.0,
    flow_field_res_x: float = 1.0,
    flow_field_res_y: float = 1.0,
    flow_field_res_z: float = 1.0,
) -> jnp.ndarray:
    """Check the input arguments for apply_flow_to_particles.

    Args:
        particle_positions: jnp.ndarray
            Array of shape (N, 2) or (N, 3) containing particle coordinates in grid_steps.
        flow_field: jnp.ndarray
            Array of shape (H, W, 2) or (H, W, 3) containing the velocity
            field at each grid_step.
        dt: float
            Time step for the simulation, used to scale the velocity
            to compute the displacement. Defaults to 1.0.
        flow_field_res_x: float
            Resolution of the flow field in the x direction
            in grid steps per length measure unit.
        flow_field_res_y: float
            Resolution of the flow field in the y direction
            in grid steps per length measure unit
        flow_field_res_z: float
            Resolution of the flow field in the z direction
            in grid steps per length measure unit
    """
    if (
        not isinstance(particle_positions, jnp.ndarray)
        or particle_positions.ndim != 2
        or particle_positions.shape[1] not in (2, 3)
    ):
        raise ValueError(
            "Particle_positions must be a 2D jnp.ndarray with shape (N, 2) or (N, 3)"
        )

    if (
        not isinstance(flow_field, jnp.ndarray)
        or flow_field.ndim != 3
        or flow_field.shape[2] not in (2, 3)
    ):
        raise ValueError(
            "Flow_field must be a 3D jnp.ndarray with shape (H, W, 2) or (H, W, 3)"
        )

    if particle_positions.shape[1] == 2 and flow_field.shape[2] != 2:
        raise ValueError("Particle positions are in 2D, but the flow field is in 3D.")

    if particle_positions.shape[1] == 3 and flow_field.shape[2] != 3:
        raise ValueError("Particle positions are in 3D, but the flow field is in 2D.")

    if not isinstance(dt, (int, float)):
        raise ValueError("dt must be a scalar (int or float)")

    if not isinstance(flow_field_res_x, (int, float)) or flow_field_res_x <= 0:
        raise ValueError("flow_field_res_x must be a positive scalar (int or float)")
    if not isinstance(flow_field_res_y, (int, float)) or flow_field_res_y <= 0:
        raise ValueError("flow_field_res_y must be a positive scalar (int or float)")
    if not isinstance(flow_field_res_z, (int, float)) or flow_field_res_z <= 0:
        raise ValueError("flow_field_res_z must be a positive scalar (int or float)")


def apply_flow_to_particles(
    particle_positions: jnp.ndarray,
    flow_field: jnp.ndarray,
    dt: float = 1.0,
    flow_field_res_x: float = 1.0,
    flow_field_res_y: float = 1.0,
    flow_field_res_z: float = 1.0,
) -> jnp.ndarray:
    """Applies a flow field to an array of particle coordinates.

    This function takes an array of particle coordinates and a flow field,
    and applies the flow field to the particles to compute their new positions.
    The function works for both 2D and 3D particle coordinates.

    Args:
        particle_positions: jnp.ndarray
            Array of shape (N, 2) or (N, 3) containing particle coordinates in grid_steps.
        flow_field: jnp.ndarray
            Array of shape (H, W, 2) or (H, W, 3) containing the velocity
            field at each grid_step.
        dt: float
            Time step for the simulation, used to scale the velocity
            to compute the displacement. Defaults to 1.0.
        flow_field_res_x: float
            Resolution of the flow field in the x direction
            in grid steps per length measure unit.
        flow_field_res_y: float
            Resolution of the flow field in the y direction
            in grid steps per length measure unit
        flow_field_res_z: float
            Resolution of the flow field in the z direction
            in grid steps per length measure unit

    Returns:
        jnp.ndarray: Array of shape (N, 2) or (N, 3)
        containing the new particle coordinates.
    """
    if particle_positions.shape[1] == 2:

        def update_position(
            yx: jnp.ndarray,
        ) -> jnp.ndarray:
            y, x = yx

            # Compute the velocity (u, v) for the given particle
            # with bilinear interpolation.
            # Note: velocity u corresponds to the x-direction and v to y.
            u = bilinear_interpolate(flow_field[..., 0], y, x) * flow_field_res_x
            v = bilinear_interpolate(flow_field[..., 1], y, x) * flow_field_res_y

            # Return the new position: (y + v * dt, x + u * dt)
            return jnp.array([y + v * dt, x + u * dt])

    else:

        def update_position(
            zyx: jnp.ndarray,
        ) -> jnp.ndarray:
            z, y, x = zyx

            # Compute the velocity (u, v, w) for the given particle
            # with trilinear interpolation.
            # Note: velocity u corresponds to the x-direction, v to y, and w to z.
            u = trilinear_interpolate(flow_field[..., 0], x, y, z) * flow_field_res_x
            v = trilinear_interpolate(flow_field[..., 1], x, y, z) * flow_field_res_y
            w = trilinear_interpolate(flow_field[..., 2], x, y, z) * flow_field_res_z

            # Return the new position: (z + w * dt, y + v * dt, x + u * dt)
            return jnp.array([z + w * dt, y + v * dt, x + u * dt])

    # Vectorize the function over all particles
    return jax.vmap(update_position)(particle_positions)
