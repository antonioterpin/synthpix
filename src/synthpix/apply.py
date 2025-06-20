"""Apply a flow field to an image of particles or directly to the particles."""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from .utils import bilinear_interpolate, trilinear_interpolate


def apply_flow_to_image_forward(
    image: jnp.ndarray,
    flow_field: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Warp a 2D image of particles according to a given flow field.

    For each pixel (y, x) in the output image, we compute a velocity (u, v)
    from `flow_field[t, y, x]`, then sample from the input image at
    (y_s, x_s) = (y - v * dt, x - u * dt) via bilinear interpolation.

    Args:
        image (jnp.ndarray): 2D array (H, W) representing the input particle image.
        flow_field (jnp.ndarray): 3D array (H, W, 2) representing the velocity field.
        dt (float): Time step for the backward mapping.

    Returns:
        jnp.ndarray: A new 2D array of shape (H, W) with the particles displaced.
    """
    H, W = image.shape
    y_grid, x_grid = jnp.indices((H, W))

    u = flow_field[..., 0]
    v = flow_field[..., 1]

    # Forward mapping: (x_d, y_d) = (x + u * dt, y + v * dt)
    x_d = x_grid + u * dt
    y_d = y_grid + v * dt

    new_image = jnp.zeros_like(image)

    def deposit_pixel(new_image, x_src, y_src, val):
        x0 = jnp.floor(x_src).astype(int)
        y0 = jnp.floor(y_src).astype(int)

        wx = x_src - x0
        wy = y_src - y0

        def in_bounds(x, y):
            return (x >= 0) & (x < W) & (y >= 0) & (y < H)

        for dx, dy, weight in [
            (0, 0, (1 - wx) * (1 - wy)),
            (1, 0, wx * (1 - wy)),
            (0, 1, (1 - wx) * wy),
            (1, 1, wx * wy),
        ]:
            xi = x0 + dx
            yi = y0 + dy
            cond = in_bounds(xi, yi)
            new_image = jax.lax.cond(
                cond,
                lambda img: img.at[yi, xi].add(val * weight),
                lambda img: img,
                operand=new_image,
            )
        return new_image

    def body_fn(i, new_image):
        y, x = divmod(i, W)
        return deposit_pixel(new_image, x_d[y, x], y_d[y, x], image[y, x])

    new_image = jax.lax.fori_loop(0, H * W, body_fn, new_image)
    return new_image


def apply_flow_to_image_backward(
    image: jnp.ndarray,
    flow_field: jnp.ndarray,
    dt: float = 1.0,
) -> jnp.ndarray:
    """Warp a 2D image of particles according to a given flow field.

    For each pixel (y, x) in the output image, we compute a velocity (u, v)
    from `flow_field[y, x]`, then sample from the input image at
    (y_s, x_s) = (y - v * dt, x - u * dt) via bilinear interpolation.

    Args:
        image (jnp.ndarray): 2D array (H, W) representing the input particle image.
        flow_field (jnp.ndarray): 3D array (H, W, 2) representing the velocity field.
        dt (float): Time step for the backward mapping.

    Returns:
        jnp.ndarray: A new 2D array of shape (H, W) with the particles displaced.
    """
    H, W = image.shape

    # Meshgrid of pixel coordinates
    ys, xs = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")

    # Real sample locations
    dx = flow_field[..., 0]
    dy = flow_field[..., 1]
    x_f = xs - dx * dt
    y_f = ys - dy * dt

    # Bilinear interpolation to sample the image at (y_f, x_f)
    return bilinear_interpolate(
        image,
        x_f,
        y_f,
    )


def apply_flow_to_image_callable(
    image: jnp.ndarray,
    flow_field: Callable[[float, float, float], Tuple[float, float]],
    t: float = 0.0,
    dt: float = 1.0,
    forward: bool = False,
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
        t (float): Time parameter passed to flow_field.
        dt (float): Time step for the backward mapping.

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
    if forward:
        return apply_flow_to_image_forward(image, uv, dt)
    return apply_flow_to_image_backward(image, uv, dt)


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
            field at each grid_step in length measure unit / s.
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
            u = bilinear_interpolate(flow_field[..., 0], x, y) * flow_field_res_x
            v = bilinear_interpolate(flow_field[..., 1], x, y) * flow_field_res_y

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
