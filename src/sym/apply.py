"""Apply a flow field to an image of particles."""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from src.utils import bilinear_interpolate


def apply_flow_to_image(
    image: jnp.ndarray,
    flow_field: Callable[[float, float, float], Tuple[float, float]],
    t: float = 0.0,
) -> jnp.ndarray:
    """Warp a 2D image of particles according to a given flow field.

    For each pixel (x, y) in the output image, we compute a displacement (u, v)
    from `flow_field(t, x, y)`, then sample from the input image at
    (x_s, y_s) = (x - u, y - v) via bilinear interpolation.

    Args:
        image (jnp.ndarray): 2D array (H, W) representing the input particle image.
        flow_field (Callable[[float, float, float], Tuple[float, float]]):
            Function that takes (x, y, t) and returns (u, v) displacement.
                - x, y: coordinates
                - t: time parameter (or any scalar)
        t (float, optional): Time parameter passed to flow_field. Defaults to 0.0.

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

    # Backward mapping: (x_s, y_s) = (x - u, y - v)
    # x_grid, y_grid are (H, W)
    x_s = x_grid - u
    y_s = y_grid - v

    # Interpolate from the original image at these source coords
    warped = bilinear_interpolate(image, x_s, y_s)
    return warped
