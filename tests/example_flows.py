"""Example flow functions for testing purposes."""

from collections.abc import Callable
import jax.numpy as jnp


def horizontal_flow(t: float, x: float, y: float) -> tuple[float, float]:
    """A simple horizontal flow: constant unit speed in the x-direction.

    The flow field is given by:
    u = 1
    v = 0

    Args:
        t (float): The time
        x (float): The x-coordinate.
        y (float): The y-coordinate.

    Returns:
        Tuple[float, float]: The flow field components.
    """
    return (1.0, 0.0)


def pipe_horizontal_flow(t: float, x: float, y: float) -> tuple[float, float]:
    """A parabolic (pipe) horizontal flow.

    The flow field is given by:
    u = 1 - y**2
    v = 0

    Args:
        t (float): The time
        x (float): The x-coordinate.
        y (float): The y-coordinate.

    Returns:
        Tuple[float, float]: The flow field components.
    """
    return (1 - y**2, 0.0)


def vortex_flow(
    t: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """A time-varying vortex flow.

    The flow field is given by:
    u = -y * cos(t)
    v =  x * cos(t)

    Args:
        t (float): The time
        x (float): The x-coordinate.
        y (float): The y-coordinate.

    Returns:
        Tuple[float, float]: The flow field components.
    """
    f = 1
    u = -y * jnp.cos(2 * jnp.pi * f * t)
    v = x * jnp.cos(2 * jnp.pi * f * t)
    return u, v


def get_flow_function(
    selected_flow: str, image_shape: tuple[int, int] = (128, 128)
) -> Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray], 
    tuple[jnp.ndarray, jnp.ndarray]
]:
    """Generate a flow field for testing purposes.

    It creates a flow field function based on the selected_flow.

    Args:
        selected_flow: The selected flow field type.
        image_shape: The image shape.

    Returns:
        Tuple[float, float]: The flow field components.
    """
    match selected_flow:
        case "horizontal":
            return lambda t, x, y: (10.0, 0.0)
        case "vertical":
            return lambda t, x, y: (0.0, 10.0)
        case "diagonal":
            return lambda t, x, y: (10.0, 10.0)
        case "no_flow":
            return lambda t, x, y: (0.0, 0.0)
        case "pipe_horizontal":
            return lambda t, x, y: (20 - 0.004 * (y - image_shape[1] / 2) ** 2, 0.0)
        case "vortex":
            return lambda t, x, y: (
                -((y - 64) * 10.0 / 64.0 * jnp.cos(2 * jnp.pi * t)),
                ((x - 64) * 10.0 / 64.0 * jnp.cos(2 * jnp.pi * t)),
            )
        case _:
            raise ValueError("Invalid flow selected")
