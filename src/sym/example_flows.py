"""Example flow fields for testing the flow estimation algorithms."""

from typing import Tuple

import jax.numpy as jnp


def horizontal_flow(t: float, x: float, y: float) -> Tuple[float, float]:
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


def pipe_horizontal_flow(t: float, x: float, y: float) -> Tuple[float, float]:
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


def vortex_flow(t: float, x: float, y: float) -> Tuple[float, float]:
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
