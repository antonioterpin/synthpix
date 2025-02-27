import os

import jax.numpy as jnp
import pytest

from src.analyses.plotting import plot_flow_field
from src.sym.example_flows import horizontal_flow, pipe_horizontal_flow, vortex_flow


@pytest.mark.parametrize(
    "flow_func, name",
    [
        (horizontal_flow, "horizontal_flow"),
        (pipe_horizontal_flow, "pipe_horizontal_flow"),
        (vortex_flow, "vortex_flow"),
    ],
)
def test_visualize_flows(flow_func, name):
    """Tests the GIF generator for each flow field function."""
    # Define a set of time points for visualization.
    t = jnp.linspace(0, 5, 10)
    xs = jnp.linspace(0, 1, 20)
    ys = jnp.linspace(-1, 1, 20)
    x, y = jnp.meshgrid(xs, ys)

    gif_path = plot_flow_field(x, y, t, flow_func, name)
    # Check that the GIF file was created.
    assert os.path.exists(gif_path)
