"""Plotting functions for the data analysis."""

import os
from typing import Callable, Tuple

import imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def plot_flow_field(
    xs: jnp.ndarray,
    ys: jnp.ndarray,
    ts: jnp.ndarray,
    func: Callable[[float, float, float], Tuple[float, float]],
    name: str = "Flow Field",
) -> str:
    """Plots the flow field at multiple time points and saves the images as a GIF.

    Args:
    time_points (jnp.ndarray): An array of time points to visualize.
    func (Callable[[float, float, float], Tuple[float, float]]):
        A function that computes the flow field at a given time.
    name (str): The name of the flow field (used for the filename).

    Returns:
    str: The path to the saved GIF.
    """
    results_dir = "results/viz"
    tmp_dir = f"{results_dir}/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    images = []
    for t in ts:
        u, v = jax.vmap(func, in_axes=(None, 0, 0))(t, xs.flatten(), ys.flatten())
        # Make sure the arrays are NumPy arrays for Matplotlib.
        x_np, y_np = np.array(xs), np.array(ys)
        u_np, v_np = np.array(u).reshape(x_np.shape), np.array(v).reshape(y_np.shape)

        plt.figure(figsize=(5, 5))
        plt.quiver(x_np, y_np, u_np, v_np, pivot="mid", color="r")
        plt.title(f"{name} at t={t:.2f}")
        plt.xlim(xs.min(), xs.max())
        plt.ylim(ys.min(), ys.max())
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()

        # Save the current figure to a temporary file.
        tmpfile = f"results/viz/tmp/{name}_{t:.2f}.png"
        plt.savefig(tmpfile)
        plt.close()

        # Read the saved image and append to our list.
        images.append(imageio.v2.imread(tmpfile))

    # Save the sequence of images as a GIF.
    gif_path = f"{results_dir}/{name}.gif"
    imageio.mimsave(gif_path, images, duration=0.2)

    import glob

    if os.path.exists("results/viz/tmp"):
        for tmpfile in glob.glob("results/viz/tmp/*"):
            os.remove(tmpfile)
        os.rmdir("results/viz/tmp")

    return gif_path
