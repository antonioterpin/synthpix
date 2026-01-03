"""Utility functions for the vision module."""

import os
from collections.abc import Callable, Sequence

import goggles as gg
import jax
import jax.numpy as jnp

DEBUG_JIT = False
SYNTHPIX_SCOPE = "synthpix"

load_configuration = gg.load_configuration


def match_histogram(
    source: jnp.ndarray, template_hist: jnp.ndarray
) -> jnp.ndarray:
    """Match the histogram of `source` to a desired histogram.

    Args:
        source: The source image (uint8 or float) with values in [0,255].
        template_hist: 1D target histogram of length 256,
            summing to number of pixels in source.

    Returns:
        The source image with its histogram matched to the target histogram.
    """
    # Flatten source and cast to float32 for computation
    flat = source.ravel().astype(jnp.float32)

    # Implicit bin edges for intensities [0..255]
    bins = jnp.arange(257, dtype=jnp.float32)

    # Source histogram counts
    s_counts, _ = jnp.histogram(flat, bins=bins)

    # Compute source CDF (normalized)
    s_cdf = jnp.cumsum(s_counts, dtype=jnp.float32)
    s_cdf = s_cdf / s_cdf[-1]

    # Compute template CDF (normalized)
    t_cdf = jnp.cumsum(template_hist.astype(jnp.float32), dtype=jnp.float32)
    t_cdf = t_cdf / t_cdf[-1]

    # Discrete levels 0..255
    levels = jnp.arange(256, dtype=jnp.int32)

    # Digitize source pixels into bin indices [0..255]
    idx = jnp.digitize(flat, bins) - 1
    idx = jnp.clip(idx, 0, 255)

    # Map pixels to source CDF quantiles
    quantiles = s_cdf[idx]

    # Map quantiles to new levels via searchsorted on template CDF
    new_idx = jnp.searchsorted(t_cdf, quantiles, side="left")
    new_idx = jnp.clip(new_idx, 0, 255)

    # Gather new pixel values and cast to original dtype
    matched = levels[new_idx].astype(source.dtype)

    # Reshape back to original image shape
    return matched.reshape(source.shape)


def bilinear_interpolate(
    image: jnp.ndarray, x_f: jnp.ndarray, y_f: jnp.ndarray
) -> jnp.ndarray:
    """Perform bilinear interpolation at floating-point pixel coordinates.

    Args:
        image: 2D image to sample from, of shape (H, W).
        x_f: 2D array of floating-point x-coordinates
        y_f: 2D array of floating-point y-coordinates

    Returns:
        Interpolated intensities at each (y, x) location, of shape (H, W).
    """
    H, W = image.shape

    # Clamp x_f and y_f to be within the image bounds
    x_f_clamped = jnp.clip(x_f, 0.0, W - 1.0)
    y_f_clamped = jnp.clip(y_f, 0.0, H - 1.0)

    # Integer neighbors & clamping
    x0 = jnp.clip(jnp.floor(x_f).astype(jnp.int32), 0, W - 1)
    x1 = jnp.clip(x0 + 1, 0, W - 1)
    y0 = jnp.clip(jnp.floor(y_f).astype(jnp.int32), 0, H - 1)
    y1 = jnp.clip(y0 + 1, 0, H - 1)

    # Fractional weights
    wx = x_f_clamped - x0
    wy = y_f_clamped - y0

    # Gather neighboring pixels
    i00 = image[y0, x0]
    i10 = image[y0, x1]
    i01 = image[y1, x0]
    i11 = image[y1, x1]

    # Bilinear
    return (
        (1 - wx) * (1 - wy) * i00
        + wx * (1 - wy) * i10
        + (1 - wx) * wy * i01
        + wx * wy * i11
    )


def trilinear_interpolate(
    volume: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray
) -> jnp.ndarray:
    """Perform trilinear interpolation at floating-point pixel coordinates.

    Args:
        volume: 3D volume to sample from, of shape (D, H, W).
        x: Array of floating-point x-coordinates.
        y: Array of floating-point y-coordinates.
        z: Array of floating-point z-coordinates.

    Returns:
        Interpolated intensities at each (z, y, x) location.
    """
    D, H, W = volume.shape

    # Floor and ceil indices for each coordinate
    x0 = jnp.floor(x).astype(int)
    y0 = jnp.floor(y).astype(int)
    z0 = jnp.floor(z).astype(int)

    x1 = jnp.ceil(x).astype(int)
    y1 = jnp.ceil(y).astype(int)
    z1 = jnp.ceil(z).astype(int)

    # Clamp indices to be within volume boundaries
    x0 = jnp.clip(x0, 0, W - 1)
    x1 = jnp.clip(x1, 0, W - 1)
    y0 = jnp.clip(y0, 0, H - 1)
    y1 = jnp.clip(y1, 0, H - 1)
    z0 = jnp.clip(z0, 0, D - 1)
    z1 = jnp.clip(z1, 0, D - 1)

    # Compute interpolation weights for each axis
    alpha_x = x - jnp.floor(x)
    alpha_y = y - jnp.floor(y)
    alpha_z = z - jnp.floor(z)

    # Retrieve intensities from the eight corners of the cube
    ia = volume[z0, y0, x0]
    ib = volume[z0, y0, x1]
    ic = volume[z0, y1, x0]
    id_corner = volume[z0, y1, x1]
    ie = volume[z1, y0, x0]
    if_corner = volume[z1, y0, x1]
    ig = volume[z1, y1, x0]
    ih = volume[z1, y1, x1]

    # Compute weights for each corner
    wa = (1.0 - alpha_x) * (1.0 - alpha_y) * (1.0 - alpha_z)
    wb = alpha_x * (1.0 - alpha_y) * (1.0 - alpha_z)
    wc = (1.0 - alpha_x) * alpha_y * (1.0 - alpha_z)
    wd = alpha_x * alpha_y * (1.0 - alpha_z)
    we = (1.0 - alpha_x) * (1.0 - alpha_y) * alpha_z
    wf = alpha_x * (1.0 - alpha_y) * alpha_z
    wg = (1.0 - alpha_x) * alpha_y * alpha_z
    wh = alpha_x * alpha_y * alpha_z

    # Compute the weighted sum of the corner intensities
    return (
        wa * ia
        + wb * ib
        + wc * ic
        + wd * id_corner
        + we * ie
        + wf * if_corner
        + wg * ig
        + wh * ih
    )


def generate_array_flow_field(
    flow_f: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    grid_shape: tuple[int, int] = (128, 128),
) -> jnp.ndarray:
    """Generate a array flow field from a flow field function.

    Args:
        flow_f: The flow field function.
        grid_shape: The shape of the grid.

    Returns:
        The array flow field.
    """
    # Get the image shape
    H, W = grid_shape
    # Create pixel coordinate grids: y in [0..H-1], x in [0..W-1]
    rows = jnp.arange(H)
    cols = jnp.arange(W)

    # vmap over both axes, and apply the flow function at time t=1
    arr = jax.vmap(
        lambda i: jax.vmap(lambda j: jnp.array(flow_f(jnp.ones(1), i, j)))(cols)
    )(rows)

    return arr


def flow_field_adapter(
    flow_fields: jnp.ndarray,
    new_flow_field_shape: tuple[int, int] = (256, 256),
    image_shape: tuple[int, int] = (256, 256),
    img_offset: tuple[int, int] = (0, 0),
    resolution: float = 1.0,
    res_x: float = 1.0,
    res_y: float = 1.0,
    position_bounds: tuple[int, int] = (256, 256),
    position_bounds_offset: tuple[int | float, int | float] = (0, 0),
    batch_size: int = 1,
    output_units: str = "pixels",
    dt: float = 1.0,
    zero_padding: tuple[int, int] = (0, 0),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Adapts a batch of flow fields to a new shape and resolution.

    Args:
        flow_fields: The original flow field batch to be adapted.
        new_flow_field_shape: The desired shape of the new flow fields.
        image_shape: The shape of the images.
        img_offset: The offset of the images from the position bounds in pixels.
        resolution: Resolution of the images in pixels per unit length.
        res_x: Flow field resolution in the x direction
            [grid steps/length measure units].
        res_y: Flow field resolution in the y direction
            [grid steps/length measure units].
        position_bounds: The bounds of the flow field in the x and y directions.
        position_bounds_offset: The offset of the position bounds in
            length measure units.
        batch_size: The desired batch size of the output flow fields.
        output_units: The units of the output flow fields.
            Can be "pixels" or "measure units per second".
        dt: The time step for the flow field adaptation.
        zero_padding:
            The amount of zero-padding to apply to the
            top and left edges of the flow field.

    Returns:
        - The adapted flow fields of shape (batch_size, new_h, new_w, 2).
        - The cropped flow field region of position bounds.
    """
    new_h, new_w = new_flow_field_shape

    def process_single(flow: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Apply zero-padding
        pad_y, pad_x = zero_padding
        flow = jnp.pad(
            flow,
            pad_width=((pad_y, 0), (pad_x, 0), (0, 0)),
            mode="edge",
        )

        # flow_resized
        # Create the grid for interpolation
        flow_resized_start = (
            position_bounds_offset[0] * res_y
            + img_offset[0] / resolution * res_y,
            position_bounds_offset[1] * res_x
            + img_offset[1] / resolution * res_x,
        )
        flow_resized_end = (
            flow_resized_start[0] + image_shape[0] / resolution * res_y - 1,
            flow_resized_start[1] + image_shape[1] / resolution * res_x - 1,
        )

        flow_resized_vec_y = jnp.linspace(
            flow_resized_start[0], flow_resized_end[0], new_h
        )
        flow_resized_vec_x = jnp.linspace(
            flow_resized_start[1], flow_resized_end[1], new_w
        )

        flow_resized_grid_x, flow_resized_grid_y = jnp.meshgrid(
            flow_resized_vec_x, flow_resized_vec_y
        )

        flow_resized_x = bilinear_interpolate(
            flow[..., 0], flow_resized_grid_x, flow_resized_grid_y
        )

        flow_resized_y = bilinear_interpolate(
            flow[..., 1], flow_resized_grid_x, flow_resized_grid_y
        )

        flow_resized = jnp.stack((flow_resized_x, flow_resized_y), axis=-1)

        # flow_position_bounds
        # Create the grid for interpolation
        flow_position_bounds_start = (
            position_bounds_offset[0] * res_y,
            position_bounds_offset[1] * res_x,
        )
        flow_position_bounds_end = (
            flow_position_bounds_start[0]
            + position_bounds[0] / resolution * res_y
            - 1,
            flow_position_bounds_start[1]
            + position_bounds[1] / resolution * res_x
            - 1,
        )
        flow_position_bounds_vec_y = jnp.linspace(
            flow_position_bounds_start[0],
            flow_position_bounds_end[0],
            max(1, int(position_bounds[0] / resolution * res_y)),
        )
        flow_position_bounds_vec_x = jnp.linspace(
            flow_position_bounds_start[1],
            flow_position_bounds_end[1],
            max(1, int(position_bounds[1] / resolution * res_x)),
        )
        flow_position_bounds_grid_x, flow_position_bounds_grid_y = jnp.meshgrid(
            flow_position_bounds_vec_x, flow_position_bounds_vec_y
        )
        flow_position_bounds_x = bilinear_interpolate(
            flow[..., 0],
            flow_position_bounds_grid_x,
            flow_position_bounds_grid_y,
        )
        flow_position_bounds_y = bilinear_interpolate(
            flow[..., 1],
            flow_position_bounds_grid_x,
            flow_position_bounds_grid_y,
        )
        flow_position_bounds = jnp.stack(
            (flow_position_bounds_x, flow_position_bounds_y), axis=-1
        )

        if output_units == "pixels":
            flow_resized = flow_resized.at[..., 0].multiply(resolution * dt)
            flow_resized = flow_resized.at[..., 1].multiply(resolution * dt)

        return flow_resized, flow_position_bounds

    adapted_flows, flow_bounds = jax.vmap(process_single)(flow_fields)

    n = adapted_flows.shape[0]
    repeats = (batch_size + n - 1) // n
    tiled = jnp.tile(adapted_flows, (repeats, 1, 1, 1))

    return tiled[:batch_size, ...], flow_bounds


def input_check_flow_field_adapter(  # noqa: PLR0912
    flow_field: jnp.ndarray,
    new_flow_field_shape: tuple[int, int],
    image_shape: tuple[int, int],
    img_offset: tuple[int, int],
    resolution: float,
    res_x: float,
    res_y: float,
    position_bounds: tuple[float, float],
    position_bounds_offset: tuple[float, float],
    batch_size: int,
    output_units: str,
    dt: float,
    zero_padding: tuple[int, ...],
) -> None:
    """Checks the input arguments of the flow field adapter function.

    Args:
        flow_field: The original flow field batch to be adapted.
        new_flow_field_shape: The desired shape of the new flow fields.
        image_shape: The shape of the images.
        img_offset: The offset of the images.
        resolution: Resolution of the images in pixels per unit length.
        res_x: Flow field resolution in the x direction
            [grid steps/length measure units].
        res_y: Flow field resolution in the y direction
            [grid steps/length measure units].
        position_bounds: The bounds of the flow field in the x and y directions.
        position_bounds_offset:
            The offset of the flow field in the x and y directions.
        batch_size: The desired batch size of the output flow fields.
        output_units: The units of the output flow fields.
            Can be "pixels" or "measure units per second".
        dt: The time step for the flow field adaptation.
        zero_padding: The amount of zero-padding to apply to the top and left
            edges of the flow field.

    Raises:
        ValueError: If any input parameter has invalid type, shape, or value.
    """
    if not isinstance(flow_field, jnp.ndarray):
        raise ValueError("flow_field must be a jnp.ndarray.")
    if flow_field.ndim != 4:
        raise ValueError(
            "flow_field must be a 4D jnp.ndarray with shape (N, H, W, 2), "
            f"got {flow_field.shape}."
        )
    if flow_field.shape[-1] != 2:
        raise ValueError(
            f"flow_field must have shape (N, H, W, 2), got {flow_field.shape}."
        )

    if (
        not isinstance(new_flow_field_shape, tuple)
        or len(new_flow_field_shape) != 2
        or not all(isinstance(s, int) and s > 0 for s in new_flow_field_shape)
    ):
        raise ValueError(
            "new_flow_field_shape must be a tuple of two positive integers, "
            f"got {new_flow_field_shape}."
        )

    if not isinstance(image_shape, tuple) or len(image_shape) != 2:
        raise ValueError(
            "image_shape must be a tuple of two positive integers."
        )
    if not all(isinstance(s, int) and s > 0 for s in image_shape):
        raise ValueError("image_shape must contain two positive integers.")

    if not isinstance(img_offset, tuple) or len(img_offset) != 2:
        raise ValueError(
            "img_offset must be a tuple of two non-negative numbers."
        )
    if not all(isinstance(s, int | float) and s >= 0 for s in img_offset):
        raise ValueError("img_offset must contain two non-negative numbers.")

    if not isinstance(resolution, int | float) or resolution <= 0:
        raise ValueError("resolution must be a positive number.")

    if not isinstance(res_x, int | float) or res_x <= 0:
        raise ValueError("res_x must be a positive number.")

    if not isinstance(res_y, int | float) or res_y <= 0:
        raise ValueError("res_y must be a positive number.")

    if not isinstance(position_bounds, tuple) or len(position_bounds) != 2:
        raise ValueError(
            "position_bounds must be a tuple of two positive numbers."
        )
    if not all(isinstance(s, int | float) and s > 0 for s in position_bounds):
        raise ValueError("position_bounds must contain two positive numbers.")

    if (
        not isinstance(position_bounds_offset, tuple)
        or len(position_bounds_offset) != 2
    ):
        raise ValueError(
            "position_bounds_offset must be a tuple of two "
            "non-negative numbers."
        )
    if not all(
        isinstance(s, int | float) and s >= 0 for s in position_bounds_offset
    ):
        raise ValueError(
            "position_bounds_offset must contain two non-negative numbers."
        )

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    if not isinstance(output_units, str) or output_units not in [
        "pixels",
        "measure units per second",
    ]:
        raise ValueError(
            "output_units must be either 'pixels' or 'measure units "
            "per second'."
        )

    if not isinstance(dt, int | float) or dt <= 0:
        raise ValueError("dt must be a positive number.")

    if (
        not isinstance(zero_padding, tuple)
        or len(zero_padding) != 2
        or not all(isinstance(s, int) and s >= 0 for s in zero_padding)
    ):
        raise ValueError(
            "zero_padding must be a tuple of two non-negative integers."
        )


def discover_leaf_dirs(
    paths: Sequence[str], *, follow_symlinks: bool = False
) -> list[str]:
    """Return every directory in `paths` containing no sub-directories.

    Args:
        paths: A sequence of file or directory paths.
        follow_symlinks: Whether to follow symlinks when checking for subdirs.

    Returns:
        A list of directory paths that are leaves (have no subdirectories).
    """
    dir_paths = {
        os.path.normpath(os.path.dirname(p)) for p in paths
    }  # dedupe upfront
    leaves: list[str] = []

    for d in dir_paths:
        try:
            with os.scandir(d) as it:
                # Early-exit on the first subdirectory
                if any(
                    entry.is_dir(follow_symlinks=follow_symlinks)
                    for entry in it
                ):
                    continue
            leaves.append(d)  # No subdirs found
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            # Skip dirs that vanished, aren't dirs, or we can't read
            continue

    return leaves
