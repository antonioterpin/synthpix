"""Utility functions for the vision module."""

import collections
import logging
import os
import signal
from typing import Tuple, Union

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

DEBUG = True
DEBUG_JIT = False


# Create a logger instance
logger = logging.getLogger(__name__)

# Configure the logging format
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="[%(levelname)s][%(asctime)s][%(filename)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def is_int(val: Union[int, float]) -> bool:
    """Check if a value is an integer.

    Args:
        val (Union[int, float]): The value to check.

    Returns:
        bool: True if the value is an integer, False otherwise.
    """
    if isinstance(val, int):
        return True
    if isinstance(val, float):
        if abs(val - int(val)) < 1e-6:
            return True
    return False


def load_configuration(file_path: str):
    """Load YAML configuration from file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def compute_image_scaled_height(
    target_width: int, image_width: int, image_height: int
) -> int:
    """Computes the height of an image given a target width keeping the aspect ratio.

    Args:
        target_width (int): The target width.
        image_width (int): The width of the image.
        image_height (int): The height of the image.

    Returns:
        int: The scaled height.
    """
    if target_width <= 0 or image_width <= 0 or image_height <= 0:
        raise ValueError("All dimensions must be positive integers.")
    return int(image_height * target_width / image_width)


def particles_per_pixel(image: jnp.ndarray, threshold: float = 0.1) -> float:
    """Estimates the number of particles per pixel in the image.

    Args:
        image: jnp.ndarray
            The input image of shape (H, W, 1).
        threshold: float
            The threshold to apply to the image.

    Returns:
        float: The estimated density.
    """
    # Simple, fast, metric for particle density is the fraction of pixels
    # above a threshold
    if not (0 <= threshold <= 255):
        raise ValueError("threshold must be a float in the range [0, 1].")
    return float(jnp.sum(image > threshold) / jnp.prod(image.size))


def bilinear_interpolate(
    image: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray
) -> jnp.ndarray:
    """Perform bilinear interpolation of `image` at floating-point pixel coordinates.

    Args:
        image: jnp.ndarray
            2D image to sample from, of shape (H, W).
        x: jnp.ndarray
            2D array of floating-point x-coordinates
        y: jnp.ndarray
            2D array of floating-point y-coordinates

    Returns:
        jnp.ndarray: Interpolated intensities at each (y, x) location, of shape (H, W).
    """
    H, W = image.shape

    # Floor of x, y
    x0 = jnp.floor(x).astype(int)
    y0 = jnp.floor(y).astype(int)
    # Ceiling (neighbor) of x, y
    x1 = jnp.ceil(x).astype(int)
    y1 = jnp.ceil(y).astype(int)

    # Clamp to image boundaries
    # Note: in this way, the positions need to be within the image boundaries
    x0 = jnp.clip(x0, 0, W - 1)
    x1 = jnp.clip(x1, 0, W - 1)
    y0 = jnp.clip(y0, 0, H - 1)
    y1 = jnp.clip(y1, 0, H - 1)

    # Compute interpolation weights
    alpha_x = x - jnp.floor(x)
    alpha_y = y - jnp.floor(y)

    # Gather intensities from the four corners
    Ia = image[y0, x0]  # top-left
    Ib = image[y0, x1]  # top-right
    Ic = image[y1, x0]  # bottom-left
    Id = image[y1, x1]  # bottom-right

    # Bilinear interpolation formula
    wa = (1.0 - alpha_x) * (1.0 - alpha_y)
    wb = alpha_x * (1.0 - alpha_y)
    wc = (1.0 - alpha_x) * alpha_y
    wd = alpha_x * alpha_y

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def trilinear_interpolate(
    volume: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray
) -> jnp.ndarray:
    """Perform trilinear interpolation of `volume` at floating-point pixel coordinates.

    Args:
        volume: jnp.ndarray
            3D volume to sample from, of shape (D, H, W).
        x: jnp.ndarray
            Array of floating-point x-coordinates.
        y: jnp.ndarray
            Array of floating-point y-coordinates.
        z: jnp.ndarray
            Array of floating-point z-coordinates.

    Returns:
        jnp.ndarray: Interpolated intensities at each (z, y, x) location.
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
    Ia = volume[z0, y0, x0]
    Ib = volume[z0, y0, x1]
    Ic = volume[z0, y1, x0]
    Id = volume[z0, y1, x1]
    Ie = volume[z1, y0, x0]
    If = volume[z1, y0, x1]
    Ig = volume[z1, y1, x0]
    Ih = volume[z1, y1, x1]

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
    return Ia * wa + Ib * wb + Ic * wc + Id * wd + Ie * we + If * wf + Ig * wg + Ih * wh


class GracefulShutdown:
    """A context manager for graceful shutdowns."""

    stop = False

    def __enter__(self):
        """Register the signal handler."""

        def handle_signal(signum, frame):
            self.stop = True

        signal.signal(signal.SIGINT, handle_signal)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Unregister the signal handler."""
        pass


def generate_array_flow_field(
    flow_f, grid_shape: tuple[int, int] = (128, 128)
) -> jnp.ndarray:
    """Generate a array flow field from a flow field function.

    Args:
        flow_f:
            The flow field function.
        grid_shape: tuple[int, int]
            The shape of the grid.

    Returns:
        arr: jnp.ndarray
            The array flow field.
    """
    # Get the image shape
    H, W = grid_shape
    # Create pixel coordinate grids: y in [0..H-1], x in [0..W-1]
    rows = jnp.arange(H)
    cols = jnp.arange(W)

    # vmap over both axes, and apply the flow function at time t=1
    arr = jax.vmap(lambda i: jax.vmap(lambda j: jnp.array(flow_f(1, i, j)))(cols))(rows)

    return arr


def missing_speeds_panel(config_path) -> tuple[float, float, float, float]:
    """Check for missing speeds in the configuration file.

    Args:
        config_path: str
            The path to the configuration file.

    Returns:
        speeds: tuple[float, float, float, float]
            The maximum and minimum speeds in the x and y directions.
    """
    # Load the configuration file
    config = load_configuration(config_path)

    missing_speeds = []
    for key in ["max_speed_x", "max_speed_y", "min_speed_x", "min_speed_y"]:
        if key not in config or not isinstance(config[key], (int, float)):
            missing_speeds.append(key)

    if missing_speeds:
        print(
            "[WARNING]: The following speed values are missing or invalid in the "
            f"configuration file: {', '.join(missing_speeds)}"
        )
        choice = input(
            "Would you like to "
            "(1) run a script to calculate them (it might take some time) or"
            " (2) stop and add them manually? Enter 1 or 2: "
        )

        if choice == "1":
            calculated_speeds = calculate_min_and_max_speeds(config["scheduler_files"])
            config.update(calculated_speeds)
            update_config_file(config_path, calculated_speeds)
            print("Calculated values:")
            for key, value in calculated_speeds.items():
                print(f"{key}: {value}")
            advance = input(
                "Do you want to continue with the updated configuration? (y/n): "
            )
            if advance.lower() == "y":
                speeds = (
                    calculated_speeds["max_speed_x"],
                    calculated_speeds["max_speed_y"],
                    calculated_speeds["min_speed_x"],
                    calculated_speeds["min_speed_y"],
                )

                return speeds
            else:
                raise RuntimeError("Exiting the script.")
        elif choice == "2":
            print(
                "Please add the missing values to the configuration file"
                " and re-run the script."
            )
            raise RuntimeError("Exiting the script.")
        else:
            print("[WARNING]: Invalid choice. Exiting.")
            raise RuntimeError("Exiting the script.")
    else:
        logger.info("All required speed values are present in the configuration file.")
        return (
            config["max_speed_x"],
            config["max_speed_y"],
            config["min_speed_x"],
            config["min_speed_y"],
        )


def calculate_min_and_max_speeds(file_list: list[str]) -> dict[str, float]:
    """Calculate the missing speeds for a list of files.

    Args:
        file_list: list[str]
            The list of files.

    Returns:
        dict[str, float]: A dictionary containing the minimum and maximum speeds
            in the x and y directions with keys:
            - "min_speed_x"
            - "max_speed_x"
            - "min_speed_y"
            - "max_speed_y"
    """
    # Input validation
    if not file_list:
        raise ValueError("The file_list must not be empty.")

    for file_path in file_list:
        if not isinstance(file_path, str):
            raise ValueError("All file paths must be strings.")
        if not os.path.isfile(file_path):
            raise ValueError(f"File {file_path} does not exist.")
        if not file_path.endswith(".h5"):
            raise ValueError(f"File {file_path} is not a .h5 file.")

    running_max_speed_x = float("-inf")
    running_max_speed_y = float("-inf")
    running_min_speed_x = float("inf")
    running_min_speed_y = float("inf")
    # Wrap the file list with tqdm for a loading bar
    for file in tqdm(file_list, desc="Processing files"):
        with h5py.File(file, "r") as f:
            # Read the file
            dataset_name = list(f.keys())[0]
            data = f[dataset_name][:]

            # TODO: set to 1, left to 2 for test
            # Find the min and max speeds along each axis
            running_max_speed_x = max(running_max_speed_x, np.max(data[:, :, :, 0]))
            running_max_speed_y = max(running_max_speed_y, np.max(data[:, :, :, 2]))
            running_min_speed_x = min(running_min_speed_x, np.min(data[:, :, :, 0]))
            running_min_speed_y = min(running_min_speed_y, np.min(data[:, :, :, 2]))

    return {
        "min_speed_x": running_min_speed_x,
        "max_speed_x": running_max_speed_x,
        "min_speed_y": running_min_speed_y,
        "max_speed_y": running_max_speed_y,
    }


def update_config_file(config_path: str, updated_values: dict):
    """Update the YAML configuration file with new values."""
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    # Convert to OrderedDict to preserve order
    config_data = collections.OrderedDict(config_data)

    # Convert all values in config_data to standard Python types
    def convert_to_standard_type(value):
        if isinstance(value, (np.floating)):
            return float(value)
        elif isinstance(value, (np.integer, jnp.integer)):
            return int(value)
        elif isinstance(value, (np.ndarray, jnp.ndarray)):
            return value.tolist()
        return value

    # Convert all values in config_data to standard Python types
    config_data = collections.OrderedDict(
        {key: convert_to_standard_type(value) for key, value in config_data.items()}
    )

    # Add new keys at the end
    for key, value in updated_values.items():
        config_data[key] = convert_to_standard_type(value)

    # Handle scheduler_files separately
    scheduler_files = config_data.pop("scheduler_files", [])

    with open(config_path, "w") as file:
        for key, value in config_data.items():
            file.write(f"{key}: {value}\n")
        if scheduler_files:
            file.write("scheduler_files:\n")
            for item in scheduler_files:
                file.write(f"  - {item}\n")


def visualize_and_save(batch, output_dir="output_images", num_images_to_display=5):
    """Visualizes and saves a specified number of images from a batch.

    Args:
        batch: Tuple containing (images1, images2, flow_field).
        output_dir: Directory to save the images.
        num_images_to_display: Number of images to display and save from each batch.
    """
    os.makedirs(output_dir, exist_ok=True)

    images1, images2, flow_field = batch

    for i in range(min(num_images_to_display, len(images1))):
        # Extract flow field
        flow_x = flow_field[..., 0]
        flow_y = flow_field[..., 1]
        # Create a grid for the quiver plot
        y, x = np.mgrid[0 : flow_x.shape[0], 0 : flow_x.shape[1]]

        # Save individual images and flow field
        plt.imsave(
            os.path.join(output_dir, f"batch_{i}_image1.png"), images1[i], cmap="gray"
        )
        plt.imsave(
            os.path.join(output_dir, f"batch_{i}_image2.png"), images2[i], cmap="gray"
        )

        # Save the quiver plot as a separate image
        quiver_fig, quiver_ax = plt.subplots(figsize=(7, 7))
        step = 1
        quiver_ax.quiver(
            x[::step, ::step],
            y[::step, ::step],
            flow_x[::step, ::step],
            flow_y[::step, ::step],
            pivot="mid",
            color="blue",
        )
        quiver_ax.set_aspect("equal")
        quiver_fig.savefig(os.path.join(output_dir, f"batch_{i}_quiver.png"))
        plt.close(quiver_fig)

    logger.info(
        f"Saved {min(num_images_to_display, len(images1))} images to {output_dir}"
    )


def flow_field_adapter(
    flow_field: jnp.ndarray, new_flow_field_shape: Tuple[int, int] = (256, 256)
):
    """Adapter to convert flow field to one with a different resolution.

    Args:
        flow_field: jnp.ndarray
            The original flow field to be adapted.
        new_flow_field_shape: Tuple[int, int]
            The desired shape of the new flow field.

    Returns:
        jnp.ndarray: The adapted flow field with the new shape.
    """
    original_shape = flow_field.shape[:2]

    # Create a 2D grid of coordinates for the new shape
    x = jnp.linspace(0, original_shape[1] - 1, new_flow_field_shape[1])
    y = jnp.linspace(0, original_shape[0] - 1, new_flow_field_shape[0])
    x_new, y_new = jnp.meshgrid(x, y)

    # Vectorize over the columns
    interp_over_cols = jax.vmap(
        lambda x_coord, y_coord: bilinear_interpolate(
            flow_field[..., 0],
            x_coord,
            y_coord,
        ),
        in_axes=(0, 0),
    )

    # Now vectorize over the rows
    new_flow_field_x = jax.vmap(
        lambda xs, ys: interp_over_cols(xs, ys), in_axes=(0, 0)
    )(x_new, y_new)

    # Repeat for the second channel
    interp_over_cols_y = jax.vmap(
        lambda x_coord, y_coord: bilinear_interpolate(
            flow_field[..., 1],
            x_coord,
            y_coord,
        ),
        in_axes=(0, 0),
    )
    new_flow_field_y = jax.vmap(
        lambda xs, ys: interp_over_cols_y(xs, ys), in_axes=(0, 0)
    )(x_new, y_new)

    # Stack the two interpolated channels along the last dimension
    new_flow_field = jnp.stack([new_flow_field_x, new_flow_field_y], axis=-1)
    return new_flow_field


def input_check_flow_field_adapter(
    flow_field: jnp.ndarray, new_flow_field_shape: Tuple[int, int] = (256, 256)
):
    """Checks the input arguments of the flow field adapter function.

    Args:
        flow_field: jnp.ndarray
            The original flow field to be adapted.
        new_flow_field_shape: Tuple[int, int]
            The desired shape of the new flow field.
    """
    if (
        not isinstance(flow_field, jnp.ndarray)
        or len(flow_field.shape) != 3
        or flow_field.shape[2] != 2
    ):
        raise ValueError("Flow_field must be a 3D jnp.ndarray with shape (H, W, 2).")
    if (
        not isinstance(new_flow_field_shape, tuple)
        or len(new_flow_field_shape) != 2
        or not all(isinstance(s, int) and s > 0 for s in new_flow_field_shape)
    ):
        raise ValueError(
            "new_flow_field_shape must be a tuple of two positive integers."
        )
