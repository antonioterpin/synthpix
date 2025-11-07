"""Script to check the sanity of the configuration file."""

import argparse
import collections
import os
import sys

import h5py
import jax.numpy as jnp
import numpy as np
from goggles import get_logger
from tqdm import tqdm

from .sampler import SyntheticImageSampler
from .scheduler import HDF5FlowFieldScheduler
from .utils import load_configuration, SYNTHPIX_SCOPE

logger = get_logger(__name__, scope=SYNTHPIX_SCOPE)


def update_config_file(config_path: str, updated_values: dict) -> None:
    """Update the YAML configuration file with new values.

    Args:
        config_path: The path to the configuration file.
        updated_values: A dictionary of values to update
            in the configuration file.
    """
    config_data = load_configuration(config_path)

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
        if scheduler_files and isinstance(scheduler_files, list):
            file.write("scheduler_files:\n")
            for item in scheduler_files:
                file.write(f"  - {str(item)}\n")


def calculate_min_and_max_speeds(file_list: list[str]) -> dict[str, float]:
    """Calculate the missing speeds for a list of files.

    Args:
        file_list: The list of files.

    Returns:
        A dictionary containing the minimum and maximum speeds in the x and y
            directions with keys:
                - "min_speed_x"
                - "max_speed_x"
                - "min_speed_y"
                - "max_speed_y"
    """
    running_max_speed_x = float("-inf")
    running_max_speed_y = float("-inf")
    running_min_speed_x = float("inf")
    running_min_speed_y = float("inf")
    # Wrap the file list with tqdm for a loading bar
    for file in tqdm(file_list, desc="Processing files"):
        with h5py.File(file, "r") as f:
            # Read the file
            dataset_name = list(f.keys())[0]
            dataset = f[dataset_name]

            # Ensure we have a dataset, not a group or datatype
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(
                    f"Expected dataset, got {type(dataset)} in file {file}"
                )

            # Convert to numpy array if needed
            data = np.array(dataset)

            # Check data shape and handle different formats
            if data.ndim == 4:
                # Standard 4D format: (batch, height, width, channels)
                x_data = data[..., 0]  # All x components
                y_data = data[..., 1]  # All y components
            elif data.ndim == 3 and data.shape[-1] == 2:
                # 3D format: (height, width, channels)
                x_data = data[..., 0]
                y_data = data[..., 1]
            else:
                raise ValueError(f"Unexpected data shape: {data.shape} in file {file}")

            # Find the min and max speeds along each axis
            running_max_speed_x = max(running_max_speed_x, float(np.max(x_data)))
            running_max_speed_y = max(running_max_speed_y, float(np.max(y_data)))
            running_min_speed_x = min(running_min_speed_x, float(np.min(x_data)))
            running_min_speed_y = min(running_min_speed_y, float(np.min(y_data)))

    return {
        "min_speed_x": running_min_speed_x,
        "max_speed_x": running_max_speed_x,
        "min_speed_y": running_min_speed_y,
        "max_speed_y": running_max_speed_y,
    }


def missing_speeds_panel(config_path: str) -> tuple[float, float, float, float]:
    """Check for missing speeds in the configuration file.

    Args:
        config_path: The path to the configuration file.

    Returns:
        speeds: The maximum and minimum speeds in the x and y directions.
    """
    config = load_configuration(config_path)

    # Input validation
    if "scheduler_files" not in config:
        raise ValueError("The configuration must contain 'scheduler_files'.")
    file_list = config["scheduler_files"]
    if not isinstance(file_list, list) or not file_list:
        raise ValueError("The file_list must not be empty.")

    for file_path in file_list:
        if not isinstance(file_path, str):
            raise ValueError("All file paths must be strings.")
        if not os.path.isfile(file_path):
            raise ValueError(f"File {file_path} does not exist.")
        if not file_path.endswith(".h5"):
            raise ValueError(f"File {file_path} is not a .h5 file.")

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
            calculated_speeds = calculate_min_and_max_speeds(file_list)
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
                    float(calculated_speeds["max_speed_x"]),
                    float(calculated_speeds["max_speed_y"]),
                    float(calculated_speeds["min_speed_x"]),
                    float(calculated_speeds["min_speed_y"]),
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


def main() -> None:  # pragma: no cover
    """Main function to check the sanity of the configuration file."""
    parser = argparse.ArgumentParser(
        description="Check the sanity of the configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )

    args = parser.parse_args()
    config_path = args.config

    if not os.path.exists(config_path):
        print(f"Configuration file does not exist: {config_path}")
        sys.exit(1)

    config = load_configuration(config_path)

    # 1. Check min/max speeds and offer to fix
    missing_speeds_panel(config_path)

    # 2. Try to instantiate the scheduler with the config
    scheduler = None
    try:
        scheduler = HDF5FlowFieldScheduler(config["scheduler_files"], loop=False)
    except Exception as e:
        logger.error(f"Error instantiating scheduler: {e}")
        logger.error(f"Please check the configuration file: {config_path}.")
        sys.exit(1)
    logger.info("Scheduler instantiated successfully.")

    # 3. Try to instantiate the sampler with the config
    try:
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            config=config,
        )
    except Exception as e:
        logger.error(f"Error instantiating sampler: {e}")
        logger.error(f"Please check the configuration file: {config_path}.")
        sys.exit(1)

    logger.info(f"Configuration file is valid: {config_path}")


if __name__ == "__main__":
    main()  # pragma: no cover
