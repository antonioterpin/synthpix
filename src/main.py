"""Main file to run the SyntheticImageSampler pipeline."""
import argparse
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

from src.sym.data_generate import generate_images_from_flow
from src.sym.image_sampler import SyntheticImageSampler
from src.sym.scheduler import HDF5FlowFieldScheduler
from src.utils import (
    calculate_min_and_max_speeds,
    load_configuration,
    logger,
    update_config_file,
)


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
        _, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(images1[i], cmap="gray")
        axes[0].set_title("Image 1")

        axes[1].imshow(images2[i], cmap="gray")
        axes[1].set_title("Image 2")

        # Quiver plot for flow visualization with downsampling
        flow_x = flow_field[..., 0]
        flow_y = flow_field[..., 1]

        # Downsample the flow field for better visualization
        # Adjust this factor as needed
        downsample_factor = 10
        flow_x_downsampled = flow_x[::downsample_factor, ::downsample_factor]
        flow_y_downsampled = flow_y[::downsample_factor, ::downsample_factor]

        y, x = np.mgrid[0 : flow_x.shape[0], 0 : flow_x.shape[1]]
        y_downsampled = y[::downsample_factor, ::downsample_factor]
        x_downsampled = x[::downsample_factor, ::downsample_factor]
        axes[2].quiver(
            x_downsampled,
            y_downsampled,
            flow_x_downsampled,
            flow_y_downsampled,
            scale=1000,  # TODO: adjust scale for better visualization
            scale_units="xy",
            color="blue",
        )
        axes[2].set_title("Flow Field (Quiver)")

        plt.savefig(os.path.join(output_dir, f"batch_image_{i}.png"))

    logger.info(
        f"Saved {min(num_images_to_display, len(images1))} images to {output_dir}"
    )


def main(
    scheduler_files,
    images_per_field,
    batch_size,
    flow_field_shape,
    flow_field_size,
    image_shape,
    resolution,
    img_offset,
    num_particles,
    p_hide_img1,
    p_hide_img2,
    diameter_range,
    intensity_range,
    rho_range,
    dt,
    seed,
    max_speed_x,
    max_speed_y,
    min_speed_x,
    min_speed_y,
    visualize,
    output_dir,
    num_images_to_display,
):
    """Main function to run the SyntheticImageSampler pipeline.

    Args:
        scheduler_files (list): List of HDF5 files for the scheduler.
        images_per_field (int): Number of synthetic images to generate per flow field.
        batch_size (int): Number of synthetic images per batch.
        image_shape (tuple): Shape of the synthetic images.
        flow_field_shape (tuple): Shape of the flow field in grid steps.
        flow_field_size (tuple): Shape of the flow field in length measure units.
        resolution (tuple): Resolution of the images in pixels per meter.
        img_offset (tuple): Offset of the images in length measure units.
        num_particles (int): Number of particles to simulate.
        p_hide_img1 (float): Probability of hiding particles in the first image.
        p_hide_img2 (float): Probability of hiding particles in the second image.
        diameter_range (tuple): Range of diameters for particles.
        intensity_range (tuple): Range of intensities for particles.
        rho_range (tuple): Range of correlation coefficients for particles.
        dt (float): Time step for the simulation.
        seed (int): Random seed for JAX PRNG.
        max_speed_x (float): Maximum speed in the x direction.
        max_speed_y (float): Maximum speed in the y direction.
        min_speed_x (float): Minimum speed in the x direction.
        min_speed_y (float): Minimum speed in the y direction.
        visualize (bool): Enable visualization of generated images.
        output_dir (str): Directory to save visualized images.
        num_images_to_display (int): Number of images to display and save per batch.
    """
    # Initialize the scheduler
    scheduler = HDF5FlowFieldScheduler(scheduler_files, loop=False)

    # Initialize the sampler
    sampler = SyntheticImageSampler(
        scheduler=scheduler,
        img_gen_fn=generate_images_from_flow,
        images_per_field=images_per_field,
        batch_size=batch_size,
        flow_field_shape=tuple(flow_field_shape),
        flow_field_size=tuple(flow_field_size),
        image_shape=tuple(image_shape),
        resolution=resolution,
        img_offset=img_offset,
        num_particles=num_particles,
        p_hide_img1=p_hide_img1,
        p_hide_img2=p_hide_img2,
        diameter_range=tuple(diameter_range),
        intensity_range=tuple(intensity_range),
        rho_range=tuple(rho_range),
        dt=dt,
        seed=seed,
        max_speed_x=max_speed_x,
        max_speed_y=max_speed_y,
        min_speed_x=min_speed_x,
        min_speed_y=min_speed_y,
    )

    # Run the sampler and print results
    logger.info("Starting the SyntheticImageSampler pipeline...")
    for i, batch in enumerate(sampler):
        logger.info(f"Batch {i + 1} generated.")
        logger.info(f"Image batch 1 shape: {batch[0].shape}")
        logger.info(f"Image batch 2 shape: {batch[1].shape}")
        logger.info(f"Flow field shape: {batch[2].shape}")

        if visualize:
            visualize_and_save(batch, output_dir, num_images_to_display)

        if i >= images_per_field // batch_size - 1:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the SyntheticImageSampler pipeline."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/JHTDB.yaml",
        help="Configuration file.",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization of generated images.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="Directory to save visualized images.",
    )

    parser.add_argument(
        "--num_images_to_display",
        type=int,
        default=5,
        help="Number of images to display and save from each batch.",
    )

    args = parser.parse_args()

    # Read config file
    config = load_configuration(args.config)

    # Check for missing speed values
    missing_speeds = []
    for key in ["max_speed_x", "max_speed_y", "min_speed_x", "min_speed_y"]:
        if key not in config:
            missing_speeds.append(key)

    if missing_speeds:
        print(
            "[WARNING]: The following speed values are missing in the "
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
            update_config_file(args.config, calculated_speeds)
            print("Calculated values:")
            for key, value in calculated_speeds.items():
                print(f"{key}: {value}")
            advance = input(
                "Do you want to continue with the updated configuration? (y/n): "
            )
            if advance.lower() == "y":
                pass
            else:
                print("Exiting the script.")
                sys.exit(0)
        elif choice == "2":
            print(
                "Please add the missing values to the configuration file"
                " and re-run the script."
            )
            sys.exit(1)
        else:
            print("[WARNING]: Invalid choice. Exiting.")
            sys.exit(1)

    # Open the first scheduler file and get its shape
    first_file = config["scheduler_files"][0]
    with h5py.File(first_file, "r") as f:
        dataset_name = list(f.keys())[0]
        data_shape = f[dataset_name].shape
        flow_field_shape = data_shape[0], data_shape[2] // 2

    main(
        scheduler_files=config["scheduler_files"],
        images_per_field=config["images_per_field"],
        batch_size=config["batch_size"],
        flow_field_shape=flow_field_shape,
        flow_field_size=tuple(config["flow_field_size"]),
        image_shape=tuple(config["image_shape"]),
        resolution=config["resolution"],
        img_offset=tuple(config["img_offset"]),
        num_particles=config["num_particles"],
        p_hide_img1=config["p_hide_img1"],
        p_hide_img2=config["p_hide_img2"],
        diameter_range=tuple(config["diameter_range"]),
        intensity_range=tuple(config["intensity_range"]),
        rho_range=tuple(config["rho_range"]),
        dt=config["dt"],
        seed=config["seed"],
        max_speed_x=config["max_speed_x"],
        max_speed_y=config["max_speed_y"],
        min_speed_x=config["min_speed_x"],
        min_speed_y=config["min_speed_y"],
        visualize=args.visualize,
        output_dir=args.output_dir,
        num_images_to_display=args.num_images_to_display,
    )
