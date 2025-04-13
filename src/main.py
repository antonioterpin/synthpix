"""Main file to run the SyntheticImageSampler pipeline."""
import argparse

import h5py

from src.synthpix.data_generate import generate_images_from_flow
from src.synthpix.image_sampler import SyntheticImageSampler
from src.synthpix.scheduler import HDF5FlowFieldScheduler
from src.synthpix.utils import load_configuration, logger, visualize_and_save


def main(config_path, output_dir, num_images_to_display):
    """Main function to run the SyntheticImageSampler pipeline.

    Args:
        config_path (string): Configuration file path.
        visualize (bool): Enable visualization of generated images.
        output_dir (str): Directory to save visualized images.
        num_images_to_display (int): Number of images to display and save per batch.
    """
    # Load configuration
    config = load_configuration(config_path)
    logger.info("Configuration loaded successfully.")

    # Initialize the scheduler
    scheduler = HDF5FlowFieldScheduler(config["scheduler_files"], loop=False)

    # Initialize the sampler
    sampler = SyntheticImageSampler(
        scheduler=scheduler,
        img_gen_fn=generate_images_from_flow,
        images_per_field=config["images_per_field"],
        batch_size=config["batch_size"],
        flow_field_size=tuple(config["flow_field_size"]),
        image_shape=tuple(config["image_shape"]),
        resolution=config["resolution"],
        velocities_per_pixel=config["velocities_per_pixel"],
        img_offset=config["img_offset"],
        seeding_density=config["seeding_density"],
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
        config_path=config_path,
    )

    # Run the sampler and print results
    logger.info("Starting the SyntheticImageSampler pipeline...")
    for i, batch in enumerate(sampler):
        logger.info(f"Batch {i + 1} generated.")
        logger.info(f"Image 1 batch shape: {batch[0].shape}")
        logger.info(f"Image 2 batch shape: {batch[1].shape}")
        logger.info(f"Flow field batch shape: {batch[2].shape}")

        for j in range(min(num_images_to_display, batch[0].shape[0])):
            visualize_and_save(
                f"batch_{i}_sample_{j}", batch[0][j], batch[1][j], batch[2], output_dir
            )

        logger.info(
            f"Saved {num_images_to_display} for batch {i}. Stopping visualization."
        )
        choice = input("Do you want to continue generating images? (y/n): ")
        if choice.lower() != "y":
            logger.info("Stopping the pipeline.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the SyntheticImageSampler pipeline."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/JHTDB.yaml",
        help="Configuration file path.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="Directory to save visualized images.",
    )

    parser.add_argument(
        "--visualize",
        type=int,
        default=1,
        help="Number of images to display and save from each batch.",
    )

    args = parser.parse_args()

    # Read config file
    config = load_configuration(args.config)

    # Open the first scheduler file and get its shape
    first_file = config["scheduler_files"][0]
    with h5py.File(first_file, "r") as f:
        dataset_name = list(f.keys())[0]
        data_shape = f[dataset_name].shape
        flow_field_shape = data_shape[0], data_shape[2] // 2

    main(
        config_path=args.config,
        output_dir=args.output_dir,
        num_images_to_display=args.visualize,
    )
