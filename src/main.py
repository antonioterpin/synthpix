"""Main file to run the SyntheticImageSampler pipeline."""
import argparse

from synthpix.data_generate import generate_images_from_flow
from synthpix.image_sampler import SyntheticImageSampler
from synthpix.scheduler import HDF5FlowFieldScheduler
from synthpix.utils import load_configuration, logger, visualize_and_save


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
    sampler = SyntheticImageSampler.from_config(
        scheduler=scheduler,
        img_gen_fn=generate_images_from_flow,
        config=config,
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

    main(
        config_path=args.config,
        output_dir=args.output_dir,
        num_images_to_display=args.visualize,
    )
