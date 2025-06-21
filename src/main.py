"""Main file to run the SyntheticImageSampler pipeline."""
import argparse

import synthpix
from synthpix.utils import logger, visualize_and_save


def main(config_path, output_dir, num_images_to_display):
    """Main function to run the SyntheticImageSampler pipeline.

    Args:
        config_path (string): Configuration file path.
        output_dir (str): Directory to save visualized images.
        num_images_to_display (int): Number of images to display and save per batch.
    """
    # Initialize the sampler
    sampler = synthpix.make(config_path, buffer_size=10, images_from_file=True)

    try:
        # Run the sampler and print results
        logger.info(f"Starting the {sampler.__class__.__name__} pipeline...")
        for i, batch in enumerate(sampler):
            # logger.info(f"Batch {i + 1} generated.")
            # logger.info(f"Image 1 batch shape: {batch['images1'].shape}")
            # logger.info(f"Image 2 batch shape: {batch['images2'].shape}")
            # logger.info(f"Flow field batch shape: {batch['flow_fields'].shape}")

            for j in range(min(num_images_to_display, batch["images1"].shape[0])):
                visualize_and_save(
                    f"batch_{i}_sample_{j}",
                    batch["images1"][j],
                    batch["images2"][j],
                    batch["flow_fields"][j],
                    output_dir,
                )

            if num_images_to_display > 0:
                # Ask user if they want to continue generating images
                choice = input("Do you want to continue generating images? (y/n): ")
                if choice.lower() != "y":
                    logger.info("Stopping the pipeline.")
                    break
    finally:
        sampler.shutdown()


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

    main(
        config_path=args.config,
        output_dir=args.output_dir,
        num_images_to_display=args.visualize,
    )
