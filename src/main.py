"""Main file to run the SyntheticImageSampler pipeline."""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import synthpix
from synthpix.utils import logger


def visualize_and_save(name, image1, image2, flow_field, output_dir="output_images"):
    """Visualizes and saves a specified number of images from a batch.

    Args:
        name (str): The name of the batch.
        image1 (jnp.ndarray): The first image to visualize.
        image2 (jnp.ndarray): The second image to visualize.
        flow_field (jnp.ndarray): The flow field to visualize.
        output_dir (str): Directory to save the images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract flow field
    flow_x = flow_field[..., 0]
    flow_y = flow_field[..., 1]
    # Create a grid for the quiver plot
    y, x = np.mgrid[0 : flow_x.shape[0], 0 : flow_x.shape[1]]

    # Save individual images and flow field
    plt.imsave(os.path.join(output_dir, f"{name}_image1.png"), image1, cmap="gray")
    plt.imsave(os.path.join(output_dir, f"{name}_image2.png"), image2, cmap="gray")

    # Save the quiver plot as a separate image
    quiver_fig, quiver_ax = plt.subplots(figsize=(7, 7))
    step = 32
    quiver_ax.quiver(
        x[::step, ::step],
        y[::step, ::step],
        flow_x[::step, ::step],
        flow_y[::step, ::step],
        pivot="mid",
        color="blue",
    )
    quiver_ax.set_aspect("equal")
    quiver_fig.savefig(os.path.join(output_dir, f"{name}_quiver.png"))
    plt.close(quiver_fig)

    logger.info(f"Saved images for {name} to {output_dir}.")


def main(config_path: str, output_dir: str, num_images_to_display: int):
    """Main function to run the SyntheticImageSampler pipeline.

    Args:
        config_path (string): Configuration file path.
        output_dir (str): Directory to save visualized images.
        num_images_to_display (int): Number of images to display and save per batch.
    """
    # Initialize the sampler
    sampler = synthpix.make(config_path, buffer_size=10, images_from_file=False)

    try:
        # Run the sampler and print results
        logger.info(f"Starting the {sampler.__class__.__name__} pipeline...")
        for i, batch in enumerate(sampler):
            # logger.info(f"Batch {i + 1} generated.")
            # logger.info(f"Image 1 batch shape: {batch['images1'].shape}")
            # logger.info(f"Image 2 batch shape: {batch['images2'].shape}")
            # logger.info(f"Flow field batch shape: {batch['flow_fields'].shape}")

            for j in range(min(num_images_to_display, batch["images1"].shape[0])):
                # Visualize and save the images
                # We visualize the images in ij coordinates, so
                # batch["flow_fields"][j, 0, 0] is the flow on the top left pixel
                # of the j-th element of the batch
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
        default="config/main.yaml",
        help="Configuration file path.",
    )

    parser.add_argument(
        "--devices",
        type=str,
        default="cpu",
        help="Devices to use for computation (e.g., 'cpu', '0,1' for GPUs).",
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
