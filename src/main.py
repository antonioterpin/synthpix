"""Main file to run the SyntheticImageSampler pipeline."""

import argparse
import logging
import os

import goggles as gg
import matplotlib.pyplot as plt
import numpy as np

import synthpix

# To see logs in the console, we need to attach a handler to
# the Synthpix scope: synthpix.SYNTHPIX_SCOPE
logger = gg.get_logger(__name__, scope=synthpix.SYNTHPIX_SCOPE)
gg.attach(
    gg.ConsoleHandler(level=logging.INFO),
)


def visualize_and_save(
    name: str,
    image1: np.ndarray,
    image2: np.ndarray,
    flow_field: np.ndarray,
    output_dir: str = "output_images",
) -> None:
    """Visualizes and saves a specified number of images from a batch.

    Args:
        name: The name of the batch.
        image1: The first image to visualize.
        image2: The second image to visualize.
        flow_field: The flow field to visualize.
        output_dir: Directory to save the images.
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
        config_path: Configuration file path.
        output_dir: Directory to save visualized images.
        num_images_to_display: Number of images to display and save per batch.
    """
    # Initialize the sampler
    sampler = synthpix.make(config_path)

    try:
        # Run the sampler and print results
        logger.info(f"Starting the {sampler.__class__.__name__} pipeline...")
        for i, batch in enumerate(sampler):
            # Batch is of type SynthpixBatch and we can access its fields
            for j in range(min(num_images_to_display, batch.images1.shape[0])):
                # Visualize and save the images
                # We visualize the images in ij coordinates, so
                # batch.flow_fields[j, 0, 0] is the flow on the
                # top left pixel of the j-th element of the batch
                visualize_and_save(
                    f"batch_{i}_sample_{j}",
                    np.asarray(batch.images1[j]),
                    np.asarray(batch.images2[j]),
                    np.asarray(batch.flow_fields[j]),
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

    gg.finish()
