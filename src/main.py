"""Main file to run the SyntheticImageSampler pipeline."""
import argparse
import os

import matplotlib.pyplot as plt

from src.sym.data_generate import generate_images_from_flow
from src.sym.image_sampler import SyntheticImageSampler
from src.sym.scheduler import HDF5FlowFieldScheduler
from src.utils import load_configuration, logger


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

        axes[2].imshow(flow_field[..., 0], cmap="viridis")
        axes[2].set_title("Flow Field")

        plt.savefig(os.path.join(output_dir, f"batch_image_{i}.png"))

    logger.info(
        f"Saved {min(num_images_to_display, len(images1))} images to {output_dir}"
    )


def main(
    scheduler_files,
    images_per_field,
    batch_size,
    image_shape,
    position_bounds,
    num_particles,
    p_hide_img1,
    p_hide_img2,
    diameter_range,
    intensity_range,
    rho_range,
    dt,
    seed,
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
        position_bounds (tuple):
            Shape of the big image from which the flow field is sampled.
        num_particles (int): Number of particles to simulate.
        p_hide_img1 (float): Probability of hiding particles in the first image.
        p_hide_img2 (float): Probability of hiding particles in the second image.
        diameter_range (tuple): Range of diameters for particles.
        intensity_range (tuple): Range of intensities for particles.
        rho_range (tuple): Range of correlation coefficients for particles.
        dt (float): Time step for the simulation.
        seed (int): Random seed for JAX PRNG.
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
        image_shape=tuple(image_shape),
        position_bounds=tuple(position_bounds),
        num_particles=num_particles,
        p_hide_img1=p_hide_img1,
        p_hide_img2=p_hide_img2,
        diameter_range=tuple(diameter_range),
        intensity_range=tuple(intensity_range),
        rho_range=tuple(rho_range),
        dt=dt,
        seed=seed,
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

    main(
        scheduler_files=config["scheduler_files"],
        images_per_field=config["images_per_field"],
        batch_size=config["batch_size"],
        image_shape=tuple(config["image_shape"]),
        position_bounds=tuple(config["position_bounds"]),
        num_particles=config["num_particles"],
        p_hide_img1=config["p_hide_img1"],
        p_hide_img2=config["p_hide_img2"],
        diameter_range=tuple(config["diameter_range"]),
        intensity_range=tuple(config["intensity_range"]),
        rho_range=tuple(config["rho_range"]),
        dt=config["dt"],
        seed=config["seed"],
        visualize=args.visualize,
        output_dir=args.output_dir,
        num_images_to_display=args.num_images_to_display,
    )
