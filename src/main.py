"""Main file to run the SyntheticImageSampler pipeline."""
import argparse
import os

import matplotlib.pyplot as plt

from src.sym.data_generate import generate_images_from_flow
from src.sym.image_sampler import SyntheticImageSampler
from src.sym.scheduler import HDF5FlowFieldScheduler
from src.utils import logger


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
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

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


def main(args):
    """Main function to run the SyntheticImageSampler pipeline."""
    # Apply predefined configuration if specified
    if args.config:
        config = PREDEFINED_CONFIGS[args.config]
        for key, value in config.items():
            setattr(args, key, value)

    # Initialize the scheduler
    scheduler = HDF5FlowFieldScheduler(args.scheduler_files, loop=False)

    # Initialize the sampler
    sampler = SyntheticImageSampler(
        scheduler=scheduler,
        img_gen_fn=generate_images_from_flow,
        images_per_field=args.images_per_field,
        batch_size=args.batch_size,
        image_shape=tuple(args.image_shape),
        position_bounds=tuple(args.position_bounds),
        num_particles=args.num_particles,
        p_hide_img1=args.p_hide_img1,
        p_hide_img2=args.p_hide_img2,
        diameter_range=tuple(args.diameter_range),
        intensity_range=tuple(args.intensity_range),
        rho_range=tuple(args.rho_range),
        dt=args.dt,
        seed=args.seed,
    )

    # Run the sampler and print results
    logger.info("Starting the SyntheticImageSampler pipeline...")
    for i, batch in enumerate(sampler):
        logger.info(f"Batch {i + 1} generated.")
        logger.info(f"Image batch 1 shape: {batch[0].shape}")
        logger.info(f"Image batch 2 shape: {batch[1].shape}")
        logger.info(f"Flow field shape: {batch[2].shape}")

        if args.visualize:
            visualize_and_save(batch, args.output_dir, args.num_images_to_display)

        if i >= args.images_per_field // args.batch_size - 1:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo for SyntheticImageSampler pipeline."
    )

    # Add arguments for SyntheticImageSampler parameters
    parser.add_argument(
        "--scheduler_files",
        nargs="+",
        required=True,
        help="List of HDF5 files for the scheduler.",
    )
    parser.add_argument(
        "--images_per_field",
        type=int,
        default=1000,
        help="Number of synthetic images to generate per flow field.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=250,
        help="Number of synthetic images per batch.",
    )
    parser.add_argument(
        "--image_shape",
        type=int,
        nargs=2,
        default=(1216, 1936),
        help="Shape of the synthetic images.",
    )
    parser.add_argument(
        "--position_bounds",
        type=int,
        nargs=2,
        default=(1536, 2048),
        help="Shape of the big image from which the flow field is sampled.",
    )
    parser.add_argument(
        "--num_particles",
        type=int,
        default=40000,
        help="Number of particles to simulate.",
    )
    parser.add_argument(
        "--p_hide_img1",
        type=float,
        default=0.01,
        help="Probability of hiding particles in the first image.",
    )
    parser.add_argument(
        "--p_hide_img2",
        type=float,
        default=0.01,
        help="Probability of hiding particles in the second image.",
    )
    parser.add_argument(
        "--diameter_range",
        type=float,
        nargs=2,
        default=(2, 4),
        help="Range of diameters for particles.",
    )
    parser.add_argument(
        "--intensity_range",
        type=float,
        nargs=2,
        default=(50, 200),
        help="Range of intensities for particles.",
    )
    parser.add_argument(
        "--rho_range",
        type=float,
        nargs=2,
        default=(-0.99, 0.99),
        help="Range of correlation coefficients for particles.",
    )
    parser.add_argument(
        "--dt", type=float, default=1.0, help="Time step for the simulation."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for JAX PRNG.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization of generated images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_images",
        help="Directory to save visualized images.",
    )
    parser.add_argument(
        "--num_images_to_display",
        type=int,
        default=5,
        help="Number of images to display and save per batch.",
    )

    # Add a predefined configuration for JHTDB
    PREDEFINED_CONFIGS = {
        "JHTDB": {
            "scheduler_files": ["/shared/fluids/channel_full_ts_0000.h5"],
            "images_per_field": 1000,
            "batch_size": 250,
            "image_shape": (1216, 1936),
            "position_bounds": (1536, 2048),
            "num_particles": 40000,
            "p_hide_img1": 0.01,
            "p_hide_img2": 0.01,
            "diameter_range": (0.1, 1.0),
            "intensity_range": (50, 200),
            "rho_range": (-0.99, 0.99),
            "dt": 1.0,
            "seed": 0,
        }
    }

    # Add argument for predefined configuration
    parser.add_argument(
        "--config",
        type=str,
        choices=PREDEFINED_CONFIGS.keys(),
        help="Use a predefined configuration.",
    )

    args = parser.parse_args()

    main(args)
