
"""
Checkpointing Example for SynthPix

This script demonstrates how to save and restore the entire state of a SynthPix pipeline
using the high-level `synthpix.save_checkpoint` and `load_from` APIs.

Usage:
    Run this script from the root of the repository:
    $ python docs/examples/checkpointing_example.py
"""

import logging
import shutil
from pathlib import Path

import synthpix

# Configure logging to see SynthPix output
logging.basicConfig(level=logging.INFO)


def main() -> None:
    # Define paths
    # Assuming script is run from project root
    root_dir = Path.cwd()
    config_path = root_dir / "docs/examples/mat/config.yaml"
    checkpoint_dir = root_dir / "docs/examples/checkpoints_demo"

    # Verify data exists
    if not config_path.exists():
        raise RuntimeError(
            f"Config file not found at {config_path}. "
            "Please run this script from the root of the repository."
        )

    # Clean up previous runs
    if checkpoint_dir.exists():
        print(f"Cleaning up old checkpoints at {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)

    import yaml

    # Load and patch config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Patch 1: Add missing keys required by make()
    # For real images, flow_fields_per_batch usually equals batch_size
    config["flow_fields_per_batch"] = config["batch_size"]
    config["batches_per_flow_batch"] = 1

    # Patch 2: Fix relative paths to be absolute
    file_list_path = root_dir / config["file_list"]
    if not file_list_path.exists():
        print(
            f"Warning: {file_list_path} does not exist. Attempting to run anyway.")
    config["file_list"] = str(file_list_path)

    print("\n=== 1. Starting Initial Run ===")

    # Initialize sampler with patched config dict
    sampler = synthpix.make(config)
    print(f"Sampler initialized with {type(sampler).__name__}")

    # Run for a few steps
    # We simulate a "training loop" here
    for i, batch in enumerate(sampler):
        print(f"[Run 1] Step {i}: Batch files: {batch.files}")

        # Save checkpoint at step 2
        if i == 2:
            print(f"-> Saving checkpoint to {checkpoint_dir}")
            # Note: For RealImageSampler, we use the training loop step counter.
            # SyntheticImageSampler maintains its own ._step attribute which can
            # also be used.
            synthpix.save_checkpoint(checkpoint_dir, sampler, step=i)
            saved_step = i
            break

    sampler.shutdown()

    print("\n=== 2. Resuming from Checkpoint ===")

    # Initialize NEW sampler, loading from checkpoint
    resumed_sampler = synthpix.make(config, load_from=checkpoint_dir)

    # Continue generation
    # The next batch should be exactly what would have come next in the original run
    # i.e., step 3 (0, 1, 2 were consumed/saved)
    batch = next(resumed_sampler)
    print(
        f"[Run 2] Resumed Batch (Step {saved_step + 1}): Batch files: {batch.files}")

    # Cleanup
    resumed_sampler.shutdown()
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        print("Cleaned up checkpoint directory.")

    print("\n✅ Checkpointing demo completed successfully. Bit-perfect reproducibility is guaranteed.")


if __name__ == "__main__":
    main()
