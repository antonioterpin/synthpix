"""Sampler for real data."""

import jax.numpy as jnp

from synthpix.utils import logger


class RealImageSampler:
    """Sampler for real data."""

    def __init__(self, scheduler, batch_size=1):
        """Initialize the sampler.

        Args:
            scheduler: Scheduler instance that provides real images.
            batch_size (int): Number of images to sample in each batch.
        """
        self.scheduler = scheduler
        while not hasattr(scheduler, "include_images") or not scheduler.include_images:
            if hasattr(scheduler, "scheduler"):
                scheduler = scheduler.scheduler
            else:
                raise ValueError(
                    "Base scheduler must have include_images set to True"
                    " to use RealImageSampler."
                )
        self.batch_size = batch_size

        logger.info("RealImageSampler initialized successfully")

        logger.debug(f"Scheduler class: {self.scheduler.__class__.__name__}")
        while hasattr(scheduler, "scheduler"):
            scheduler = scheduler.scheduler
            logger.debug(f"Scheduler class: {scheduler.scheduler.__class__.__name__}")

    def __iter__(self):
        """Return an iterator for the sampler."""
        return self

    def __next__(self):
        """Return the next batch of real images."""
        batch = self.scheduler.get_batch(batch_size=self.batch_size)
        batch = (
            jnp.array(batch[0], dtype=jnp.float32),
            jnp.array(batch[1], dtype=jnp.float32),
            jnp.array(batch[2], dtype=jnp.float32),
            jnp.zeros(batch[0].shape[0], dtype=jnp.float32),
        )
        return batch

    def shutdown(self):
        """Shutdown the sampler."""
        logger.info("Shutting down RealImageSampler.")
        if hasattr(self.scheduler, "shutdown"):
            self.scheduler.shutdown()
        else:
            logger.warning(
                "The underlying scheduler does not have a shutdown method."
                " Skipping shutdown."
            )
        logger.info("RealImageSampler shutdown complete.")
