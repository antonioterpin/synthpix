"""Sampler for real data."""

import jax.numpy as jnp

from ..utils import logger
from .base import Sampler


class RealImageSampler(Sampler):
    """Sampler for real data."""

    @classmethod
    def from_config(cls, scheduler, batch_size=1):
        """Create a RealImageSampler instance from a configuration.

        Args:
            scheduler: Scheduler instance that provides real images.
            batch_size (int): Number of images to sample in each batch.

        Returns:
            RealImageSampler: An instance of the sampler.
        """
        return cls(scheduler, batch_size)

    def __init__(self, scheduler, batch_size=1):
        """Initialize the sampler.

        Args:
            scheduler: Scheduler instance that provides real images.
            batch_size (int): Number of images to sample in each batch.
        """
        super().__init__(scheduler, batch_size)

        while not hasattr(scheduler, "include_images") or not scheduler.include_images:
            if hasattr(scheduler, "scheduler"):
                scheduler = scheduler.scheduler
            else:
                raise ValueError(
                    "Base scheduler must have include_images set to True"
                    " to use RealImageSampler."
                )

        logger.info("RealImageSampler initialized successfully")

    def ___next__(self):
        """Return the next batch of real images."""
        # Get the next batch of flow fields from the scheduler
        batch = self.scheduler.get_batch(batch_size=self.batch_size)
        batch = {
            "images1": jnp.array(batch[0], dtype=jnp.float32),
            "images2": jnp.array(batch[1], dtype=jnp.float32),
            "flow_fields": jnp.array(batch[2], dtype=jnp.float32),
            "params": None,
        }

        return batch
