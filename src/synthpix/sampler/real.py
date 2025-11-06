"""Sampler for real data."""

from typing_extensions import Self
import jax.numpy as jnp
from goggles import get_logger

from .base import Sampler
from synthpix.scheduler.base import BaseFlowFieldScheduler

logger = get_logger(__name__)


class RealImageSampler(Sampler):
    """Sampler for real data."""

    @classmethod
    def from_config(
        cls, scheduler: BaseFlowFieldScheduler, batch_size: int = 1
    ) -> Self:
        """Create a RealImageSampler instance from a configuration.

        Args:
            scheduler: Scheduler instance that provides real images.
            batch_size: Number of images to sample in each batch.

        Returns: An instance of the sampler.
        """
        return cls(scheduler, batch_size)

    def __init__(self, scheduler: BaseFlowFieldScheduler, batch_size: int = 1):
        """Initialize the sampler.

        Args:
            scheduler: Scheduler instance that provides real images.
            batch_size: Number of images to sample in each batch.
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

    def ___next__(self) -> dict[str, jnp.ndarray]:
        """Return the next batch of real images.

        Returns: A dictionary containing batches of images and flow fields.
            - images1: Batch of first images.
            - images2: Batch of second images.
            - flow_fields: Batch of flow fields.
            - params: None (placeholder for compatibility,
                since the data is not generated).
        """
        # Get the next batch of flow fields from the scheduler
        batch = self.scheduler.get_batch(batch_size=self.batch_size)
        batch = {
            "images1": jnp.array(batch[0], dtype=jnp.float32),
            "images2": jnp.array(batch[1], dtype=jnp.float32),
            "flow_fields": jnp.array(batch[2], dtype=jnp.float32),
            "params": None,
        }

        return batch
