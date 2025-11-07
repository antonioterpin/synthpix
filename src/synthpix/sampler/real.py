"""Sampler for real data."""

import jax.numpy as jnp
from goggles import get_logger

from synthpix.utils import SYNTHPIX_SCOPE
from synthpix.types import SynthpixBatch
from synthpix.sampler.base import Sampler
from synthpix.scheduler import SchedulerProtocol

logger = get_logger(__name__, scope=SYNTHPIX_SCOPE)


class RealImageSampler(Sampler):
    """Sampler for real data."""

    def __init__(self, scheduler: SchedulerProtocol, batch_size: int = 1):
        """Initialize the sampler.

        Args:
            scheduler: Scheduler instance that provides real images.
            batch_size: Number of images to sample in each batch.
        """
        super().__init__(scheduler, batch_size)

        while (
            not hasattr(scheduler, "include_images") 
            or not scheduler.include_images # pyright: ignore[reportAttributeAccessIssue]
        ):
            if hasattr(scheduler, "scheduler"):
                scheduler = scheduler.scheduler # pyright: ignore[reportAttributeAccessIssue]
            else:
                raise ValueError(
                    "Base scheduler must have include_images set to True"
                    " to use RealImageSampler."
                )

        logger.info("RealImageSampler initialized successfully")

    def _get_next(self) -> SynthpixBatch:
        # Get the next batch of flow fields from the scheduler
        batch = self.scheduler.get_batch(batch_size=self.batch_size)
        batch = SynthpixBatch(
            images1=jnp.array(batch[0], dtype=jnp.float32),
            images2=jnp.array(batch[1], dtype=jnp.float32),
            flow_fields=jnp.array(batch[2], dtype=jnp.float32),
            params=None,
            done=None,  # Done is handled in the Episodic wrapper if needed
        )

        return batch
