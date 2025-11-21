"""Sampler for real data."""

import jax.numpy as jnp
from goggles import get_logger
from typing_extensions import Self

from synthpix.scheduler.episodic import EpisodicFlowFieldScheduler
from synthpix.scheduler.prefetch import PrefetchingFlowFieldScheduler
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

        while not getattr(scheduler, "include_images", False):
            if isinstance(
                scheduler, (EpisodicFlowFieldScheduler, PrefetchingFlowFieldScheduler)
            ):
                scheduler = scheduler.scheduler
            else:
                raise ValueError(
                    "Base scheduler must have include_images set to True"
                    " to use RealImageSampler."
                )

        logger.info("RealImageSampler initialized successfully")

    def _get_next(self) -> SynthpixBatch:
        """Get the next batch of real images and flow fields.

        Returns:
            A batch of real images and flow fields.
        """
        # Get the next batch of flow fields from the scheduler
        batch = self.scheduler.get_batch(batch_size=self.batch_size)
        batch = SynthpixBatch(
            images1=jnp.array(batch.images1, dtype=jnp.float32),
            images2=jnp.array(batch.images2, dtype=jnp.float32),
            flow_fields=jnp.array(batch.flow_fields, dtype=jnp.float32),
            params=None,
            done=None,
            mask=jnp.array(batch.mask) if batch.mask is not None else None,
        )
        return batch

    @classmethod
    def from_config(cls, scheduler, config) -> Self:
        """Create a RealImageSampler from configuration.

        Args:
            scheduler: Scheduler instance.
            config: Configuration dictionary.

        Returns:
            An instance of RealImageSampler.
        """
        batch_size = config.get("batch_size", 1)
        return cls(scheduler=scheduler, batch_size=batch_size)
