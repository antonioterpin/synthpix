"""Sampler for real data."""

import jax
import jax.numpy as jnp
from typing_extensions import Self

from synthpix.sampler.base import Sampler
from synthpix.scheduler import SchedulerProtocol
from synthpix.scheduler.episodic import EpisodicFlowFieldScheduler
from synthpix.scheduler.prefetch import PrefetchingFlowFieldScheduler
from synthpix.types import SynthpixBatch
from synthpix.utils import SYNTHPIX_SCOPE, get_logger

logger = get_logger(__name__, scope=SYNTHPIX_SCOPE)

_make_keys = jax.jit(jax.vmap(jax.random.PRNGKey))


@jax.jit
def _as_float32(*arrays: jax.Array) -> tuple[jax.Array, ...]:
    return tuple(a.astype(jnp.float32) for a in arrays)


class RealImageSampler(Sampler):
    """Sampler for real data."""

    def __init__(self, scheduler: SchedulerProtocol, batch_size: int = 1):
        """Initialize the sampler.

        Args:
            scheduler: Scheduler instance that provides real images.
            batch_size: Number of images to sample in each batch.

        Raises:
            ValueError: If base scheduler does not have include_images=True.
        """
        super().__init__(scheduler, batch_size)

        while not getattr(scheduler, "include_images", False):
            if isinstance(
                scheduler,
                EpisodicFlowFieldScheduler | PrefetchingFlowFieldScheduler,
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
        keys = (
            _make_keys(jnp.asarray(batch.jax_seed))
            if batch.jax_seed is not None
            else None
        )
        # Copy in the stored dtypes (images are often uint8, 4x smaller)
        # and cast on the device: the host-to-device copy dominates.
        images1, images2, flow_fields = _as_float32(
            *jax.device_put((batch.images1, batch.images2, batch.flow_fields))
        )
        batch = SynthpixBatch(
            images1=images1,
            images2=images2,
            flow_fields=flow_fields,
            params=None,
            done=None,
            mask=(
                jnp.asarray(batch.mask, dtype=jnp.bool_)
                if batch.mask is not None
                else jnp.ones(batch.flow_fields.shape[0], dtype=jnp.bool_)
            ),
            files=batch.files,
            epoch=jnp.array(batch.epoch) if batch.epoch is not None else None,
            seeds=jnp.array(batch.jax_seed)
            if batch.jax_seed is not None
            else None,
            keys=keys,
        )
        return batch

    @classmethod
    def from_config(cls, scheduler: SchedulerProtocol, config: dict) -> Self:
        """Create a RealImageSampler from configuration.

        Args:
            scheduler: Scheduler instance.
            config: Configuration dictionary.

        Returns:
            An instance of RealImageSampler.
        """
        batch_size = config.get("batch_size", 1)
        return cls(scheduler=scheduler, batch_size=batch_size)
