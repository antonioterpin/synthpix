"""Sampler for real data."""

import jax.numpy as jnp

from ..utils import logger


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

        if hasattr(self.scheduler, "episode_length"):
            self._episodic = True
            logger.info("The underlying scheduler is episodic.")
        else:
            self._episodic = False
            logger.info("The underlying scheduler is not episodic.")

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
        # Get the next batch of flow fields from the scheduler
        if self._episodic and self.scheduler.steps_remaining() == 0:
            raise IndexError(
                "Episode ended. No more flow fields available. "
                "Use next_episode() to continue."
            )

        batch = self.scheduler.get_batch(batch_size=self.batch_size)
        batch = {
            "images1": jnp.array(batch[0], dtype=jnp.float32),
            "images2": jnp.array(batch[1], dtype=jnp.float32),
            "flow_fields": jnp.array(batch[2], dtype=jnp.float32),
            "params": {
                "seeding_densities": jnp.zeros(batch[0].shape[0], dtype=jnp.float32),
                "diameter_ranges": jnp.zeros((batch[0].shape[0], 2), dtype=jnp.float32),
                "intensity_ranges": jnp.zeros(
                    (batch[0].shape[0], 2), dtype=jnp.float32
                ),
                "rho_ranges": jnp.zeros((batch[0].shape[0], 2), dtype=jnp.float32),
            },
        }
        if self._episodic:
            batch["done"] = self._make_done()

        return batch

    def next_episode(self):
        """Flush the current episode and return the first batch of the next one.

        The underlying scheduler is expected to be the prefetching scheduler.

        Returns:
            next(self): dict
                The first batch of the next episode.
        """
        if not hasattr(self.scheduler, "next_episode"):
            raise AttributeError("Underlying scheduler lacks next_episode() method.")

        self.scheduler.next_episode()

        return next(self)

    def _make_done(self):
        """Return a `(batch_size,)` bool array if episodic, else None."""
        if not self._episodic:
            raise NotImplementedError("The underlying scheduler is not episodic.")

        is_last_step = self.scheduler.steps_remaining() == 0
        logger.debug(f"Is last step: {is_last_step}")
        logger.debug(f"Steps remaining: {self.scheduler.steps_remaining()}")
        # broadcast identical flag to every episode (synchronous horizons)
        # implemented like this to make it easier in the future to implement
        # asynchronous horizons
        return jnp.full((self.batch_size,), is_last_step, dtype=bool)

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
