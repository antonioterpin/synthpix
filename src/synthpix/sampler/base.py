"""Base class for Samplers in the SynthPix framework."""

from abc import abstractmethod

import jax.numpy as jnp
from goggles import get_logger

logger = get_logger(__name__)


class Sampler:
    """Base class for Samplers in the SynthPix framework."""

    def __init__(self, scheduler, batch_size=1):
        """Initialize the sampler.

        Args:
            scheduler: Scheduler instance that provides data.
            batch_size (int): Number of samples to return in each batch.
        """
        if not hasattr(scheduler, "__iter__"):
            raise ValueError("scheduler must be an iterable object.")
        if not hasattr(scheduler, "__next__"):
            raise ValueError(
                "scheduler must be an iterable object with __next__ method."
            )
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        self.scheduler = scheduler
        self.batch_size = batch_size
        self._episodic = False

        self._episodic = hasattr(scheduler, "episode_length")
        logger.info(
            f"The underlying scheduler is {'' if self._episodic else 'not'} episodic."
        )  # pragma: no cover

        self.batch_size = batch_size
        logger.debug(f"Scheduler class: {self.scheduler.__class__.__name__}")

    def _shutdown(self):
        """Custom shutdown logic for the sampler."""

    def _reset(self):
        """Custom reset logic for the sampler."""

    def shutdown(self):
        """Shutdown the sampler."""
        logger.info(f"Shutting down {self.__class__.__name__}.")
        self._shutdown()
        if hasattr(self.scheduler, "shutdown"):
            self.scheduler.shutdown()
        logger.info(f"{self.__class__.__name__} shutdown complete.")

    def __iter__(self):
        """Returns the iterator instance itself."""
        return self

    @abstractmethod
    def ___next__(self):
        """Generates the next batch of data."""

    def __next__(self):
        """Return the next batch of data."""
        if self._episodic and self.scheduler.steps_remaining() == 0:
            raise IndexError(
                "Episode ended. No more flow fields available. "
                "Use next_episode() to continue."
            )

        batch = self.___next__()

        if self._episodic:
            done = self._make_done()
            batch["done"] = done

        return batch

    def reset(self, scheduler_reset: bool = True):
        """Reset the sampler to its initial state."""
        self._reset()
        if scheduler_reset:
            self.scheduler.reset()
        logger.debug(f"{self.__class__.__name__} has been reset.")

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
