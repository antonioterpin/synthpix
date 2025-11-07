"""Base class for Samplers in the SynthPix framework."""

from abc import ABC, abstractmethod
from typing_extensions import Self

import jax.numpy as jnp
from goggles import get_logger

from synthpix.utils import SYNTHPIX_SCOPE
from synthpix.types import SynthpixBatch
from synthpix.scheduler.protocol import (
    SchedulerProtocol,
    EpisodicSchedulerProtocol,
    PrefetchedSchedulerProtocol
)

logger = get_logger(__name__, scope=SYNTHPIX_SCOPE)


class Sampler(ABC):
    """Base class for Samplers in the SynthPix framework."""

    @abstractmethod
    def _get_next(self) -> SynthpixBatch:
        """Generates the next batch of data.

        Returns: The next batch of data as a SynthpixBatch instance.
        """

    def __init__(self, scheduler: SchedulerProtocol, batch_size: int = 1):
        """Initialize the sampler.

        Args:
            scheduler: Scheduler instance that provides data.
            batch_size: Number of samples to return in each batch.
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        self.scheduler = scheduler
        self.batch_size = batch_size

        episodic = isinstance(self.scheduler, EpisodicSchedulerProtocol)
        logger.info(
            "The underlying scheduler is "
            f"{'' if episodic else 'not'} episodic."
        )  # pragma: no cover

        self.batch_size = batch_size
        logger.debug(f"Scheduler class: {self.scheduler.__class__.__name__}")

    def _shutdown(self) -> None:
        """Custom shutdown logic for the sampler."""

    def _reset(self) -> None:
        """Custom reset logic for the sampler."""

    def shutdown(self) -> None:
        """Shutdown the sampler."""
        logger.info(f"Shutting down {self.__class__.__name__}.")
        self._shutdown()
        if isinstance(self.scheduler, PrefetchedSchedulerProtocol):
            self.scheduler.shutdown()
        logger.info(f"{self.__class__.__name__} shutdown complete.")

    def __iter__(self) -> Self:
        """Returns the iterator instance itself."""
        return self

    def __next__(self) -> SynthpixBatch:
        """Return the next batch of data.

        Returns: The next batch of data as a SynthpixBatch instance.
        """
        if (
            isinstance(self.scheduler, EpisodicSchedulerProtocol)
            and self.scheduler.steps_remaining() == 0
        ):
            raise IndexError(
                "Episode ended. No more flow fields available. "
                "Use next_episode() to continue."
            )

        batch = self._get_next()

        if isinstance(self.scheduler, EpisodicSchedulerProtocol):
            done = self._make_done()
            batch = batch.update(done=done)

        return batch

    def reset(self, scheduler_reset: bool = True) -> None:
        """Reset the sampler to its initial state.

        Args:
            scheduler_reset: If True, also resets the underlying scheduler.
        """
        self._reset()
        if scheduler_reset:
            self.scheduler.reset()
        logger.debug(f"{self.__class__.__name__} has been reset.")

    def next_episode(self) -> SynthpixBatch:
        """Flush the current episode and return the first batch of the next one.

        The underlying scheduler is expected to be the prefetching scheduler.

        Returns: The first batch of the next episode.
        """
        if not isinstance(self.scheduler, EpisodicSchedulerProtocol):
            raise AttributeError(
                "Underlying scheduler lacks next_episode() method."
            )

        self.scheduler.next_episode()

        return next(self)

    def _make_done(self) -> jnp.ndarray:
        """Return a `(batch_size,)` bool array if episodic, else None."""
        if not isinstance(self.scheduler, EpisodicSchedulerProtocol):
            raise NotImplementedError(
                "The underlying scheduler is not episodic."
            )

        is_last_step = self.scheduler.steps_remaining() == 0
        logger.debug(f"Is last step: {is_last_step}")
        logger.debug(f"Steps remaining: {self.scheduler.steps_remaining()}")
        # broadcast identical flag to every episode (synchronous horizons)
        # implemented like this to make it easier in the future to implement
        # asynchronous horizons
        return jnp.full((self.batch_size,), is_last_step, dtype=bool)
