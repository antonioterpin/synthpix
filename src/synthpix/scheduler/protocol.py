"""Protocol that needs to be followed by schedulers."""

from typing import Protocol, runtime_checkable
from synthpix.types import SchedulerData


@runtime_checkable
class SchedulerProtocol(Protocol):
    """Protocol that needs to be followed by schedulers."""

    def get_flow_fields_shape(self) -> tuple[int, int, int]:
        """Returns the shape of the flow field.

        Returns: Shape of the flow field.
        """
        ...

    def get_batch(self, batch_size: int) -> list[SchedulerData]:
        """Retrieves a batch of flow fields using the current scheduler state.

        This method repeatedly calls `__next__()` to store a batch
        of flow field slices.

        Args:
            batch_size: Number of flow field slices to retrieve.

        Returns: A list of SchedulerData containing the batch of flow fields 
            and, optionally, images.
        """
        ...

    def reset(self, reset_epoch: bool = True) -> None:
        """Resets the state and, optionally, epoch count.

        Args:
            reset_epoch: If True, resets the epoch counter to zero.
        """
        ...

@runtime_checkable
class EpisodicSchedulerProtocol(SchedulerProtocol, Protocol):
    """Protocol that needs to be followed by episodic schedulers."""

    def steps_remaining(self) -> int:
        """Returns the number of steps remaining in the current episode.

        Returns: Number of steps remaining.
        """
        ...

    def next_episode(self) -> None:
        """Flush the current episode and prepare for the next one.

        The scheduler should reset any internal state necessary for
        starting a new episode.
        """
        ...

@runtime_checkable
class PrefetchedSchedulerProtocol(SchedulerProtocol, Protocol):
    """Protocol that needs to be followed by prefetched schedulers."""

    def shutdown(self) -> None:
        """Shuts down any background processes or threads used for prefetching."""
        ...