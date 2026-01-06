"""Protocol that needs to be followed by schedulers."""

from typing import Any, Protocol, runtime_checkable

from synthpix.types import SchedulerData


@runtime_checkable
class SchedulerProtocol(Protocol):
    """Protocol that needs to be followed by schedulers."""

    def shutdown(self) -> None:
        """Shuts down background processes used for prefetching."""

    def get_flow_fields_shape(self) -> tuple[int, int, int]:
        """Returns the shape of the flow field.

        Returns:
            Shape of the flow field.
        """
        ...

    def get_batch(self, batch_size: int) -> SchedulerData:
        """Retrieves a batch of flow fields using the current scheduler state.

        This method repeatedly calls `__next__()` to store a batch
        of flow field slices.

        Args:
            batch_size: Number of flow field slices to retrieve.

        Returns:
            SchedulerData containing the batch of flow fields
                and, optionally, images.
        """
        ...

    def reset(self) -> None:
        """Resets the state."""
        ...

    @property
    def file_list(self) -> list[str]:
        """Returns the list of files used by the scheduler.

        Returns:
            List of file paths.
        """
        ...

    @file_list.setter
    def file_list(self, value: list[str]) -> None:
        """Sets the list of files used by the scheduler.

        Args:
            value: List of file paths to set.
        """
        ...

    @property
    def state(self) -> dict[str, Any]:
        """Returns the state of the scheduler.

        Returns:
            Dictionary containing the scheduler state.
        """
        ...

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        """Sets the state of the scheduler.

        Args:
            value: Dictionary containing the scheduler state.
        """
        ...

    @property
    def grain_iterator(self) -> Any | None:
        """Returns the underlying Grain iterator if available.

        Returns:
            The Grain iterator or None if not supported.
        """
        ...


@runtime_checkable
class EpisodicSchedulerProtocol(SchedulerProtocol, Protocol):
    """Protocol that needs to be followed by episodic schedulers."""

    def steps_remaining(self) -> int:
        """Returns the number of steps remaining in the current episode.

        Returns:
            Number of steps remaining.
        """
        ...

    def next_episode(self) -> None:
        """Flush the current episode and prepare for the next one.

        The scheduler should reset any internal state necessary for
        starting a new episode.
        """
        ...

    @property
    def episode_length(self) -> int:
        """Returns the length of the episode.

        Returns:
            The length of the episode.
        """
        ...


@runtime_checkable
class PrefetchedSchedulerProtocol(EpisodicSchedulerProtocol, Protocol):
    """Protocol that needs to be followed by prefetched schedulers."""


class EpisodeEndError(Exception):
    """Exception raised when an episode ends in a prefetched scheduler."""


class FileEndedError(Exception):
    """Exception raised when the end of a file's data is reached."""
