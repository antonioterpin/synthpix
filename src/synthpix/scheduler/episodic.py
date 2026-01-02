"""EpisodicFlowFieldScheduler to organize flow fields in episodes."""

import glob
import os

import goggles as gg
import jax
import jax.numpy as jnp
from typing_extensions import Self

from synthpix.scheduler.protocol import (
    EpisodeEndError,
    EpisodicSchedulerProtocol,
    SchedulerProtocol,
)
from synthpix.types import PRNGKey, SchedulerData
from synthpix.utils import SYNTHPIX_SCOPE, discover_leaf_dirs

logger = gg.get_logger(__name__, scope=SYNTHPIX_SCOPE)


class EpisodicFlowFieldScheduler(EpisodicSchedulerProtocol):
    """Wrapper that serves flow-field *episodes* in parallel batches.

    The wrapper rearranges the ``file_list`` of an underlying
    :class:`BaseFlowFieldScheduler` so that **one call** to
    :py:meth:`BaseFlowFieldScheduler.get_batch` (or ``next(self)``) returns the
    *same* time-step for ``batch_size`` independent episodes.

    The data on disk must be organised as::

        root/
          ├── seq_A/       # leaf directory (no further sub-dirs)
          │   ├── 0000.mat
          │   ├── 0001.mat
          │   └── ...
          ├── seq_B/
          │   ├── 0000.mat
          │   └── ...
          └── ...

    The files inside each leaf directory must already be in temporal order when
    sorted alphabetically (e.g. zero-padded integers in the file name).

    Example:
    -------
    >>> base = MATFlowFieldScheduler("/data/flows")
    >>> episodic  = EpisodicFlowFieldScheduler(
    ...             scheduler=base,
    ...             batch_size=16,
    ...             episode_length=32,
    ...             key=jax.random.PRNGKey(42))
    >>> batch_t0 = next(episodic)        # first time-step from 16 episodes
    >>> batch_t1 = next(episodic)        # second time-step
    >>> episodic.reset()                 # start 16 fresh episodes

    Notes:
    -----
    * The underlying scheduler (and its prefetching thread) are **not**
      initialized on every episode reset—only the order of ``file_list`` is
      mutated. This keeps disk I/O sequential and maximises throughput.
    * The wrapper follows the “vector-environment” pattern popular in Gym,
      Gymnax and Brax, so your JAX RL loop can `vmap` or `pmap` over the first
      dimension without shape changes.
    """

    def __init__(
        self,
        scheduler: SchedulerProtocol,
        batch_size: int,
        episode_length: int,
        key: PRNGKey | None = None,
    ):
        """Constructs an episodic scheduler wrapper.

        Args:
            scheduler: Any scheduler that implements SchedulerProtocol
                (e.g., :class:`MATFlowFieldScheduler`,
                :class:`HDF5FlowFieldScheduler`).
            batch_size: Episodes to run in parallel
                (== first dim of each batch).
            episode_length: Number of consecutive flow-fields that
                make up one episode.
            key: Random key for reproducibility.

        Raises:
            TypeError: If scheduler does not implement SchedulerProtocol.
            ValueError: If batch_size or episode_length are not positive,
                or if the dataset does not contain enough distinct starting
                positions to form at least one complete batch of episodes.
        """
        if not isinstance(scheduler, SchedulerProtocol):
            raise TypeError(
                f"Expected scheduler to be a SchedulerProtocol, "
                f"got {type(scheduler)}"
            )
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        if not isinstance(episode_length, int) or episode_length <= 0:
            raise ValueError("episode_length must be a positive integer")

        self.scheduler = scheduler
        self.batch_size = batch_size
        self._episode_length = episode_length

        self._key = key if key is not None else jax.random.PRNGKey(0)
        self._t = 0

        # Calculate the possible starting positions to sample from
        self.dir2files, self._starts = self._calculate_starts()
        self._sample_new_episodes()

    def __iter__(self) -> Self:
        """Returns self so the object can be used in a ``for`` loop."""
        self._t = 0
        return self

    @property
    def file_list(self) -> list[str]:
        """Return the current file list from the underlying scheduler.

        Returns:
            The current file list.
        """
        return list(self.scheduler.file_list)

    @file_list.setter
    def file_list(self, value: list[str]) -> None:
        """Set the file list in the underlying scheduler.

        Args:
            value: New file list to set.
        """
        self.scheduler.file_list = value

    def get_batch(self, batch_size: int) -> SchedulerData:
        """Return exactly one time-step for `batch_size` parallel episodes.

        *Does not* loop internally, we delegate to the wrapped base
        scheduler once, because `__next__` already returns a full batch.

        Args:
            batch_size: Must match the ``batch_size`` used at initialization.

        Returns:
            SchedulerData containing the flow fields for the current time-step
            across all episodes.

        Raises:
            EpisodeEndError: If the current episode has been exhausted.
            ValueError: If batch_size does not match initialization value.
        """
        if batch_size != self.batch_size:
            raise ValueError(
                f"Requested batch_size {batch_size}, "
                f"but EpisodicFlowFieldScheduler was initialized with "
                f"{self.batch_size}"
            )
        logger.debug(f"get_batch() called with batch_size {batch_size}")

        if self._t >= self.episode_length:
            # If we've exhausted the current horizon,
            # wait for the next episode
            raise EpisodeEndError

        self._t += 1

        batch = self.scheduler.get_batch(batch_size)

        logger.debug(f"timestep: {self._t}")
        return batch

    def __len__(self) -> int:
        """Return the episode length.

        Returns:
            The length of the episode.
        """
        return self.episode_length

    @property
    def episode_length(self) -> int:
        """Return the length of the episode.

        Returns:
            The length of the episode.
        """
        return self._episode_length

    def reset_episode(self) -> None:
        """Start *batch_size* brand-new episodes.

        The call is cheap: it only reshuffles ``file_list`` and resets cursors
        """
        self._sample_new_episodes()
        self._t = 0

    def reset(self) -> None:
        """Start *batch_size* brand-new episodes.

        Alias for reset_episode() to maintain API consistency with
        BaseFlowFieldScheduler.
        """
        self.reset_episode()

    def steps_remaining(self) -> int:
        """Return the number of steps remaining in the current episode.

        Returns:
            Number of steps remaining in the current episode.
        """
        return self.episode_length - self._t

    def next_episode(self) -> None:
        """Advance to the next episode, independent of the current step.

        This method is useful when you want to skip to the next episode
        without waiting for the current episode to finish.
        """
        self._sample_new_episodes()
        self._t = 0

    def get_flow_fields_shape(self) -> tuple[int, int, int]:
        """Return the shape of the flow fields from the underlying scheduler.

        Returns:
            Shape of the flow fields as returned by the underlying scheduler.
        """
        shape = self.scheduler.get_flow_fields_shape()
        return (int(shape[0]), int(shape[1]), int(shape[2]))

    def _calculate_starts(
        self,
    ) -> tuple[dict[str, list[str]], list[tuple[str, int]]]:
        """Calculate the possible starting positions for the episodes.

        Returns:
            - A dictionary mapping directories to their sorted file lists
            - A list of tuples containing the directory and the starting index
                for each possible episode start position.

        Raises:
            ValueError: If any directory has fewer files than episode_length.
        """
        # Extract the leaf directories from the file list
        leaf_dirs = discover_leaf_dirs(self.file_list)
        dir2files = {
            d: sorted(glob.glob(os.path.join(d, "*.mat"))) for d in leaf_dirs
        }

        # Sanity-check: all directories must contain enough frames
        for d, files in dir2files.items():
            if len(files) < self.episode_length:
                raise ValueError(
                    f"Directory {d} has only {len(files)} files, "
                    f"but episode_length is {self.episode_length}."
                )
        # Enumerate every admissible (dir, start_index) combination
        starts: list[tuple[str, int]] = []
        for d, files in dir2files.items():
            last_start = len(files) - self.episode_length
            starts.extend((d, s) for s in range(last_start + 1))

        return dir2files, starts

    def _sample_new_episodes(self) -> None:
        """Create a new interleaved file order for the ``scheduler``."""
        # Randomly choose batch_size starts indices without replacement
        cpu = jax.devices("cpu")[0]
        indices = jnp.arange(len(self._starts), dtype=jnp.int32)

        self._key, starts_key = jax.random.split(self._key)
        # We perform sampling on CPU device to avoid GPU transfers if possible
        with jax.default_device(cpu):
            sampled_indices = jax.random.choice(
                starts_key, indices, shape=(self.batch_size,), replace=True
            )

        # Map the indices to directories
        sampled_starts = [self._starts[int(s)] for s in sampled_indices]

        # Build individual episode sequences
        episodes = []
        for d, s in sampled_starts:
            # Extract the time-series pattern for this episode
            episodes.append(self.dir2files[d][s : s + self.episode_length])

        # Interleave “time major” → t0_ep0, t0_ep1, …, t1_ep0, …
        interleaved = [
            episodes[ep][t]
            for t in range(self.episode_length)
            for ep in range(self.batch_size)
        ]

        logger.debug(
            "Order rebuilt — "
            f"{self.batch_size} episodes x "
            f"{self.episode_length} steps = "
            f"{len(interleaved)} files"
        )

        # Inject new order and reset cursors without reshuffling internally
        self.scheduler.file_list = interleaved
        self.scheduler.reset()
