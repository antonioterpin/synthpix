"""EpisodicFlowFieldScheduler to organize flow fields in episodes."""
import glob
import os
import random

from ..utils import discover_leaf_dirs, logger
from .base import BaseFlowFieldScheduler


class EpisodicFlowFieldScheduler:
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

    Example
    -------
    >>> base = MATFlowFieldScheduler("/data/flows")
    >>> episodic  = EpisodicFlowFieldScheduler(
    ...             scheduler=base,
    ...             batch_size=16,
    ...             episode_length=32,
    ...             seed=42)
    >>> batch_t0 = next(episodic)        # first time-step from 16 episodes
    >>> batch_t1 = next(episodic)        # second time-step
    >>> episodic.reset()                 # start 16 fresh episodes

    Notes
    -----
    * The underlying scheduler (and its prefetching thread) are **not** initialized
      on every episode reset—only the order of ``file_list`` is mutated.  This
      keeps disk I/O sequential and maximises throughput.
    * The wrapper follows the “vector-environment” pattern popular in Gym,
      Gymnax and Brax, so your JAX RL loop can `vmap` or `pmap` over the first
      dimension without shape changes.
    """

    def __init__(
        self,
        scheduler: BaseFlowFieldScheduler,
        batch_size: int,
        episode_length: int,
        seed: int = 0,
    ) -> None:
        """Constructs an episodic scheduler wrapper.

        Args:
            scheduler: Any concrete subclass of :class:`BaseFlowFieldScheduler`
            (e.g. :class:`MATFlowFieldScheduler`).
            batch_size: episodes to run in parallel (== first dim of each batch).
            episode_length: Number of consecutive flow-fields that make up *one* episode.
            seed: Seed for the internal pseudo-random number generator.  Use a fixed
            seed for deterministic episode sampling; use different seeds across
            workers for data-parallel training.

        Raises:
            ValueError
                If ``batch_size`` or ``episode_length`` are not positive, or if the
                dataset does not contain enough distinct starting positions to form
                at least one complete batch of episodes.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if episode_length <= 0:
            raise ValueError("episode_length must be positive")

        self.scheduler = scheduler
        self.batch_size = batch_size
        self.episode_length = episode_length

        self._rng = random.Random(seed)
        self._t = 0

        # Calculate the possible starting positions to sample from
        self.dir2files, self._starts = self._calculate_starts()
        self._sample_new_episodes()

    def __iter__(self) -> "EpisodicFlowFieldScheduler":
        """Returns self so the object can be used in a ``for`` loop."""
        self._t = 0
        return self

    def __next__(self):
        """Return one time-step of shape ``(batch_size, …)``.

        Returns:
            batch: A `(batch_size, …)` tensor that holds one flow-field per episode.
        """
        # If we’ve exhausted the current horizon, start fresh episodes
        if self._t >= self.episode_length:
            self.next_episode()

        self._t += 1

        batch = self.scheduler.get_batch(self.batch_size)

        logger.debug("__next__() called, returning batch of shape {batch.shape}")
        logger.debug(f"timestep: {self._t}")
        return batch

    def get_batch(self, batch_size: int):
        """Return exactly one time-step for `batch_size` parallel episodes.

        *Does not* loop internally – we delegate to the wrapped base
        scheduler once, because `__next__` already returns a full batch.
        """
        if batch_size != self.batch_size:
            raise ValueError(
                f"Requested batch_size {batch_size}, "
                f"but EpisodicFlowFieldScheduler was initialized with "
                f"{self.batch_size}"
            )
        logger.debug(f"get_batch() called with batch_size {batch_size}")
        return next(self)

    def __len__(self) -> int:
        """Return the episode length.

        Returns:
            int: The length of the episode.
        """
        return self.episode_length

    def reset_episode(self):
        """Start *batch_size* brand-new episodes.

        The call is cheap: it only reshuffles ``file_list`` and resets cursors
        """
        self._sample_new_episodes()
        self._t = 0

    def steps_remaining(self) -> int:
        """Return the number of steps remaining in the current episode.

        Returns:
            int: Number of steps remaining in the current episode.
        """
        return self.episode_length - self._t

    def next_episode(self):
        """Advance to the next episode, independenet of the current step.

        This method is useful when you want to skip to the next episode
        without waiting for the current episode to finish.
        """
        self._sample_new_episodes()
        self._t = 0

    def get_flow_fields_shape(self):
        """Return the shape of the flow fields from the underlying scheduler.

        Returns:
            tuple: Shape of the flow fields as returned by the underlying scheduler.
        """
        return self.scheduler.get_flow_fields_shape()

    def _calculate_starts(self) -> tuple[dict[str, list[str]], list[tuple[str, int]]]:
        """Calculate the possible starting positions for the episodes.

        Returns:
            list[tuple[str, int]]:
                A list of tuples containing the directory and the starting index
                for each episode.
        """
        # Extract the leaf directories from the file list
        leaf_dirs = discover_leaf_dirs(self.scheduler.file_list)
        dir2files = {d: sorted(glob.glob(os.path.join(d, "*.mat"))) for d in leaf_dirs}

        # Sanity-check: all directories must contain enough frames
        for d, files in dir2files.items():
            if len(files) < self.episode_length:
                raise ValueError(
                    f"Directory {d} has only {len(files)} files, "
                    f"but episode_length is {self.episode_length}."
                )
        # Enumerate every admissible (dir, start_index) combination
        starts = []
        for d, files in dir2files.items():
            last_start = len(files) - self.episode_length
            starts.extend((d, s) for s in range(last_start + 1))

        return dir2files, starts

    def _sample_new_episodes(self):
        """Create a new interleaved file order and push it into ``scheduler``."""
        # Randomly choose batch_size starts without replacement
        starts = self._rng.choices(self._starts, k=self.batch_size)

        # Build individual episode sequences
        episodes = []
        for d, s in starts:
            # Extract the time-series pattern for this episode
            episodes.append(self.dir2files[d][s : s + self.episode_length])

        # Interleave “time major” → t0_ep0, t0_ep1, …, t1_ep0, …
        interleaved = [
            episodes[ep][t]  # type: ignore[index]
            for t in range(self.episode_length)
            for ep in range(self.batch_size)
        ]

        logger.debug(
            "Order rebuilt — "
            f"{self.batch_size} episodes × "
            f"{self.episode_length} steps = "
            f"{len(interleaved)} files"
        )

        # Inject new order and reset cursors *without* reshuffling internally
        self.scheduler.file_list = interleaved
        self.scheduler.reset(reset_epoch=False)
