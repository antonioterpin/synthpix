"""EpisodicDataSource implementation (Wrapper)."""

import logging
import os

import grain.python as grain
import numpy as np

from .base import FileDataSource

logger = logging.getLogger(__name__)


class EpisodicDataSource(grain.RandomAccessDataSource):
    """Wrapper that serves flow-field episodes in parallel batches using Grain.

    The wrapper reorganizes the random access order of an underlying
    :class:`FileDataSource` so that a sequential read (e.g. by ``grain.Batch``)
    returns the *same* time-step for ``batch_size`` independent episodes.

    The data on disk must be organized as::

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

    This class serves as a drop-in replacement for the legacy
    ``EpisodicFlowFieldScheduler``, but follows the Grain DataSource API.

    Wrapper Pattern:
    ----------------
    This class wraps a concrete :class:`FileDataSource` (like
    :class:`MATDataSource` or :class:`HDF5DataSource`). The discovery of files
    is handled by the wrapped source, but ``EpisodicDataSource`` re-indexes
    them to form episodes.
    It then delegates the actual file loading back to the wrapped source via
    ``source.load_file()``. This ensures that file-format specifics (MAT vs
    HDF5) remain decoupled from the episodic batching logic.
    """

    def __init__(
        self,
        source: FileDataSource,
        batch_size: int,
        episode_length: int,
        seed: int = 42,
    ):
        """Initializes the EpisodicDataSource wrapper.

        Args:
            source: The underlying FileDataSource (e.g. MATDataSource) that
                loads individual files.
            batch_size: Number of parallel episodes (must match the batch_size
                of the DataLoader).
            episode_length: Number of steps per episode.
            seed: Random seed for shuffling episodes.

        Raises:
            TypeError: If source is not an instance of FileDataSource.
        """
        if not isinstance(source, FileDataSource):
            raise TypeError(
                f"source must be an instance of FileDataSource, got "
                f"{type(source)}"
            )

        self.source = source
        self.batch_size = batch_size
        self.episode_length = episode_length
        self._rng = np.random.default_rng(seed)

        # 1. Discover Episodes from source.file_list
        # We assume the source.file_list contains ALL valid files sorted/grouped
        # by dir?
        # FileDataSource scans recursively.
        # We need to re-group them by directory to identify episodes.
        self.dir2files, self._starts = self._calculate_starts(source.file_list)

        logger.info(
            f"EpisodicDataSource: Found {len(self._starts)} valid episode "
            f"starts from {len(source.file_list)} files."
        )

        # 2. Build Interleaved List (Pre-compute epoch order)
        self._interleaved_files = self._build_interleaved_file_list()

    def _calculate_starts(
        self, file_list: list[str]
    ) -> tuple[dict[str, list[str]], list[tuple[str, int]]]:
        """Group files by directory and calculate valid start indices.

        Args:
            file_list: List of file paths.

        Returns:
            Tuple containing:
                - dir2files: Dictionary mapping directories to lists of file
                  paths.
                - starts: List of valid start indices for episodes.
        """
        # Group by directory (assuming leaf dir = episode)
        leaf_dirs: dict[str, list[str]] = {}
        for f in file_list:
            d = os.path.dirname(f)
            if d not in leaf_dirs:
                leaf_dirs[d] = []
            leaf_dirs[d].append(f)

        # Sort files in each dir to ensure temporal order
        for files in leaf_dirs.values():
            files.sort()

        starts = []
        for d, files in leaf_dirs.items():
            # Check length availability
            if len(files) >= self.episode_length:
                # Valid start indices: from 0 up to (N - L)
                last_start = len(files) - self.episode_length
                for s in range(last_start + 1):
                    starts.append((d, s))

        return leaf_dirs, starts

    def _build_interleaved_file_list(self) -> list[tuple[str, int, int]]:
        """Builds the single interleaved list of files for the epoch.

        Returns:
            List of tuples containing (file_path, chunk_id,
            timestep_in_episode).
        """
        # 1. Shuffle all possible episode starts
        shuffled_starts = list(self._starts)  # copy
        self._rng.shuffle(shuffled_starts)

        # 2. Partition into chunks handling batch_size
        num_chunks = len(shuffled_starts) // self.batch_size
        # Note: We drop remainder episodes that don't fit into a full batch
        # set. This is consistent with 'drop_remainder' batching logic often
        # used in training.

        interleaved = []
        for i in range(num_chunks):
            # Take a chunk of 'batch_size' episodes
            chunk_starts = shuffled_starts[
                i * self.batch_size: (i + 1) * self.batch_size
            ]

            # Resolve to file paths for these episodes
            chunk_episodes = []
            for d, s in chunk_starts:
                # Slice the episode files: [s : s + L]
                chunk_episodes.append(
                    self.dir2files[d][s: s + self.episode_length]
                )

            # 3. Interleave in Time-Major order
            # (Time 0 of Ep 0, Time 0 of Ep 1... Time 1 of Ep 0...)
            for t in range(self.episode_length):
                for ep_idx in range(self.batch_size):
                    file_path = chunk_episodes[ep_idx][t]
                    # Store (path, chunk_id, timestep_in_episode) so we can
                    # attach metadata
                    interleaved.append((file_path, i, t))

        return interleaved

    def __len__(self) -> int:
        """Returns the number of files in the dataset."""
        return len(self._interleaved_files)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Loads a file and returns the data dictionary.

        Args:
            idx: Index of the file to load.

        Returns:
            Dictionary containing data (e.g. 'flow_fields', 'images1', etc).
        """
        file_path, chunk_id, t = self._interleaved_files[idx]

        # DELEGATION: Load the file using the wrapped source
        data = self.source.load_file(file_path)

        # Attach Episodic Metadata
        # (This allows downstream collaors/samplers to know boundaries)
        data["_chunk_id"] = chunk_id
        data["_timestep"] = t
        data["_is_last_step"] = t == self.episode_length - 1

        return data

    @property
    def include_images(self) -> bool:
        """Returns True if the underlying data source provides images."""
        return self.source.include_images

    @property
    def file_list(self) -> list[str]:
        """Returns the list of files in the dataset."""
        return self.source.file_list
