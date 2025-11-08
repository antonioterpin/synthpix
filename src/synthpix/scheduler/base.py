"""BaseFlowFieldScheduler abstract class."""

import glob
import os
from abc import ABC, abstractmethod
import numpy as np
from typing_extensions import Self

import jax
import jax.numpy as jnp
from goggles import get_logger

from synthpix.utils import SYNTHPIX_SCOPE
from synthpix.types import PRNGKey, SchedulerData
from synthpix.scheduler.protocol import SchedulerProtocol

logger = get_logger(__name__, scope=SYNTHPIX_SCOPE)


class FileEndedError(Exception):
    """Exception raised when the end of a file's data is reached."""

    pass


class BaseFlowFieldScheduler(ABC, SchedulerProtocol):
    """Abstract class for scheduling access to flow field data.

    This class provides iteration, looping, caching, and batch loading.
    Subclasses must implement:
    - file-specific loading
    - y-slice extraction logic.
    """

    _file_pattern = "*"
    _file_list: list[str]

    @property
    def file_list(self) -> list[str]:
        """Returns the list of files used by the scheduler.

        Returns: List of file paths.
        """
        return self._file_list

    @file_list.setter
    def file_list(self, value: list[str]) -> None:
        """Sets the list of files used by the scheduler.

        Args:
            value: List of file paths.
        """
        self._file_list = value

    def __init__(
        self,
        file_list: list[str],
        randomize: bool = False,
        loop: bool = False,
        key: PRNGKey | None = None,
    ) -> None:
        """Initializes the scheduler.

        Args:
            file_list:  List of file paths to flow field datasets.
            randomize: If True, shuffle the order of files.
            loop: If True, loop over the dataset indefinitely.
            key: Random key for reproducibility.
        """
        # Check if file_list is a list of files or directories
        self._file_list = []
        if (
            file_list is None
            or not isinstance(file_list, list)
            or not all(isinstance(f, str) for f in file_list)
        ):
            raise ValueError("file_list must be a list of file paths.")
        for file_path in file_list:
            if os.path.isdir(file_path):
                logger.debug(f"Searching for files in {file_path}")
                pattern = os.path.join(file_path, self._file_pattern)
                found_files = sorted(glob.glob(pattern, recursive=True))
                logger.debug(f"Found {len(found_files)} files in {file_path}")
                self._file_list.extend(found_files)
            else:
                self._file_list.append(file_path)

        if not self._file_list:
            raise ValueError("The file_list must not be empty.")

        for file_path in self._file_list:
            if not isinstance(file_path, str):
                raise ValueError("All file paths must be strings.")
            if not os.path.isfile(file_path):
                raise ValueError(f"File {file_path} does not exist.")

        if not isinstance(randomize, bool):
            raise ValueError("randomize must be a boolean value.")
        self.randomize = randomize

        if key is not None:
            self.key = key
        else:
            self.key = jax.random.PRNGKey(0)
            cpu = jax.devices("cpu")[0]
            self.key = jax.device_put(self.key, cpu)

        if not isinstance(loop, bool):
            raise ValueError("loop must be a boolean value.")
        self.loop = loop

        self.index = 0

        if self.randomize:
            self.key, shuffle_key = jax.random.split(self.key)
            cpu = jax.devices("cpu")[0]
            file_list_indices = jnp.arange(len(self.file_list), device=cpu)
            file_list_indices = jax.random.permutation(shuffle_key, file_list_indices)
            self.file_list = [self.file_list[i] for i in file_list_indices.tolist()]

        self._cached_data: SchedulerData | None = None
        self._cached_file: str | None = None
        self._slice_idx: int = 0

        logger.debug(
            f"Initialized with {len(self.file_list)} files, "
            f"randomize={self.randomize}, loop={self.loop}"
        )

    def __len__(self) -> int:
        """Returns the number of files in the dataset.

        Returns: Number of files in file_list.
        """
        return len(self.file_list)

    def __iter__(self) -> Self:
        """Returns the iterator instance itself.

        Returns: The iterator instance.
        """
        return self

    def reset(self) -> None:
        """Resets the state."""
        self.index = 0
        self._slice_idx = 0
        self._cached_data = None
        self._cached_file = None
        if self.randomize:
            self.key, shuffle_key = jax.random.split(self.key)
            cpu = jax.devices("cpu")[0]
            file_list_indices = jnp.arange(len(self.file_list), device=cpu)
            file_list_indices = jax.random.permutation(shuffle_key, file_list_indices)
            self.file_list = [self.file_list[i] for i in file_list_indices.tolist()]

    def _get_next(self):
        """Returns the next flow field slice from the dataset.

        Returns: A single flow field slice.

        Raises:
            StopIteration: If no more data and loop is False.
        """
        while self.index < len(self.file_list) or self.loop:
            print("Current index:", self.index)
            print("Current slice index:", self._slice_idx)
            print("File list length:", len(self.file_list))
            # Handle loop around
            if self.index >= len(self.file_list):
                self.reset()

            # Get the current file path
            path = self.file_list[self.index]

            # Load and cache the file if not already cached
            if self._cached_file != path:
                print("Loading file:", path)
                self._cached_file = None
                self._cached_data = None
                try:
                    self._cached_data = self.load_file(path)
                    self._cached_file = path
                    self._slice_idx = 0
                except Exception as e:
                    logger.error(f"Error loading file {path}: {e}")
                    self.index += 1
                    continue  # Skip to the next file

            try:
                sample = self.get_next_slice()
                self._slice_idx += 1
                print("Got next slice:", sample.flow_fields.shape)
                return sample
            except FileEndedError as e:
                self.index += 1
                continue  # Skip to the next file

        raise StopIteration

    def get_batch(self, batch_size: int) -> SchedulerData:
        """Retrieves a batch of flow fields using the current scheduler state.

        This method repeatedly calls `__next__()` to store a batch
        of flow field slices.

        Args:
            batch_size: Number of flow field slices to retrieve.

        Returns: SchedulerData containing the batch of flow field slices.

        Raises:
            StopIteration: If the dataset is exhausted before reaching the
                desired batch size and `loop` is set to False.
        """
        batch = []
        mask = None
        for _ in range(batch_size):
            try:
                print("Getting next at index", _)
                scheduler_data = self._get_next()
                batch.append(scheduler_data)
            except StopIteration:
                print("Caught StopIteration at index ", _)
                break
        if len(batch) == 0:
            if not self.loop:
                raise StopIteration
            self.reset()
        if len(batch) < batch_size and not self.loop:
            mask = np.zeros((batch_size,), dtype=bool)
            mask[: len(batch)] = True

        logger.debug(f"Loaded batch of {len(batch)} flow field slices.")

        images1, images2 = None, None
        if all(data.images1 is not None for data in batch):
            images1 = np.stack([data.images1 for data in batch])
        if all(data.images2 is not None for data in batch):
            images2 = np.stack([data.images2 for data in batch])

        flow_fields = np.stack([data.flow_fields for data in batch])

        # Pad if needed
        if len(batch) < batch_size:
            pad_size = batch_size - len(batch)
            flow_fields = np.pad(
                flow_fields,
                ((0, pad_size), (0, 0), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            if images1 is not None:
                images1 = np.pad(
                    images1,
                    ((0, pad_size), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
            if images2 is not None:
                images2 = np.pad(
                    images2,
                    ((0, pad_size), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

        return SchedulerData(
            flow_fields=flow_fields,
            images1=images1,
            images2=images2,
            mask=mask,
        )

    @abstractmethod
    def load_file(self, file_path: str) -> SchedulerData:
        """Loads a file and returns the dataset for caching.

        Args:
            file_path: Path to the file to be loaded.

        Returns: The loaded dataset.
        """

    @abstractmethod
    def get_next_slice(self) -> SchedulerData:
        """Extracts the next slice from the cached data.

        Returns: SchedulerData containing the next flow field slice
            (and optionally images).
        """

    @abstractmethod
    def get_flow_fields_shape(self) -> tuple[int, int, int]:
        """Returns the shape of the flow field.

        Returns: Shape of the flow field.
        """

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> Self:
        """Creates a BaseFlowFieldScheduler instance from a configuration.

        Args:
            config:
                Configuration dictionary containing the scheduler parameters.

        Returns: A BaseFlowFieldScheduler instance.
        """
