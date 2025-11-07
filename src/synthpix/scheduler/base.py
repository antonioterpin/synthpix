"""BaseFlowFieldScheduler abstract class."""

import glob
import os
from abc import ABC, abstractmethod
import numpy as np
from typing_extensions import Self

import jax
import jax.numpy as jnp
import itertools as it
from goggles import get_logger

from synthpix.utils import SYNTHPIX_SCOPE
from synthpix.types import PRNGKey, SchedulerData
from synthpix.scheduler.protocol import SchedulerProtocol

logger = get_logger(__name__, scope=SYNTHPIX_SCOPE)


class BaseFlowFieldScheduler(ABC, SchedulerProtocol):
    """Abstract class for scheduling access to flow field data.

    This class provides iteration, looping, caching, and batch loading.
    Subclasses must implement:
    - file-specific loading
    - y-slice extraction logic.
    """

    _file_pattern = "*"

    def __init__(
        self,
        file_list: list[str] | str,
        randomize: bool = False,
        loop: bool = False,
        key: PRNGKey | None = None,
    ) -> None:
        """Initializes the scheduler.

        Args:
            file_list:  List of file paths to flow field datasets.
            randomize: If True, shuffle the order of files each epoch.
            loop: If True, loop over the dataset indefinitely.
            key: Random key for reproducibility.
        """
        # Check if file_list is a directory or a list of files
        if isinstance(file_list, str) and os.path.isdir(file_list):
            logger.debug(f"Searching for files in {file_list}")
            file_path = file_list
            pattern = os.path.join(file_list, self._file_pattern)
            file_list = sorted(glob.glob(pattern, recursive=True))
            logger.debug(f"Found {len(file_list)} files in {file_path}")
        elif isinstance(file_list, str) and os.path.isfile(file_list):
            file_list = [file_list]

        if not file_list:
            raise ValueError("The file_list must not be empty.")

        for file_path in file_list:
            if not isinstance(file_path, str):
                raise ValueError("All file paths must be strings.")
            if not os.path.isfile(file_path):
                raise ValueError(f"File {file_path} does not exist.")

        self.file_list = file_list

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

        self.epoch = 0
        self.index = 0

        if self.randomize:
            self.key, shuffle_key = jax.random.split(self.key)
            cpu = jax.devices("cpu")[0]
            file_list_indices = jnp.arange(len(self.file_list), device=cpu)
            file_list_indices = jax.random.permutation(shuffle_key, file_list_indices)
            self.file_list = [self.file_list[i] for i in file_list_indices.tolist()]

        self._cached_data = None
        self._cached_file = None
        self._slice_idx = 0

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

    def reset(self, reset_epoch: bool = True) -> None:
        """Resets the state and, optionally, epoch count.

        Args:
            reset_epoch: If True, resets the epoch counter to zero.
        """
        if reset_epoch:
            self.epoch = 0
        self.index = 0
        self._slice_idx = 0
        self._cached_data = None
        self._cached_file = None
        if self.randomize:
            self.key, shuffle_key = jax.random.split(self.key)
            cpu = jax.devices("cpu")[0]
            file_list_indices = jnp.arange(len(self.file_list), device=cpu)
            file_list_indices = jax.random.permutation(shuffle_key, file_list_indices)
            self.file_list = [
                self.file_list[i] for i in file_list_indices.tolist()
            ]
        if reset_epoch:
            logger.info("Scheduler state has been reset.")

    def _get_next(self):
        """Returns the next flow field slice from the dataset.

        Returns: A single flow field slice.

        Raises:
            StopIteration: If no more data and loop is False.
        """
        while self.index < len(self.file_list) or self.loop:
            if self.index >= len(self.file_list):
                self.reset(reset_epoch=False)
                logger.info(f"Starting epoch {self.epoch}")

            path = self.file_list[self.index]
            try:
                # load and cache
                self._cached_file = path
                self._cached_data = self.load_file(path)

                # extract and return
                sample = self.get_next_slice()
                self.index += 1
                return sample

            except Exception as e:
                logger.error(f"Skipping {path}: {e}")
                self.index += 1
                continue

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
        for _ in range(batch_size):
            try:
                scheduler_data = self._get_next()
                batch.append(scheduler_data)
            except StopIteration:
                break
        if len(batch) < batch_size and not self.loop:
            logger.warning(
                f"Skipping the last {len(batch)} slices."
                "If undesired, use loop or a batch size dividing "
                "the number of slices in the dataset."
            )
            raise StopIteration

        logger.debug(f"Loaded batch of {len(batch)} flow field slices.")

        images1, images2 = None, None
        if all(
            data.images1 is not None for data in batch
        ):
            images1 = np.stack([data.images1 for data in batch])
        if all(
            data.images2 is not None for data in batch
        ):
            images2 = np.stack([data.images2 for data in batch])

        return SchedulerData(
            flow_fields=np.stack([data.flow_fields for data in batch]),
            images1=images1,
            images2=images2,
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