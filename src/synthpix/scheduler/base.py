"""BaseFlowFieldScheduler abstract class."""
import glob
import os
import random
from abc import ABC, abstractmethod

import numpy as np

from ..utils import logger


class BaseFlowFieldScheduler(ABC):
    """Abstract class for scheduling access to flow field data from various file formats.

    This class provides iteration, looping, caching, and batch loading capabilities.
    Subclasses must implement file-specific loading and y-slice extraction logic.
    """

    _file_pattern = "*"

    def __init__(self, file_list, randomize=False, loop=False):
        """Initializes the scheduler.

        Args:
            file_list: list
                List of file paths to flow field datasets.
            randomize: bool
                If True, shuffle the order of files each epoch.
            loop: bool
                If True, loop over the dataset indefinitely.
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

        if not isinstance(loop, bool):
            raise ValueError("loop must be a boolean value.")
        self.loop = loop

        self.epoch = 0
        self.index = 0

        if self.randomize:
            random.shuffle(self.file_list)

        self._cached_data = None
        self._cached_file = None
        self._slice_idx = 0

        logger.debug(
            f"Initialized with {len(self.file_list)} files, "
            f"randomize={self.randomize}, loop={self.loop}"
        )

    def __len__(self):
        """Returns the number of files in the dataset.

        Returns:
            int: Number of files in file_list.
        """
        return len(self.file_list)

    def __iter__(self):
        """Returns the iterator instance itself.

        Returns:
            BaseFlowFieldScheduler: The iterator instance.
        """
        return self

    def reset(self, reset_epoch=True):
        """Resets the state, including file pointers and, optionally, epoch count."""
        if reset_epoch:
            self.epoch = 0
        self.index = 0
        self._slice_idx = 0
        self._cached_data = None
        self._cached_file = None
        if self.randomize:
            random.shuffle(self.file_list)
        if reset_epoch:
            logger.info("Scheduler state has been reset.")

    def __next__(self) -> np.ndarray:
        """Returns the next flow field slice from the dataset.

        Returns:
            np.ndarray: A single flow field slice.

        Raises:
            StopIteration: If no more data and loop is False.
        """
        while self.index < len(self.file_list) or self.loop:
            if self.index >= len(self.file_list):
                self.reset(reset_epoch=False)

            file_path = self.file_list[self.index]

            try:
                if self._cached_file != file_path:
                    self._cached_data = self.load_file(file_path)
                    self._cached_file = file_path
                    self._slice_idx = 0

                if self._slice_idx >= self._cached_data.shape[1]:
                    self.index += 1
                    self._cached_file = None
                    self._cached_data = None
                    continue

                flow_field = self.get_next_slice()
                logger.debug(
                    f"Loaded slice y={self._slice_idx} from {file_path}, "
                    f"shape {flow_field.shape}"
                )
                self._slice_idx += 1
                return flow_field

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                self.index += 1
                self._cached_data = None
                self._cached_file = None
                continue

        raise StopIteration

    def get_batch(self, batch_size) -> np.ndarray:
        """Retrieves a batch of flow fields using the current scheduler state.

        This method repeatedly calls `__next__()` to store a batch of flow field slices.

        Args:
            batch_size: int
                Number of flow field slices to retrieve.

        Returns:
            np.ndarray: A np.ndarray of flow field slices with length `batch_size`.

        Raises:
            StopIteration: If the dataset is exhausted before reaching the
                           desired batch size and `loop` is set to False.
        """
        batch = []
        try:
            for _ in range(batch_size):
                batch.append(next(self))
        except StopIteration:
            if not self.loop and batch:
                logger.warning(
                    f"Only {len(batch)} slices could be loaded before exhaustion."
                )
                return np.array(batch)
            raise

        logger.debug(f"Loaded batch of {len(batch)} flow field slices.")
        return np.array(batch)

    @abstractmethod
    def load_file(self, file_path):
        """Loads a file and returns the dataset for caching.

        Args:
            file_path: str
                Path to the file to be loaded.

        Returns:
            np.ndarray: The loaded dataset.
        """
        pass

    @abstractmethod
    def get_next_slice(self):
        """Extracts the next slice from the cached data.

        Returns:
            np.ndarray: A 2D flow field slice.
        """
        pass

    @abstractmethod
    def get_flow_fields_shape(self):
        """Returns the shape of the flow field.

        Returns:
            tuple: Shape of the flow field.
        """
        pass
