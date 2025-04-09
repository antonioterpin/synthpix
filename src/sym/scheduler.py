import logging
import os
import random
from abc import ABC, abstractmethod

import h5py
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseFlowFieldScheduler(ABC):
    """Abstract base class for scheduling access to flow field data from various file formats.

    This class provides iteration, looping, caching, and batch loading capabilities.
    Subclasses must implement file-specific loading and y-slice extraction logic.
    """

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
            f"BaseFlowFieldScheduler initialized with {len(self.file_list)} files, "
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
        logger.info("Scheduler state has been reset.")

    def __next__(self):
        """Returns the next flow field slice from the dataset.

        Returns:
            np.ndarray: A single flow field slice.

        Raises:
            StopIteration: If no more data and loop is False.
        """
        while True:
            if self.index >= len(self.file_list):
                if not self.loop:
                    raise StopIteration
                self.reset(reset_epoch=False)
                logger.info(f"Starting epoch {self.epoch}")

            file_path = self.file_list[self.index]

            try:
                if self._cached_file != file_path:
                    self._cached_data = self.load_file(file_path)
                    self._cached_file = file_path
                    self._slice_idx = 0
                    logger.info(
                        f"Prefetched data from {file_path}, shape {self._cached_data.shape}"
                    )

                if self._slice_idx >= self._cached_data.shape[1]:
                    self.index += 1
                    self._cached_file = None
                    self._cached_data = None
                    continue

                flow_field = self.get_next_slice()
                logger.debug(
                    f"Loaded slice y={self._slice_idx} from {file_path}, shape {flow_field.shape}"
                )
                self._slice_idx += 1
                return flow_field

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                self.index += 1
                self._cached_data = None
                self._cached_file = None
                continue

    def get_batch(self, batch_size):
        """Retrieves a batch of flow fields.

        Args:
            batch_size: int
                Number of flow field slices to retrieve.

        Returns:
            list: A list of flow field slices.
        """
        current_cached_file = self._cached_file
        current_data = self._cached_data
        current_epoch = self.epoch
        current_index = self.index
        current_slice_idx = self._slice_idx

        self._cached_file = None
        self._cached_data = None
        self.index = 0
        self._slice_idx = 0
        self.epoch = 0
        if self.randomize:
            random.shuffle(self.file_list)

        batch = [next(self) for _ in range(batch_size)]

        self._cached_file = current_cached_file
        self._cached_data = current_data
        self.epoch = current_epoch
        self.index = current_index
        self._slice_idx = current_slice_idx

        logger.debug(f"Loaded batch of {batch_size} flow field slices.")
        return batch

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


class HDF5FlowFieldScheduler(BaseFlowFieldScheduler):
    """Scheduler for loading flow field data from HDF5 files.

    Assumes each file contains a single dataset with shape (T, Y, Z, C),
    and extracts the x and z components (0 and 2) from each y-slice.
    """
    def __init__(self, file_list, randomize=False, loop=False):
        super().__init__(file_list, randomize, loop)
        if not all(file_path.endswith(".h5") for file_path in file_list):
            raise ValueError("All files must be HDF5 files with .h5 extension.")
        
    def load_file(self, file_path):
        """Loads the dataset from the HDF5 file.

        Args:
            file_path: str
                Path to the HDF5 file.

        Returns:
            np.ndarray: Loaded dataset with truncated x-axis.
        """
        with h5py.File(file_path, "r") as file:
            dataset_key = list(file)[0]
            dset = file[dataset_key]
            data = dset[:, :, : dset.shape[2] // 2, :]
            # Known issue: We're not using the full dataset
            # because the length step along the x axes is
            # twice as much as the z axis. We need to fix this by changing
            # the dataset structure in the first place.
        return data

    def get_next_slice(self):
        """Retrieves a flow field slice (x and z components) for the current y index.

        Returns:
            np.ndarray: Flow field with shape (T, Z, 2).
        """
        data_slice = self._cached_data[:, self._slice_idx, :, :]
        flow_field_x = data_slice[:, :, 0]
        flow_field_z = data_slice[:, :, 2]
        return np.stack([flow_field_x, flow_field_z], axis=2)
