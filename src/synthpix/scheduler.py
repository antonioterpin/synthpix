"""FlowFieldScheduler to load the flow field data from files."""
import os
import random
from abc import ABC, abstractmethod

import h5py
import numpy as np
from PIL import Image

from synthpix.utils import logger


class BaseFlowFieldScheduler(ABC):
    """Abstract class for scheduling access to flow field data from various file formats.

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
                logger.info(f"Starting epoch {self.epoch}")

            file_path = self.file_list[self.index]

            try:
                if self._cached_file != file_path:
                    self._cached_data = self.load_file(file_path)
                    self._cached_file = file_path
                    self._slice_idx = 0
                    logger.info(
                        f"Prefetched data from {file_path}, "
                        f"shape {self._cached_data.shape}"
                    )

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

    def get_batch(self, batch_size) -> list:
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


class HDF5FlowFieldScheduler(BaseFlowFieldScheduler):
    """Scheduler for loading flow field data from HDF5 files.

    Assumes each file contains a single dataset with shape (X, Y, Z, C),
    and extracts the x and z components (0 and 2) from each y-slice.
    """

    def __init__(self, file_list, randomize=False, loop=False):
        """Initializes the HDF5 scheduler."""
        super().__init__(file_list, randomize, loop)
        if not all(file_path.endswith(".h5") for file_path in file_list):
            raise ValueError("All files must be HDF5 files with .h5 extension.")

    def load_file(self, file_path) -> np.ndarray:
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
            data = dset[...]
            logger.debug(f"Loading file {file_path} with shape {data.shape}")
        return data

    def get_next_slice(self) -> np.ndarray:
        """Retrieves a flow field slice (x and z components) for the current y index.

        Returns:
            np.ndarray: Flow field with shape (X, Z, 2).
        """
        data_slice = self._cached_data[:, self._slice_idx, :, :]

        return data_slice

    def get_flow_fields_shape(self) -> tuple:
        """Returns the shape of all the flow fields.

        It is assumed that all the flow fields have the same shape.

        Returns:
            tuple: Shape of all the flow fields.
        """
        file_path = self.file_list[0]
        with h5py.File(file_path, "r") as file:
            dataset_key = list(file)[0]
            dset = file[dataset_key]
            shape = dset.shape[0], dset.shape[2], 2  # (X, Z, 2)
            logger.debug(f"Flow field shape: {shape}")
        return shape


class PIVLabFlowFieldScheduler(BaseFlowFieldScheduler):
    """Scheduler for loading flow fields from .npy files and .jpg image pairs.

    Each .npy file is expected to be named as 'flow_t.npy' and is associated with:
        - 'img_{t-1}.jpg'
        - 'img_{t}.jpg'
    """

    def __init__(self, file_list, randomize=False, loop=False):
        """Initializes the PIVLab scheduler."""
        super().__init__(file_list, randomize, loop)
        if not all(file_path.endswith((".npy", ".jpg")) for file_path in file_list):
            raise ValueError(
                "All files must be either '.npy' (flow) or '.jpg' (image) files."
            )

    def load_file(self, file_path: str):
        """Load the flow field from a .npy file and locate the associated image files.

        Returns:
            A tuple (flow_array, img_prev_path, img_next_path)
        """
        # Extract frame index t from the filename: assumes format like flow_1.npy
        filename = os.path.basename(file_path)
        t_str = filename.replace("flow_", "").replace(".npy", "")
        try:
            t = int(t_str)
        except ValueError:
            raise ValueError(
                f"Filename {filename} does not match expected 'flow_<int>.npy' pattern."
            )

        # Build associated image paths
        folder = os.path.dirname(file_path)
        img_prev_path = os.path.join(folder, f"img_{t - 1}.jpg")
        img_next_path = os.path.join(folder, f"img_{t}.jpg")

        if not (os.path.isfile(img_prev_path) and os.path.isfile(img_next_path)):
            raise FileNotFoundError(
                f"Missing images for flow_{t}: {img_prev_path}, {img_next_path}"
            )

        # Load .npy flow file
        flow = np.load(file_path)

        # Store all 3 in a dict for get_next_slice to consume
        return (flow, img_prev_path, img_next_path)

    def get_next_slice(self):
        """Retrieve the next flow field slice and associated images.

        Returns:
            dict: A dictionary containing the flow field and associated images.
        """
        flow, img_prev_path, img_next_path = self._cached_data

        img_prev = np.array(Image.open(img_prev_path).convert("RGB"))
        img_next = np.array(Image.open(img_next_path).convert("RGB"))

        return {"flow": flow, "img_prev": img_prev, "img_next": img_next}

    def get_flow_fields_shape(self):
        """Returns the shape of all the flow fields.

        It is assumed that all the flow fields have the same shape.

        Returns:
            tuple: Shape of all the flow fields.
        """
        file_path = self.file_list[0]
        flow = np.load(file_path)
        return flow.shape

    def __next__(self):
        """Returns the next flow field slice from the dataset."""
        while self.index < len(self.file_list) or self.loop:
            if self.index >= len(self.file_list):
                self.reset(reset_epoch=False)

            file_path = self.file_list[self.index]

            try:
                if self._cached_file != file_path:
                    self._cached_data = self.load_file(file_path)
                    self._cached_file = file_path
                    self._slice_idx = 0

                sample = self.get_next_slice()

                # Advance to next file
                self.index += 1
                self._cached_file = None
                self._cached_data = None
                self._slice_idx = 0

                return sample

            except Exception as e:
                print(f"[ERROR] Skipping file {file_path}: {e}")
                self.index += 1
                self._cached_file = None
                self._cached_data = None
                continue

        raise StopIteration
