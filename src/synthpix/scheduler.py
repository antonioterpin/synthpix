"""FlowFieldScheduler to load the flow field data from files."""
import glob
import os
import random
import re
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
            pattern = os.path.join(file_list, self._file_pattern)
            file_list = sorted(glob.glob(pattern))
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

    _file_pattern = "*.h5"

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


class NumpyFlowFieldScheduler(BaseFlowFieldScheduler):
    """Scheduler for loading flow fields from .npy files, with optional image pairing.

    Each .npy file must be named 'flow_<t>.npy' and, if images are enabled,
    will be paired with 'img_<t-1>.jpg' and 'img_<t>.jpg' in the same folder.
    """

    # instruct base class to glob only the flow_*.npy files
    _file_pattern = "flow_*.npy"

    def __init__(
        self, file_list, randomize=False, loop=False, include_images: bool = False
    ):
        """Initializes the Numpy scheduler.

        This scheduler loads flow fields from .npy files and can optionally
        validate and return paired JPEG images.
        The .npy files must be named 'flow_<t>.npy' and the images must be
        named 'img_<t-1>.jpg' and 'img_<t>.jpg' in the same folder.

        Args:
            file_list: str or list of str
                A directory, single .npy file, or list of .npy paths.
            randomize: bool
                If True, shuffle file order per epoch.
            loop: bool
                If True, cycle indefinitely.
            include_images: bool
                If True, validate and return paired JPEG images.
        """
        self.include_images = include_images
        super().__init__(file_list, randomize, loop)

        # ensure all supplied files are .npy
        if not all(fp.endswith(".npy") for fp in self.file_list):
            raise ValueError("All files must be numpy files with '.npy' extension")

        # validate image pairs only if requested
        if self.include_images:
            for flow_path in self.file_list:
                mb = re.match(r"flow_(\d+)\.npy$", os.path.basename(flow_path))
                if not mb:
                    raise ValueError(f"Bad filename: {flow_path}")
                t = int(mb.group(1))
                folder = os.path.dirname(flow_path)
                prev_img = os.path.join(folder, f"img_{t-1}.jpg")
                next_img = os.path.join(folder, f"img_{t}.jpg")
                if not (os.path.isfile(prev_img) and os.path.isfile(next_img)):
                    raise FileNotFoundError(
                        f"Missing images for frame {t}: {prev_img}, {next_img}"
                    )

    def load_file(self, file_path: str):
        """Load the raw flow array from .npy."""
        return np.load(file_path)

    def get_next_slice(self):
        """Return either the flow array or, if enabled, flow plus images."""
        flow = self._cached_data
        if not self.include_images:
            return flow

        # load images on-demand
        mb = re.match(r"flow_(\d+)\.npy$", os.path.basename(self._cached_file))
        t = int(mb.group(1))
        folder = os.path.dirname(self._cached_file)
        prev = np.array(
            Image.open(os.path.join(folder, f"img_{t-1}.jpg")).convert("RGB")
        )
        nxt = np.array(Image.open(os.path.join(folder, f"img_{t}.jpg")).convert("RGB"))
        return {"flow": flow, "img_prev": prev, "img_next": nxt}

    def get_flow_fields_shape(self):
        """Return the shape of a single flow array."""
        return np.load(self.file_list[0]).shape

    def __next__(self):
        """Iterate over .npy files, returning flow or flow+images."""
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
