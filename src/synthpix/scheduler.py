"""FlowFieldScheduler to load the flow field data from files."""
import os
import queue
import random
import threading
from abc import ABC, abstractmethod

import h5py
import numpy as np

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


class PrefetchingFlowFieldScheduler:
    """Prefetching Wrapper around a FlowFieldScheduler.

    It asynchronously prefetches batches of flow fields using a
    background thread to keep the GPU fed.
    """

    def __init__(self, scheduler, batch_size: int, buffer_size: int = 8):
        """Initializes the prefetching scheduler.

        Args:
            scheduler:
                The underlying flow field scheduler.
            batch_size: int
                Flow field slices per batch, must match the underlying scheduler.
            buffer_size: int
                Number of batches to prefetch.
        """
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self._queue = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)

        self._started = False

    def __iter__(self):
        """Returns the iterator instance itself and starts the background thread."""
        if not self._started:
            self._started = True
            self._thread.start()
        return self

    def __next__(self):
        """Returns the next batch of flow fields from the prefetch queue.

        Raises:
            StopIteration: If the queue is empty or no more data is available.
        """
        try:
            batch = self._queue.get(timeout=30.0)
            if batch is None:
                logger.info(
                    "[PrefetchingFlowFieldScheduler] No more data available. "
                    "Stopping."
                )
                raise StopIteration
            return batch
        except queue.Empty:
            logger.info(
                "[PrefetchingFlowFieldScheduler] Prefetch queue is empty. "
                "Waiting for data."
            )
            raise StopIteration

    def get_batch(self, batch_size):
        """Return the next batch from the prefetch queue, matching scheduler interface.

        Returns:
            np.ndarray: A preloaded batch of flow fields.
        """
        if batch_size != self.batch_size:
            raise ValueError(
                f"Batch size {batch_size} does not match the "
                f"prefetching batch size {self.batch_size}."
            )
        if not self._started:
            self.__iter__()
        return next(self)

    def get_flow_fields_shape(self):
        """Return the shape of the flow fields from the underlying scheduler.

        Returns:
            tuple: Shape of the flow fields as returned by the underlying scheduler.
        """
        return self.scheduler.get_flow_fields_shape()

    def _worker(self):
        """Background thread that continuously fetches batches from the scheduler."""
        # This will run until the stop event is set:
        while not self._stop_event.is_set():
            try:
                batch = self.scheduler.get_batch(self.batch_size)
            except StopIteration:
                # Intended behavior here: if I called get_batch() and ran into a
                # StopIteration, it means that I don't want the whole batch
                # i.e. I don't want offsize batches.
                # Signal end‑of‑stream to consumer
                self._queue.put(None, block=False)
                return

            # This will block until there is free space in the queue:
            # no busy‑waiting needed.
            try:
                self._queue.put(batch, block=True)
            except Exception as e:
                logger.warning(f"[Prefetching] Failed to put batch: {e}")
                return

    def reset(self):
        """Resets the prefetching scheduler and underlying scheduler."""
        # Stop the background thread and clear the queue
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()
        self._queue.queue.clear()

        # Reinitialize the scheduler and start the thread
        self._stop_event.clear()
        self._queue = queue.Queue(maxsize=self.buffer_size)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._started = False
        self.scheduler.reset()

    def shutdown(self, join_timeout=5.0):
        """Gracefully shuts down the background prefetching thread."""
        self._stop_event.set()

        # If producer is stuck on put(), free up one slot
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

        # If consumer is stuck on get(), inject the end-of-stream signal
        try:
            self._queue.put(None, block=False)
        except queue.Full:
            pass

        # Wait for the thread to finish
        if self._thread.is_alive():
            self._thread.join(timeout=join_timeout)

    def __del__(self):
        """Gracefully shuts down the scheduler upon deletion."""
        self.shutdown()
