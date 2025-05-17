"""FlowFieldScheduler to load the flow field data from files."""
import glob
import os
import queue
import random
import re
import threading
from abc import ABC, abstractmethod

import cv2
import h5py
import numpy as np
import scipy.io
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
            logger.debug(f"[Base] Searching for files in {file_list}")
            pattern = os.path.join(file_list, self._file_pattern)
            file_list = sorted(glob.glob(pattern, recursive=True))
            logger.debug(f"[Base] Found {len(file_list)} files in {file_list}")
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
            f"[Base] initialized with {len(self.file_list)} files, "
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
        logger.info("[Base] Scheduler state has been reset.")
        logger.debug("[Base] File list: %s", self.file_list)

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
                    f"[Base] Loaded slice y={self._slice_idx} from {file_path}, "
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
                    f"[Base] Only {len(batch)} slices could be loaded before exhaustion."
                )
                return np.array(batch)
            raise

        logger.debug(f"[Base] Loaded batch of {len(batch)} flow field slices.")
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

    _file_pattern = "**/*.h5"

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

    @staticmethod
    def from_config(config: dict) -> "HDF5FlowFieldScheduler":
        """Creates a HDF5FlowFieldScheduler instance from a configuration dictionary.

        Args:
            config: dict
                Configuration dictionary containing the scheduler parameters.

        Returns:
            HDF5FlowFieldScheduler: An instance of the scheduler.
        """
        return HDF5FlowFieldScheduler(
            file_list=config["scheduler_files"],
            randomize=config.get("randomize", False),
            loop=config.get("loop", True),
        )


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

    @staticmethod
    def from_config(config: dict) -> "NumpyFlowFieldScheduler":
        """Creates a NumpyFlowFieldScheduler instance from a configuration dictionary.

        Args:
            config: dict
                Configuration dictionary containing the scheduler parameters.

        Returns:
            NumpyFlowFieldScheduler: An instance of the scheduler.
        """
        return NumpyFlowFieldScheduler(
            file_list=config["scheduler_files"],
            randomize=config.get("randomize", False),
            loop=config.get("loop", True),
            include_images=config.get("include_images", False),
        )


class MATFlowFieldScheduler(BaseFlowFieldScheduler):
    """Scheduler for loading flow fields from .mat files.

    Assumes each file contains a dataset with three keys:
    I0: previous image, I1: current image, V: associated flow field.

    The scheduler can extract the flow field data and return it as a numpy array,
    but can also return the images if requested.

    Notice that the flow field data is expected to be already in pixels,
    and the images are in the same resolution as the flow fields.
    The size of the flow fields in the dataset varies is either 256x256, 512x512,
    or 1024x1024, and the images are in the same resolution as the flow fields.
    The scheduler will downscale all the data to 256x256.
    """

    _file_pattern = "**/*.mat"

    def __init__(
        self,
        file_list,
        randomize=False,
        loop=False,
        include_images: bool = False,
        output_shape=(256, 256),
    ):
        """Initializes the MATFlowFieldScheduler."""
        if not isinstance(include_images, bool):
            raise ValueError("include_images must be a boolean value.")
        self.include_images = include_images

        if not isinstance(output_shape, tuple) or len(output_shape) != 2:
            raise ValueError("output_shape must be a tuple of two integers.")
        if not all(isinstance(dim, int) and dim > 0 for dim in output_shape):
            raise ValueError("output_shape must contain positive integers.")
        self.output_shape = output_shape

        super().__init__(file_list, randomize, loop)
        # ensure all supplied files are .mat
        if not all(file_path.endswith(".mat") for file_path in self.file_list):
            raise ValueError("All files must be MATLAB .mat files with HDF5 format")

        logger.debug("[MAT] Initialized with %s files", len(self.file_list))
        logger.debug("[MAT] File list: %s", self.file_list)

    @staticmethod
    def _looks_like_hdf5(path: str) -> bool:
        try:
            return h5py.is_hdf5(path)
        except OSError:
            return False

    def load_file(self, file_path: str):
        """Load any MATLAB .mat file (v4, v5/6/7, or v7.3) and return its data dict.

        Parameters
        ----------
        file_path : str
            Path to the .mat file.

        Returns
        -------
        dict
            Dictionary containing at least 'V' (flow field).  When
            `self.include_images` is True, it must also hold 'I0' and 'I1'.
        """

        def recursively_load_hdf5_group(group, prefix=""):
            """Flatten all datasets in an HDF5 tree into a dict keyed by full path."""
            out = {}
            for name, item in group.items():
                path = f"{prefix}/{name}" if prefix else name
                if isinstance(item, h5py.Dataset):
                    out[path] = item[()]
                elif isinstance(item, h5py.Group):
                    out.update(recursively_load_hdf5_group(item, path))
            return out

        # Guarantee data is always defined
        data = None

        # First try SciPy (handles MATLAB v4-v7.2)
        try:
            mat = scipy.io.loadmat(
                file_path,
                struct_as_record=False,
                squeeze_me=True,
            )  # SciPy raises NotImplementedError for v7.3
            data = {k: v for k, v in mat.items() if not k.startswith("__")}
        except (NotImplementedError, ValueError):
            if self._looks_like_hdf5(file_path):
                # MATLAB v7.3 ⇒ fall back to h5py
                try:
                    with h5py.File(file_path, "r") as f:
                        data = recursively_load_hdf5_group(f)
                except Exception as e:
                    raise ValueError(
                        f"Failed to load {file_path} as HDF5 or legacy MATLAB: {e}"
                    ) from e

        if data is None:
            raise ValueError(f"Failed to load {file_path} as HDF5 or legacy MATLAB.")

        # Validate the loaded data
        if "V" not in data:
            raise ValueError(f"Flow field not found in {file_path} (missing 'V').")
        if self.include_images and not all(k in data for k in ("I0", "I1")):
            raise ValueError(
                f"Image visualization not supported for {file_path}: "
                "missing required keys 'I0'/'I1'."
            )

        # Resizing images and flow to output_shape
        if self.include_images:
            for key in ("I0", "I1"):
                img = data[key]
                if img.shape[:2] != self.output_shape:
                    data[key] = np.asarray(
                        Image.fromarray(img).resize(self.output_shape)
                    )

        flow = data["V"]
        if flow.shape[2] != 2:
            if flow.shape[0] == 2:
                flow = np.transpose(flow, (1, 2, 0))
            elif flow.shape[1] == 2:
                flow = np.transpose(flow, (0, 2, 1))
            data["V"] = flow
        if flow.shape[:2] != self.output_shape:
            # Resize flow to output_shape and scale by the ratio
            # The original flow is assumed to be in pixels
            ratio_y = self.output_shape[0] / flow.shape[0]
            ratio_x = self.output_shape[1] / flow.shape[1]
            flow_resized = cv2.resize(
                flow, self.output_shape, interpolation=cv2.INTER_LINEAR
            )
            flow_resized[..., 0] *= ratio_x
            flow_resized[..., 1] *= ratio_y
            data["V"] = flow_resized

        logger.debug("[MAT] Loaded %s with keys %s", file_path, list(data.keys()))
        return data

    def get_next_slice(self):
        """Retrieves the flow field slice and optionally the images.

        Returns:
            np.ndarray or dict: Flow field with shape (X, Y, Z, 2) or a dict
            containing the flow field and images.
        """
        data = self._cached_data
        flow_field = data["V"]

        if not self.include_images:
            return flow_field

        img_prev = data["I0"]
        img_next = data["I1"]
        return {"flow": flow_field, "img_prev": img_prev, "img_next": img_next}

    def get_flow_fields_shape(self):
        """Returns the shape of the flow field.

        Returns:
            tuple: Shape of the flow field.
        """
        return self.output_shape

    def __next__(self):
        """Iterate over .mat files, returning flow or flow+images."""
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

    @staticmethod
    def from_config(config: dict) -> "MATFlowFieldScheduler":
        """Creates a MATFlowFieldScheduler instance from a configuration dictionary.

        Args:
            config: dict
                Configuration dictionary containing the scheduler parameters.

        Returns:
            MATFlowFieldScheduler: An instance of the scheduler.
        """
        return MATFlowFieldScheduler(
            file_list=config["scheduler_files"],
            randomize=config.get("randomize", False),
            loop=config.get("loop", True),
            include_images=config.get("include_images", False),
            output_shape=tuple(config.get("output_shape", (256, 256))),
        )


class PrefetchingFlowFieldScheduler:
    """Prefetching Wrapper around a FlowFieldScheduler.

    It asynchronously prefetches batches of flow fields using a
    background thread to keep the GPU fed.
    """

    def __init__(self, scheduler, batch_size: int, buffer_size: int = 8):
        """Initializes the prefetching scheduler.

        If the underlying scheduler is episodic, it will recognize it and handle
        moving to the next episode seamlessly.

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

        if hasattr(self.scheduler, "episode_length"):
            self.episode_length = getattr(self.scheduler, "episode_length")

        self._queue = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)

        self._t_in_episode = 0
        self._started = False

    def __iter__(self):
        """Returns the iterator instance itself and starts the background thread."""
        if not self._started:
            self._started = True
            self._thread.start()
            logger.debug("[Prefetching] Background thread started.")
        return self

    def __next__(self):
        """Returns the next batch of flow fields from the prefetch queue.

        Raises:
            StopIteration: If the queue is empty or no more data is available.
        """
        try:
            batch = self._queue.get(timeout=5.0)
            if hasattr(self.scheduler, "episode_length"):
                if self._t_in_episode >= self.episode_length:
                    self._t_in_episode = 0
                self._t_in_episode += 1

            if batch is None:
                logger.info("[Prefetching] No more data available. " "Stopping.")
                raise StopIteration
            logger.debug(f"[Prefetching] Fetched batch of shape {batch.shape}, ")
            if hasattr(self.scheduler, "episode_length"):
                logger.debug(f"[Prefetching] t_in_episode = {self._t_in_episode}")
            return batch
        except queue.Empty:
            logger.info("[Prefetching] Prefetch queue is empty. " "Waiting for data.")
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

        logger.debug("[Prefetching] Prefetching thread stopped, " "queue cleared.")

        # Reinitialize the scheduler and start the thread
        self._stop_event.clear()
        self._queue = queue.Queue(maxsize=self.buffer_size)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._started = False
        self.scheduler.reset()

        logger.debug(
            "[Prefetching] Prefetching thread reinitialized, " "scheduler reset."
        )

    def next_episode(self, join_timeout=5.0):
        """Flush the remaining items of the current episode and restart.

        This method removes the remaining items from the current episode from
        the queue but does not inform the underlying scheduler.
        This implementation works as far as the buffer size is larger than
        the episode length.

        Args:
            join_timeout: float
                Timeout for the thread to join.
        """
        if not self._started:
            self.__iter__()
        # halt producer thread
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=join_timeout)

        # drain the part of the queue that belongs to the unfinished episode
        # discard up to `n_left` batches (may be fewer if queue < n_left)
        for _ in range(self.steps_remaining()):
            logger.debug(
                "[Prefetching] Moving to next episode, "
                f"of length {self.episode_length}, "
                f"while {self._t_in_episode} steps have been taken so far"
            )
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        logger.debug(
            f"[Prefetching] Flushed {self.steps_remaining()} batches from queue."
        )

        self._t_in_episode = 0
        # restart producer thread
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        logger.debug(
            "[Prefetching] Next episode started, "
            "remaining items flushed from the queue."
        )

    def steps_remaining(self) -> int:
        """Return the number of steps remaining in the current episode.

        Returns:
            int: Number of steps remaining in the current episode.
        """
        return self.episode_length - self._t_in_episode

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
    >>> epi  = EpisodicFlowFieldScheduler(
    ...             scheduler=base,
    ...             batch_size=16,
    ...             episode_length=32,
    ...             seed=42)
    >>> batch_t0 = next(epi)        # first time-step from 16 episodes
    >>> batch_t1 = next(epi)        # second time-step
    >>> epi.reset()                 # start 16 fresh episodes

    Notes
    -----
    * The underlying scheduler (and its prefetching thread) are **not** rebuilt
      on every episode reset—only the order of ``file_list`` is mutated.  This
      keeps disk I/O sequential and maximises throughput.
    * The wrapper follows the “vector-environment” pattern popular in Gym,
      Gymnax and Brax, so your JAX RL loop can `vmap` or `pmap` over the first
      dimension without shape changes.
    """

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        scheduler: BaseFlowFieldScheduler,
        batch_size: int,
        episode_length: int,
        num_episodes: int = 100,
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

        if num_episodes <= 0:
            raise ValueError("num_episodes must be positive")

        self.scheduler = scheduler
        self.batch_size = batch_size
        self.episode_length = episode_length
        self.num_episodes = num_episodes + 1  # + 1 to account for initial reset

        self._rng = random.Random(seed)
        self._t_in_episode = 0
        self._episode_count = 0

        # Populate the underlying scheduler with an initial file ordering
        self._rebuild_file_order()

    def __iter__(self) -> "EpisodicFlowFieldScheduler":
        """Returns self so the object can be used in a ``for`` loop."""
        self._t_in_episode = 0
        return self

    def __next__(self):
        """Return one time-step of shape ``(batch_size, …)``.

        Returns:
            batch: A `(batch_size, …)` tensor that holds one flow-field per episode.
        """
        # If we’ve exhausted the current horizon, start fresh episodes
        if self._t_in_episode >= self.episode_length:
            self._episode_count += 1
            if self._episode_count >= self.num_episodes:
                logger.info(
                    "[Episodic] All %d episodes have been served. " "Stopping.",
                    self.num_episodes,
                )
                raise StopIteration
            self._rebuild_file_order()
            self._t_in_episode = 0

        self._t_in_episode += 1

        batch = self.scheduler.get_batch(self.batch_size)

        logger.debug(
            "[Episodic] __next__() called, " "returning batch of shape %s",
            batch.shape,
        )
        logger.debug(f"[Episodic] timestep: {self._t_in_episode}")
        return batch

    def get_batch(self, batch_size: int):
        """Return exactly one time-step for `batch_size` parallel episodes.

        *Does not* loop internally – we delegate to the wrapped base
        scheduler once, because `__next__` already returns a full batch.
        """
        if batch_size != self.batch_size:
            raise ValueError(
                f"[Episodic] Requested batch_size {batch_size}, "
                f"but EpisodicFlowFieldScheduler was initialized with "
                f"{self.batch_size}"
            )
        logger.debug(
            "[Episodic] get_batch() called with batch_size %d",
            batch_size,
        )
        return next(self)

    def __len__(self) -> int:
        """Return the episode length.

        Returns:
            int: The length of the episode.
        """
        return self.episode_length

    def reset_episode(self):
        """Start *batch_size* brand-new episodes.

        The call is cheap: it only reshuffles ``file_list`` and resets cursors;
        the prefetch thread of the underlying scheduler is left intact.
        """
        self._rebuild_file_order()
        self._t_in_episode = 0

    def steps_remaining(self) -> int:
        """Return the number of steps remaining in the current episode.

        Returns:
            int: Number of steps remaining in the current episode.
        """
        return self.episode_length - self._t_in_episode

    def get_flow_fields_shape(self):
        """Return the shape of the flow fields from the underlying scheduler.

        Returns:
            tuple: Shape of the flow fields as returned by the underlying scheduler.
        """
        return self.scheduler.get_flow_fields_shape()

    # ------------------------------------------------------------------ #
    # Internal helpers (private)                                         #
    # ------------------------------------------------------------------ #
    def _rebuild_file_order(self):
        """Create a new interleaved file order and push it into ``scheduler``."""
        leaf_dirs = self._discover_leaf_dirs(self.scheduler.file_list)
        dir2files = {d: sorted(glob.glob(os.path.join(d, "*.mat"))) for d in leaf_dirs}

        # Sanity-check: all directories must contain enough frames
        for d, files in dir2files.items():
            if len(files) < self.episode_length:
                raise ValueError(
                    f"Directory {d} has only {len(files)} files, "
                    f"but episode_length is {self.episode_length}."
                )

        # Enumerate every admissible (dir, start_index) combination
        candidates = []
        for d, files in dir2files.items():
            last_start = len(files) - self.episode_length
            candidates.extend((d, s) for s in range(last_start + 1))

        if len(candidates) < self.batch_size:
            raise ValueError(
                f"Only {len(candidates)} distinct start positions available—"
                f"need at least batch_size = {self.batch_size}."
            )

        # Randomly choose batch_size starts without replacement
        starts = self._rng.sample(candidates, self.batch_size)

        # Build individual episode sequences
        episodes = []
        for d, s in starts:
            episodes.append(dir2files[d][s : s + self.episode_length])

        # Interleave “time major” → t0_ep0, t0_ep1, …, t1_ep0, …
        interleaved = [
            episodes[ep][t]  # type: ignore[index]
            for t in range(self.episode_length)
            for ep in range(self.batch_size)
        ]

        logger.debug(
            "[Episodic] Order rebuilt — %d episodes × %d steps = %d files",
            self.batch_size,
            self.episode_length,
            len(interleaved),
        )

        # Inject new order and reset cursors *without* reshuffling internally
        self.scheduler.file_list = interleaved
        self.scheduler.reset(reset_epoch=False)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _discover_leaf_dirs(paths: list[str]) -> list[str]:
        """Return every directory that contains data but *no* sub-directories.

        Parameters
        ----------
        paths
            Any list of file paths (e.g. ``scheduler.file_list``).

        Returns
        -------
        list[str]
            Deduplicated list of leaf directory paths.
        """
        leaf_dirs = set()
        for path in paths:
            dir_path = os.path.dirname(path)
            # A leaf dir has *no* child directories
            has_subdirs = any(
                os.path.isdir(os.path.join(dir_path, entry))
                for entry in os.listdir(dir_path)
            )
            if not has_subdirs:
                leaf_dirs.add(dir_path)
        return list(leaf_dirs)
