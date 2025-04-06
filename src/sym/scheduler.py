"""FlowFieldScheduler to load the flow field data from HDF5 files."""
import logging
import os
import random

import h5py
import numpy as np


class FlowFieldScheduler:
    """Iterator class that sequentially loads flow field data from a list of HDF5 files.

    It provides both iterative access and batch retrieval of flow fields,
    with options for preloading files into memory, randomizing the file order each epoch,
    and looping indefinitely over the dataset.
    """

    def __init__(
        self, file_list, randomize=False, loop=False, prefetch=True, DEBUG=False
    ):
        """Initializes the FlowFieldScheduler.

        Args:
            file_list (list): List of HDF5 file paths.
            randomize (bool): If True, randomize the order of files each epoch.
            loop (bool): If True, loop over the dataset indefinitely.
            prefetch (bool): If True, preload the entire current file into RAM
                             and keep it cached until switching to the next file.
                             If False, load only one slice at a time directly from disk.
            DEBUG (bool): If True, enable debug logging.
        """
        if not file_list:
            raise ValueError("The file_list must not be empty.")
        self.file_list = file_list
        self.randomize = randomize
        self.loop = loop
        self.prefetch = prefetch
        self.DEBUG = DEBUG

        # Validate file paths
        for file_path in self.file_list:
            if not isinstance(file_path, str):
                raise ValueError("All file paths must be strings.")
            if not file_path.endswith(".h5"):
                raise ValueError(f"File {file_path} is not an HDF5 file.")
            if not os.path.isfile(file_path):
                raise ValueError(f"File {file_path} does not exist.")

        # Argument validation
        if not isinstance(self.randomize, bool):
            raise ValueError("randomize must be a boolean value.")
        if not isinstance(self.loop, bool):
            raise ValueError("loop must be a boolean value.")
        if not isinstance(self.prefetch, bool):
            raise ValueError("prefetch must be a boolean value.")

        # Initialize state variables
        self.epoch = 0
        self.index = 0
        self.y_sel = 0

        self._cached_data = None
        self._cached_file = None

        if self.randomize:
            random.shuffle(self.file_list)
        logging.basicConfig(level=logging.INFO if not DEBUG else logging.DEBUG)

    def __len__(self):
        """Length of the file list.

        Returns:
            int: Number of files in the file list.
        """
        return len(self.file_list)

    def __iter__(self):
        """Return the iterator instance itself.

        This allows the FlowFieldScheduler instance to be used in for-loops
        and other iterable contexts, as it implements both __iter__ and __next__.

        Returns:
            FlowFieldScheduler: The iterator instance (self).
        """
        return self

    def __next__(self):
        """Returns the next flow field from the dataset.

        Raises:
            StopIteration: If the end of the dataset is reached and loop is False.
            Exception: If there is an error loading the file.

        Returns:
            flow_field (np.ndarray): Flow field data for the current y-slice.
        """
        while True:
            # Check if we need to reset the index for the next epoch
            if self.index >= len(self.file_list):
                if not self.loop:
                    raise StopIteration

                # Prepare for the next epoch
                self.index = 0
                self.y_sel = 0
                self.epoch += 1
                if self.randomize:
                    random.shuffle(self.file_list)
                logging.info(f"Starting epoch {self.epoch}")

            file_path = self.file_list[self.index]

            try:
                if self.prefetch:
                    # Load full file into memory once
                    if self._cached_file != file_path:
                        with h5py.File(file_path, "r") as file:
                            dataset_key = list(file)[0]
                            dset = file[dataset_key]
                            self._cached_data = file[dataset_key][
                                :, :, : dset.shape[2] // 2, :
                            ]  # full preload
                            # Known issue: We're not using the full dataset
                            # because the length step along the x axes is
                            # twice as much as the z axis. We need to fix this by changing
                            # the dataset structure in the first place.
                        self._cached_file = file_path
                        self.y_sel = 0

                    if self.y_sel >= self._cached_data.shape[1]:
                        self.index += 1
                        self.y_sel = 0
                        self._cached_file = None
                        self._cached_data = None
                        continue

                    data_slice = self._cached_data[:, self.y_sel, :, :]
                else:
                    # Load slice on demand
                    with h5py.File(file_path, "r") as file:
                        dataset_key = list(file)[0]
                        dset = file[dataset_key]

                        if self.y_sel >= dset.shape[1]:
                            self.index += 1
                            self.y_sel = 0
                            continue

                        data_slice = dset[:, self.y_sel, : dset.shape[2] // 2, :]
                        # Known issue: We're not using the full dataset, refer to
                        # the comment above for details.
                flow_field_x = data_slice[:, :, 0]
                flow_field_z = data_slice[:, :, 2]
                flow_field = np.stack([flow_field_x, flow_field_z], axis=2)

                logging.info(
                    f"Loaded y={self.y_sel} from {file_path}, shape {flow_field.shape}"
                )

                self.y_sel += 1
                return flow_field

            except Exception as e:
                logging.error(f"Error loading {file_path}: {e}")
                self.index += 1
                self._cached_data = None
                self._cached_file = None
                continue

    def get_batch(self, batch_size):
        """Retrieves a batch of flow fields.

        TODO: Implement and test this method to return a batch of flow fields
        in a more efficient way.

        Args:
            batch_size (int): Number of flow fields to load in the batch.

        Returns:
            list: A list of flow fields.
        """
        # Save current state
        current_prefetch = self.prefetch
        current_cached_file = self._cached_file
        current_data = self._cached_data
        current_epoch = self.epoch
        current_index = self.index
        current_y_sel = self.y_sel

        # Reset state for batch loading and force prefetch
        self.prefetch = True
        self._cached_file = None
        self._cached_data = None
        self.index = 0
        self.y_sel = 0
        self.epoch = 0
        if self.randomize:
            random.shuffle(self.file_list)
        batch = [next(self) for _ in range(batch_size)]

        # Restore state
        self.prefetch = current_prefetch
        self._cached_file = current_cached_file
        self._cached_data = current_data
        self.epoch = current_epoch
        self.index = current_index
        self.y_sel = current_y_sel

        return batch
