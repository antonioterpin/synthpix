"""HDF5FlowFieldScheduler to load flow fields from .h5 files."""
import h5py
import jax
import numpy as np
from goggles import get_logger

from ..scheduler import BaseFlowFieldScheduler

logger = get_logger(__name__)


class HDF5FlowFieldScheduler(BaseFlowFieldScheduler):
    """Scheduler for loading flow field data from HDF5 files.

    Assumes each file contains a single dataset with shape (X, Y, Z, C),
    and extracts the x and z components (0 and 2) from each y-slice.
    """

    _file_pattern = "**/*.h5"

    def __init__(
        self,
        file_list: list,
        randomize: bool = False,
        loop: bool = False,
        key: jax.random.PRNGKey = None,
    ):
        """Initializes the HDF5 scheduler.

        Args:
            file_list (list): A directory, single .h5 file, or list of .h5 paths.
            randomize (bool): If True, shuffle file order per epoch.
            loop (bool): If True, cycle indefinitely.
            key (jax.random.PRNGKey): Random key for reproducibility.
        """
        super().__init__(file_list, randomize, loop, key)
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

    @classmethod
    def from_config(cls, config: dict) -> "HDF5FlowFieldScheduler":
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
