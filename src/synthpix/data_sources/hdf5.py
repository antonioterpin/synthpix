"""HDF5DataSource implementation."""

from typing import Any
import h5py
import logging

from .base import FileDataSource

logger = logging.getLogger(__name__)


class HDF5DataSource(FileDataSource):
    """DataSource for loading flow fields from .h5 files."""

    _file_pattern = "**/*.h5"

    def load_file(self, file_path: str) -> dict[str, Any]:
        """Loads the dataset from the HDF5 file.

        Args:
            file_path: Path to the .h5 file.

        Returns:
            Dictionary containing data (e.g. 'flow_fields', 'images1', etc).
        """
        data = None
        # NOTE: This implementation remains stateless and loads the full volume
        # from the HDF5 file. It does not yet guarantee the same performance
        # as the legacy HDF5FlowFieldScheduler which served individual slices.
        with h5py.File(file_path, "r") as file:
            dataset_key = list(file)[0]
            dset = file[dataset_key]
            if not isinstance(dset, h5py.Dataset):
                raise ValueError(
                    f"Expected Dataset but got {type(dset)} for key '{dataset_key}' in {file_path}"
                )
            data = dset[...]

        return {"flow_fields": data, "file": file_path}
