"""Base FileDataSource abstract class."""

import glob
import os
from abc import ABC, abstractmethod
import logging
from typing import Any

import grain.python as grain

logger = logging.getLogger(__name__)

class FileDataSource(grain.RandomAccessDataSource, ABC):
    """Abstract base class for file-based datasources.
    
    Handles recursive file discovery from a list of paths or directories.
    """

    _file_pattern = "*"

    def __init__(self, dataset_path: list[str] | str) -> None:
        """Initializes the FileDataSource.

        Args:
            dataset_path: A single path or list of paths (files or directories).
        """
        self._file_list = []
        
        # Normalize input to list
        if isinstance(dataset_path, str):
            dataset_path = [dataset_path]
            
        if (
            dataset_path is None
            or not isinstance(dataset_path, list)
            or not all(isinstance(f, str) for f in dataset_path)
        ):
            raise ValueError("dataset_path must be a list of file paths (or a single string).")

        for file_path in dataset_path:
            if os.path.isdir(file_path):
                logger.debug(f"Searching for files in {file_path}")
                pattern = os.path.join(file_path, self._file_pattern)
                # Recursive glob search
                found_files = sorted(glob.glob(pattern, recursive=True))
                logger.debug(f"Found {len(found_files)} files in {file_path}")
                self._file_list.extend(found_files)
            else:
                self._file_list.append(file_path)

        if not self._file_list:
            raise ValueError(f"No files found in {dataset_path} matching pattern {self._file_pattern}")
        
        super().__init__()

    @property
    def file_list(self) -> list[str]:
        """Returns the list of files discovered."""
        return self._file_list

    @property
    def include_images(self) -> bool:
        """Returns True if the underlying data source provides images."""
        return False

    def __len__(self) -> int:
        """Returns the number of files in the dataset."""
        return len(self._file_list)

    def __getitem__(self, record_key: int) -> dict[str, Any]:
        """Loads a file and returns the data dictionary.
        
        Args:
            record_key: Index of the file to load.
            
        Returns:
            Dictionary containing data (e.g. 'flow_fields', 'images1', etc).
        """
        return self.load_file(self._file_list[record_key])

    @abstractmethod
    def load_file(self, file_path: str) -> dict[str, Any]:
        """Loads a file and returns the data dictionary.
        
        Args:
            file_path: Absolute path to the file.
            
        Returns:
            Dictionary containing data (e.g. 'flow_fields', 'images1', etc).
        """
        pass # pragma: no cover
