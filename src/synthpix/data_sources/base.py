"""Base FileDataSource abstract class."""

import glob
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import grain.python as grain

logger = logging.getLogger(__name__)

_HF_SCHEME = "hf://"


class FileDataSource(grain.RandomAccessDataSource, ABC):
    """Abstract base class for file-based datasources.

    Handles recursive file discovery from a list of paths or directories.
    """

    _file_pattern = "*"

    def __init__(self, dataset_path: list[str] | str) -> None:
        """Initializes the FileDataSource.

        Args:
            dataset_path: A single path or list of paths (files or directories).

        Raises:
            ValueError: If dataset_path is invalid or no files are found.
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
            raise ValueError(
                "dataset_path must be a list of file paths (or a string)."
            )

        for raw_path in dataset_path:
            if raw_path.startswith(_HF_SCHEME):
                resolved_path = self._resolve_hf_path(raw_path)
            else:
                resolved_path = raw_path
            if os.path.isdir(resolved_path):
                logger.debug(f"Searching for files in {resolved_path}")
                pattern = os.path.join(resolved_path, self._file_pattern)
                # Recursive glob search
                found_files = sorted(glob.glob(pattern, recursive=True))
                logger.debug(
                    f"Found {len(found_files)} files in {resolved_path}"
                )
                self._file_list.extend(found_files)
            else:
                self._file_list.append(resolved_path)

        if not self._file_list:
            raise ValueError(
                f"No files found in {dataset_path} matching pattern "
                f"{self._file_pattern}"
            )

        super().__init__()

    @staticmethod
    def _resolve_hf_path(spec: str) -> str:
        """Resolve an ``hf://`` URI to a local directory string.

        The resolver and ``huggingface_hub`` are both lazy-imported — the
        ``[hf]`` extra is not required for local-only datasets. Either
        import (the resolver module itself or ``huggingface_hub`` inside
        ``pull_dataset``) can fail with ``ImportError`` when the extra is
        missing, and both paths are surfaced with the same actionable
        message while preserving the original cause.

        Args:
            spec: The full ``hf://...`` URI.

        Returns:
            str: Absolute path to the resolved cache directory the
                surrounding glob logic can walk.

        Raises:
            ImportError: When either the resolver module or
                ``huggingface_hub`` cannot be imported. The original
                exception is attached as ``__cause__``.
        """
        try:
            # Lazy import: only loaded when an hf:// URI is seen,
            # so the [hf] extra is not required for local datasets.
            from synthpix.data_sources import (  # noqa: PLC0415
                hf_resolver,
            )

            resolved = hf_resolver.resolve_to_directory(spec)
        except ImportError as exc:
            raise ImportError(
                "hf:// paths require the [hf] extra. "
                "Install with: pip install synthpix[hf]"
            ) from exc
        logger.debug(f"Resolved hf:// URI {spec} to local dir {resolved}")
        return str(resolved)

    def __repr__(self) -> str:
        """Returns a stable string representation for Grain checkpointing.

        Returns:
            A string representation of the FileDataSource.
        """
        return f"{self.__class__.__name__}(file_list={self._file_list})"

    @property
    def file_list(self) -> list[str]:
        """Returns the list of files discovered."""
        return self._file_list

    @property
    def include_images(self) -> bool:
        """Returns True if the underlying data source provides images."""
        return False

    def __len__(self) -> int:
        """Returns the number of files in the dataset.

        Returns:
            The number of files in the dataset.
        """
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
