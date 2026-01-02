"""NumpyDataSource implementation."""

import os
import re
import numpy as np
from PIL import Image
import logging
from typing import Any

from .base import FileDataSource

logger = logging.getLogger(__name__)


class NumpyDataSource(FileDataSource):
    """DataSource for loading .npy flow files and paired images."""

    # Only look for flow_*.npy files
    _file_pattern = "flow_*.npy"

    def __init__(
        self, dataset_path: list[str] | str, include_images: bool = False
    ) -> None:
        """Initializes the NumpyDataSource.

        Args:
            dataset_path: Path to the dataset.
            include_images: Whether to include images.
        """
        self._include_images = include_images
        super().__init__(dataset_path)

        # Pre-validation
        if not all(fp.endswith(".npy") for fp in self._file_list):
            raise ValueError("All files must be numpy files with '.npy' extension")

        if self.include_images:
            for flow_path in self._file_list:
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

    def load_file(self, file_path: str) -> dict[str, Any]:
        """Load .npy flow and optionally paired images.

        Args:
            file_path: Path to the .npy file.

        Returns:
            Dictionary containing data (e.g. 'flow_fields', 'images1', etc).
        """
        # Load Flow
        flow = np.load(file_path)

        data = {"flow_fields": flow, "file": file_path}

        # Load Images
        if self.include_images:
            mb = re.match(r"flow_(\d+)\.npy$", os.path.basename(file_path))
            if not mb:
                # Should be caught in init? But let's check safety.
                raise ValueError(f"Bad filename: {file_path}")
            t = int(mb.group(1))
            folder = os.path.dirname(file_path)

            prev_path = os.path.join(folder, f"img_{t-1}.jpg")
            curr_path = os.path.join(folder, f"img_{t}.jpg")

            # Load and convert to array
            prev = np.array(Image.open(prev_path).convert("RGB"))
            curr = np.array(Image.open(curr_path).convert("RGB"))

            data["images1"] = prev
            data["images2"] = curr

        return data

    @property
    def include_images(self) -> bool:
        """Whether to include images.

        Returns:
            bool: Whether to include images.
        """
        return self._include_images
