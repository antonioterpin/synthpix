"""FloFlowFieldScheduler to load flow fields from Middlebury .flo files."""
import os

import numpy as np

from synthpix.scheduler import BaseFlowFieldScheduler
from synthpix.utils import logger


class FloFlowFieldScheduler(BaseFlowFieldScheduler):
    """Scheduler for loading flow field data from .flo files.

    Middlebury .flo format (https://github.com/shengzesnail/PIV_dataset):
      - 1xfloat32 TAG = 202021.25
      - 1xint32  width
      - 1xint32  height
      - widthxheightx2 float32  data in (u,v) order, row-major.
    """

    _file_pattern = "**/*.flo"
    TAG_FLOAT = 202021.25
    N_BANDS = 2

    def __init__(self, file_list, randomize=False, loop=False):
        """Initializes the .flo scheduler."""
        super().__init__(file_list, randomize, loop)
        if isinstance(file_list, str):
            # consider file list as a directory
            file_list = [os.path.join(file_list, f) for f in os.listdir(file_list)]
        not_flo = [fp for fp in file_list if not fp.lower().endswith(".flo")]
        if not_flo:
            raise ValueError(f"All files must be .flo flow files. Found: {not_flo}")

    def load_file(self, file_path: str) -> np.ndarray:
        """Reads a .flo file and returns a (H, W, 2) ndarray of flow (u,v)."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found.")
        with open(file_path, "rb") as f:
            # header
            tag = np.fromfile(f, dtype="<f4", count=1)[0]
            if tag != self.TAG_FLOAT:
                raise ValueError(
                    f"readFlowFile({file_path}): wrong tag (maybe big‐endian?)"
                )
            width = int(np.fromfile(f, dtype="<i4", count=1)[0])
            height = int(np.fromfile(f, dtype="<i4", count=1)[0])

            # sanity checks
            if width < 1 or width > 1e5 or height < 1 or height > 1e5:
                raise ValueError(
                    f"readFlowFile({file_path}): illegal dimensions {width}x{height}"
                )

            # read the flow data
            n_vals = width * height * self.N_BANDS
            data = np.fromfile(f, dtype="<f4", count=n_vals)
            if data.size != n_vals:
                raise IOError(f"readFlowFile({file_path}): unexpected EOF")
            data = data.reshape((height, width, self.N_BANDS))

        logger.debug(f"Loaded .flo file {file_path} with shape {data.shape}")
        return data

    def get_next_slice(self) -> np.ndarray:
        """Returns the next flow field. Since each .flo is one field, ignore slicing."""
        # BaseFlowFieldScheduler will have already called load_file and cached it
        return self._cached_data

    def get_flow_fields_shape(self) -> tuple:
        """Inspects the first file to return (H, W, 2)."""
        first = self.file_list[0]
        with open(first, "rb") as f:
            tag = np.fromfile(f, dtype="<f4", count=1)[0]
            if tag != self.TAG_FLOAT:
                raise ValueError("Bad .flo tag in first file.")
            width = int(np.fromfile(f, dtype="<i4", count=1)[0])
            height = int(np.fromfile(f, dtype="<i4", count=1)[0])
        shape = (height, width, self.N_BANDS)
        logger.debug(f".flo flow field shape: {shape}")
        return shape

    @staticmethod
    def from_config(config: dict) -> "FloFlowFieldScheduler":
        """Factory from config dict."""
        return FloFlowFieldScheduler(
            file_list=config["scheduler_files"],
            randomize=config.get("randomize", False),
            loop=config.get("loop", True),
        )
