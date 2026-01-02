"""FlowFieldScheduler to load the flow field data from files."""

import h5py
import numpy as np
import scipy.io
from goggles import get_logger
from PIL import Image
from typing_extensions import Self

from synthpix.scheduler.protocol import FileEndedError
from synthpix.types import PRNGKey, SchedulerData
from synthpix.utils import SYNTHPIX_SCOPE

from .base import BaseFlowFieldScheduler

logger = get_logger(__name__, scope=SYNTHPIX_SCOPE)


class MATFlowFieldScheduler(BaseFlowFieldScheduler):
    """Scheduler for loading flow fields from .mat files.

    Assumes each file contains a dataset with three keys:
    I0: previous image, I1: current image, V: associated flow field.

    The scheduler can extract the flow field data and return it as a numpy
    array, but can also return the images if requested.

    Notice that the flow field data is expected to be already in pixels,
    and the images are in the same resolution as the flow fields.
    The size of the flow fields in the dataset varies is either 256x256,
    512x512, or 1024x1024, and the images are in the same resolution as the
    flow fields.
    The scheduler will downscale all the data to 256x256.
    """

    _file_pattern = "**/*.mat"

    def __init__(
        self,
        file_list: list[str],
        randomize: bool = False,
        loop: bool = False,
        include_images: bool = False,
        output_shape: tuple[int, int] = (256, 256),
        key: PRNGKey | None = None,
    ):
        """Initializes the MATFlowFieldScheduler.

        Args:
            file_list: A list of (directories containing) .mat paths.
            randomize: If True, shuffle file order each reset.
            loop: If True, cycle indefinitely by wrapping around.
            include_images: If True, return a tuple (I0, I1, V).
            output_shape: The desired output shape for the flow fields.
                Must be a tuple of two integers (height, width).
            key: Random key for reproducibility.

        Raises:
            ValueError: If include_images is not a boolean or output_shape
                is invalid.
        """
        if not isinstance(include_images, bool):
            raise ValueError("include_images must be a boolean value.")
        self.include_images = include_images

        if not isinstance(output_shape, tuple) or len(output_shape) != 2:
            raise ValueError("output_shape must be a tuple of two integers.")
        if not all(isinstance(dim, int) and dim > 0 for dim in output_shape):
            raise ValueError("output_shape must contain positive integers.")
        self.output_shape = output_shape

        super().__init__(file_list, randomize, loop, key)
        # ensure all supplied files are .mat
        if not all(file_path.endswith(".mat") for file_path in self.file_list):
            raise ValueError(
                "All files must be MATLAB .mat files with HDF5 format"
            )

        logger.debug(
            f"Initializing MATFlowFieldScheduler with "
            f"output_shape={self.output_shape}, "
            f"include_images={self.include_images}, "
            f"randomize={self.randomize}, loop={self.loop}"
        )

        logger.debug(f"Found {len(self.file_list)} files")

    @classmethod
    def _path_is_hdf5(cls, path: str) -> bool:
        """Check if a file is in HDF5 format.

        Args:
            path: Path to the file.

        Returns:
            True if the file is in HDF5 format, False otherwise.
        """
        return bool(h5py.is_hdf5(path))

    def load_file(self, file_path: str) -> SchedulerData:
        """Load .mat file (v4-v7.3) and return data dict.

        Args:
            file_path: Path to the .mat file.

        Returns:
            Dictionary containing 'V' (flow field).  When
            `self.include_images` is True, it must also hold 'I0' and 'I1'.

        Raises:
            ValueError: If required keys are missing or data has invalid shape.
        """

        def recursively_load_hdf5_group(
            group: h5py.Group | h5py.File, prefix: str = ""
        ) -> dict[str, np.ndarray]:
            """Flatten all datasets in an HDF5 tree into a dict."""
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
            logger.debug(f"Loaded {file_path} with version MATLAB v4-v7.2")
            data = {k: v for k, v in mat.items() if not k.startswith("__")}
        except (NotImplementedError, ValueError):
            if self._path_is_hdf5(file_path):
                # MATLAB v7.3 ⇒ fall back to h5py
                logger.debug(f"Falling back to HDF5 for {file_path}")
                with h5py.File(file_path, "r") as f:
                    data = recursively_load_hdf5_group(f)

        if data is None:
            raise ValueError(
                f"Failed to load {file_path} as HDF5 or legacy MATLAB."
            )

        # Validate the loaded data
        if "V" not in data:
            raise ValueError(
                f"Flow field not found in {file_path} (missing 'V')."
            )
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
        assert flow.shape[2] == 2 or flow.shape[0] == 2, (
            f"Flow field shape {flow.shape} is not valid. "
            "Expected shape to have 2 channels (e.g., (H, W, 2) or (2, H, W))."
        )
        if flow.shape[2] != 2:
            if flow.shape[0] == 2:
                flow = np.transpose(flow, (1, 2, 0))
            data["V"] = flow
        if flow.shape[:2] != self.output_shape:
            # Resize flow to output_shape and scale by the ratio
            # The original flow is assumed to be in pixels
            ratio_y = self.output_shape[0] / flow.shape[0]
            ratio_x = self.output_shape[1] / flow.shape[1]
            # Resize each channel separately using PIL (bilinear interpolation)
            # PIL resize expects (width, height)
            size = (self.output_shape[1], self.output_shape[0])
            flow_u = np.asarray(
                Image.fromarray(flow[..., 0]).resize(
                    size, Image.Resampling.BILINEAR
                )
            )
            flow_v = np.asarray(
                Image.fromarray(flow[..., 1]).resize(
                    size, Image.Resampling.BILINEAR
                )
            )
            flow_resized = np.stack(
                [flow_u * ratio_x, flow_v * ratio_y], axis=-1
            )
            data["V"] = flow_resized

        logger.debug(f"Loaded {file_path} with keys {list(data.keys())}")
        return SchedulerData(
            flow_fields=data["V"],
            images1=data["I0"] if self.include_images else None,
            images2=data["I1"] if self.include_images else None,
            files=(file_path,),
        )

    def get_next_slice(self) -> SchedulerData:
        """Retrieves the flow field slice and optionally the images.

        Returns:
            SchedulerData containing the flow field and, if requested,
            the previous and next images.

        Raises:
            FileEndedError: If the end of file data is reached.
            RuntimeError: If no data is currently cached.
        """
        data = self._cached_data
        if data is None:
            raise RuntimeError("No data is currently cached.")
        if self._slice_idx != 0:
            raise FileEndedError("End of file data reached.")
        if not self.include_images:
            data = data.update(images1=None, images2=None)

        return data

    def get_flow_fields_shape(self) -> tuple[int, int, int]:
        """Returns the shape of the flow field.

        Returns:
            Shape of the flow field.
        """
        return (*self.output_shape, 2)

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """Creates a MATFlowFieldScheduler from a dictionary.

        Args:
            config:
                Configuration dictionary containing the scheduler parameters.

        Returns:
            An instance of the scheduler.
        """
        return cls(
            file_list=config.get("file_list", []),
            randomize=config.get("randomize", False),
            loop=config.get("loop", False),
            include_images=config.get("include_images", False),
            output_shape=tuple(config.get("output_shape", (256, 256))),
        )
