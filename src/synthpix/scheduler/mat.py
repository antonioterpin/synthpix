"""FlowFieldScheduler to load the flow field data from files."""

import itertools as it
from typing_extensions import Self
import cv2
import h5py
import jax
import numpy as np
import scipy.io
from goggles import get_logger
from PIL import Image

from .base import BaseFlowFieldScheduler

logger = get_logger(__name__)


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
        file_list: list[str] | str,
        randomize: bool = False,
        loop: bool = False,
        include_images: bool = False,
        output_shape: tuple[int, int] = (256, 256),
        key: jax.random.PRNGKey = None,
    ):
        """Initializes the MATFlowFieldScheduler.

        Args:
            file_list: A directory, single .mat file, or list of .mat paths.
            randomize: If True, shuffle file order per epoch.
            loop: If True, cycle indefinitely by wrapping around.
            include_images: If True, return a tuple (I0, I1, V).
            output_shape: The desired output shape for the flow fields.
                Must be a tuple of two integers (height, width).
            key: Random key for reproducibility.
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
            raise ValueError("All files must be MATLAB .mat files with HDF5 format")

        logger.debug(
            f"Initializing MATFlowFieldScheduler with output_shape={self.output_shape}, "
            f"include_images={self.include_images}, "
            f"randomize={self.randomize}, loop={self.loop}"
        )

        logger.debug(f"Found {len(self.file_list)} files")

    @classmethod
    def _path_is_hdf5(cls, path: str) -> bool:
        """Check if a file is in HDF5 format.

        Args:
            path: Path to the file.

        Returns: True if the file is in HDF5 format, False otherwise.
        """
        return h5py.is_hdf5(path)

    def load_file(self, file_path: str):
        """Load any MATLAB .mat file (v4, v5/6/7, or v7.3) and return its data dict.

        Args:
            file_path: Path to the .mat file.

        Returns: Dictionary containing 'V' (flow field).  When
            `self.include_images` is True, it must also hold 'I0' and 'I1'.
        """

        def recursively_load_hdf5_group(group, prefix=""):
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
            flow_resized = cv2.resize(
                flow, self.output_shape, interpolation=cv2.INTER_LINEAR
            )
            flow_resized[..., 0] *= ratio_x
            flow_resized[..., 1] *= ratio_y
            data["V"] = flow_resized

        logger.debug(f"Loaded {file_path} with keys {list(data.keys())}")
        return data

    def get_next_slice(self) -> np.ndarray | dict:
        """Retrieves the flow field slice and optionally the images.

        Returns:
            Flow field with shape (X, Y, Z, 2) or
            a dict containing the flow field and images.
        """
        data = self._cached_data
        flow_field = data["V"]

        if not self.include_images:
            return flow_field

        img_prev = data["I0"]
        img_next = data["I1"]
        return {"flow": flow_field, "img_prev": img_prev, "img_next": img_next}

    def get_flow_fields_shape(self) -> tuple[int, ...]:
        """Returns the shape of the flow field.

        Returns: Shape of the flow field.
        """
        return self.output_shape + (2,)

    def __next__(self) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Iterate over .mat files, returning flow or flow+images.

        Returns:
            Flow field with shape (X, Y, Z, 2) or
            a tuple (I0, I1, V) if `include_images` is True.
        """
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

    def get_batch(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        """Retrieves a batch of flow fields using the current scheduler state.

        Args:
            batch_size: int
                The number of flow fields to retrieve in the batch.

        Returns:
            A tuple containing:
            - img_prevs: np.ndarray of previous images
            - img_nexts: np.ndarray of next images
            - flows: np.ndarray of flow fields
            or, `include_images` is False, a batch of flow fields.

        Raises:
            StopIteration: If the iterator is fully exhausted.
            Warning: If fewer slices than `batch_size` are available
                and `loop` is False
        """
        if self.include_images:
            batch = [
                (s["flow"], s["img_prev"], s["img_next"])
                for s in it.islice(self, batch_size)
            ]

            if len(batch) < batch_size and not self.loop:
                logger.warning(
                    f"Skipping the last {len(batch)} slices."
                    "If undesired, use loop or a batch size dividing "
                    "the number of slices in the dataset."
                )
                raise StopIteration

            flows, img_prevs, img_nexts = zip(*batch)
            return (
                np.array(img_prevs, dtype=np.float32),
                np.array(img_nexts, dtype=np.float32),
                np.array(flows, dtype=np.float32),
            )

        else:
            return super().get_batch(batch_size)

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """Creates a MATFlowFieldScheduler instance from a configuration dictionary.

        Args:
            config:
                Configuration dictionary containing the scheduler parameters.

        Returns:
            An instance of the scheduler.
        """
        return MATFlowFieldScheduler(
            file_list=config["scheduler_files"],
            randomize=config.get("randomize", False),
            loop=config.get("loop", False),
            include_images=config.get("include_images", False),
            output_shape=tuple(config.get("output_shape", (256, 256))),
        )
