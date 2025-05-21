"""FlowFieldScheduler to load the flow field data from files."""
import cv2
import h5py
import numpy as np
import scipy.io
from PIL import Image

from synthpix.scheduler import BaseFlowFieldScheduler
from synthpix.utils import logger


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

        logger.debug(f"Initialized with {len(self.file_list)} files")
        logger.debug(f"File list: {self.file_list}")

    @staticmethod
    def _path_is_hdf5(path: str) -> bool:
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
            if self._path_is_hdf5(file_path):
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

        logger.debug(f"Loaded {file_path} with keys {list(data.keys())}")
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
