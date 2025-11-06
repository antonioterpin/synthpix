"""NumpyFlowFieldScheduler to load flow fields from .npy files."""

import os
import re
from typing_extensions import Self

import jax
import numpy as np
from goggles import get_logger
from PIL import Image

from .base import BaseFlowFieldScheduler

logger = get_logger(__name__)


class NumpyFlowFieldScheduler(BaseFlowFieldScheduler):
    """Scheduler for loading flow fields from .npy files.

    Optionally, paired JPEG images can be validated and returned
    alongside the flow fields.

    Each .npy file must be named 'flow_<t>.npy' and, if images are enabled,
    will be paired with 'img_<t-1>.jpg' and 'img_<t>.jpg' in the same folder.
    """

    # instruct base class to glob only the flow_*.npy files
    _file_pattern = "flow_*.npy"

    def __init__(
        self,
        file_list: list[str] | str,
        randomize: bool = False,
        loop: bool = False,
        include_images: bool = False,
        key: jax.random.PRNGKey = None,
    ):
        """Initializes the Numpy scheduler.

        This scheduler loads flow fields from .npy files and can optionally
        validate and return paired JPEG images.
        The .npy files must be named 'flow_<t>.npy' and the images must be
        named 'img_<t-1>.jpg' and 'img_<t>.jpg' in the same folder.

        Args:
            file_list: A directory, single .npy file, or list of .npy paths.
            randomize: If True, shuffle file order per epoch.
            loop: If True, cycle indefinitely.
            include_images: If True, validate and return paired JPEG images.
            key: Random key for reproducibility.
        """
        if not isinstance(include_images, bool):
            raise ValueError("include_images must be a boolean value.")

        self.include_images = include_images
        super().__init__(file_list, randomize, loop, key)

        # ensure all supplied files are .npy
        if not all(fp.endswith(".npy") for fp in self.file_list):
            raise ValueError("All files must be numpy files with '.npy' extension")

        # validate image pairs only if requested
        if self.include_images:
            for flow_path in self.file_list:
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

    def load_file(self, file_path: str) -> np.ndarray:
        """Load the raw flow array from .npy.

        Args:
            file_path: Path to the .npy file.

        Returns: Loaded flow array.
        """
        return np.load(file_path)

    def get_next_slice(self) -> np.ndarray | dict[str, np.ndarray]:
        """Return either the flow array or, if enabled, flow plus images.

        Returns:
            Either the flow array or a dictionary with flow and images.
        """
        flow = self._cached_data
        if not self.include_images:
            return flow

        # load images on-demand
        mb = re.match(r"flow_(\d+)\.npy$", os.path.basename(self._cached_file))
        t = int(mb.group(1))
        folder = os.path.dirname(self._cached_file)
        prev = np.array(
            Image.open(os.path.join(folder, f"img_{t-1}.jpg")).convert("RGB")
        )
        nxt = np.array(Image.open(os.path.join(folder, f"img_{t}.jpg")).convert("RGB"))
        return {"flow": flow, "img_prev": prev, "img_next": nxt}

    def get_flow_fields_shape(self) -> tuple[int, ...]:
        """Return the shape of a single flow array.

        Returns: Shape of the flow array.
        """
        return np.load(self.file_list[0]).shape

    def __next__(self) -> np.ndarray | dict[str, np.ndarray]:
        """Iterate over .npy files, returning flow or flow+images.

        Returns: Either the flow array or a dictionary with flow and images.
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

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """Creates a NumpyFlowFieldScheduler instance from a configuration.

        Args:
            config:
                Configuration dictionary containing the scheduler parameters.

        Returns: An instance of the scheduler.
        """
        return NumpyFlowFieldScheduler(
            file_list=config["scheduler_files"],
            randomize=config.get("randomize", False),
            loop=config.get("loop", True),
            include_images=config.get("include_images", False),
        )
