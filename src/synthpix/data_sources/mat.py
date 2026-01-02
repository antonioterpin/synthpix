"""MATDataSource implementation."""

import scipy.io
import h5py
import numpy as np
from PIL import Image
import logging
from typing import Any

from .base import FileDataSource

logger = logging.getLogger(__name__)

def recursively_load_hdf5_group(group: h5py.Group, prefix: str = "") -> dict:
    """Flatten all datasets in an HDF5 tree into a dict.
    
    Args:
        group: HDF5 group to flatten.
        prefix: Prefix for the dataset paths.
    
    Returns:
        Dictionary containing the flattened datasets.
    """
    out = {}
    for name, item in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(item, h5py.Dataset):
            out[path] = item[()]
        elif isinstance(item, h5py.Group):
            out.update(recursively_load_hdf5_group(item, path))
    return out

class MATDataSource(FileDataSource):
    """DataSource for loading .mat files (MATLAB v4-v7.3)."""
    
    _file_pattern = "**/*.mat"

    def __init__(
        self, 
        dataset_path: list[str] | str,
        include_images: bool = False,
        output_shape: tuple[int, int] = (256, 256)
    ) -> None:
        """Initializes the MATDataSource.
        
        Args:
            dataset_path: Path to the dataset.
            include_images: Whether to include images.
            output_shape: Output shape for the flow field.
        """
        self._include_images = include_images
        self.output_shape = output_shape
        super().__init__(dataset_path)

    @property
    def include_images(self) -> bool:
        """Whether to include images.
        
        Returns:
            bool: Whether to include images.
        """
        return self._include_images

    def load_file(self, file_path: str) -> dict[str, Any]:
        """Load a .mat file and return flow field (and optionally images).
        
        Args:
            file_path: Path to the .mat file.

        Returns:
            Dictionary containing data (e.g. 'flow_fields', 'images1', etc).
        """
        data = None
        
        # 1. Try Scipy (v4-v7.2)
        try:
            mat = scipy.io.loadmat(
                file_path,
                struct_as_record=False,
                squeeze_me=True,
            )
            data = {k: v for k, v in mat.items() if not k.startswith("__")}
        except Exception:
            # 2. Try HDF5 (v7.3)
            if h5py.is_hdf5(file_path):
                with h5py.File(file_path, "r") as f:
                    data = recursively_load_hdf5_group(f)
        
        if data is None:
            raise ValueError(f"Failed to load {file_path} as HDF5 or legacy MATLAB.")
        
        if "V" not in data:
            raise ValueError(f"Flow field not found in {file_path} (missing 'V').")
        
        # 3. Process Flow
        flow = data["V"]
        # Ensure channel last: (H, W, 2)
        if flow.shape[0] == 2 and flow.shape[2] != 2:
            flow = np.transpose(flow, (1, 2, 0)) # (2, H, W) -> (H, W, 2)
            
        # Resize Flow
        if flow.shape[:2] != self.output_shape:
            # Resize logic
            ratio_y = self.output_shape[0] / flow.shape[0]
            ratio_x = self.output_shape[1] / flow.shape[1]
            size = (self.output_shape[1], self.output_shape[0]) # PIL expects (W, H)
            
            flow_u = np.asarray(
                Image.fromarray(flow[..., 0]).resize(size, Image.Resampling.BILINEAR)
            )
            flow_v = np.asarray(
                Image.fromarray(flow[..., 1]).resize(size, Image.Resampling.BILINEAR)
            )
            # Scale values by resize ratio
            flow = np.stack([flow_u * ratio_x, flow_v * ratio_y], axis=-1)

        result = {
            "flow_fields": flow.astype(np.float32), 
            "file": file_path
        }
        
        # 4. Process Images (if requested)
        if self.include_images:
            if not all(k in data for k in ("I0", "I1")):
                raise ValueError(
                    f"Image loading not supported for {file_path}: "
                    "missing required keys 'I0'/'I1'."
                )

            img0 = data["I0"]
            img1 = data["I1"]
            
            # Resize Images
            if img0.shape[:2] != self.output_shape:
                size = (self.output_shape[1], self.output_shape[0])
                img0 = np.asarray(Image.fromarray(img0).resize(size))
                img1 = np.asarray(Image.fromarray(img1).resize(size))
            
            result["images1"] = img0
            result["images2"] = img1
        return result
