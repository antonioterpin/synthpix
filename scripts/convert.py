"""Convert PIV dataset triples into packed .mat files.

TODO: Merge this to download_piv_1.py.
TODO: Make sure there is only one script that does everything.
    just like download_piv_2.py
"""

import os
import glob
import numpy as np
import h5py
from PIL import Image
import struct
import argparse


def read_flow(path: str) -> np.ndarray:
    """Read a flow file in .flo or .mat format.

    Args:
        path: Path to the flow file.

    Returns:
        Numpy array of the optical flow.
    """
    if path.endswith(".flo"):
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"PIEH":
                raise Exception("Invalid .flo file (bad magic number)")
            width = struct.unpack("i", f.read(4))[0]
            height = struct.unpack("i", f.read(4))[0]
            data = np.fromfile(f, np.float32, count=2 * width * height)
            flow = np.reshape(data, (height, width, 2))
        return flow
    elif path.endswith(".mat"):
        import scipy.io

        mat = scipy.io.loadmat(path)
        for k in ["V", "flow", "u", "uv", "F"]:
            if k in mat:
                arr = mat[k]
                # Make sure (H, W, 2)
                if arr.ndim == 3 and arr.shape[2] == 2:
                    return arr
                elif arr.ndim == 3 and arr.shape[0] == 2:
                    return np.transpose(arr, (1, 2, 0))
        raise ValueError(f"Unknown structure in {path}")
    else:
        raise ValueError(f"Unknown flow format: {path}")


def read_img(path: str) -> np.ndarray:
    """Read a grayscale image and return as a numpy array.

    Args:
        path: Path to the image file.

    Returns:
        Numpy array of the grayscale image.
    """
    img = Image.open(path)
    img = img.convert("L") if img.mode != "L" else img

    return np.array(img)


def resize_flow(flow: np.ndarray, shape: tuple) -> np.ndarray:
    """Resize flow to target shape while scaling flow values appropriately.

    Args:
        flow: Input flow array of shape (H, W, 2).
        shape: Target shape (height, width).

    Returns:
        Resized flow array of shape (target_height, target_width, 2).
    """
    h0, w0 = flow.shape[:2]
    if (h0, w0) == shape:
        print(f"Flow already at target shape {shape}, skipping resize.")
        return flow
    # Resize each channel separately using PIL (bilinear interpolation)
    # PIL resize expects (width, height), so we reverse the shape
    flow_u = np.asarray(
        Image.fromarray(flow[..., 0]).resize(shape[::-1], Image.Resampling.BILINEAR)
    )
    flow_v = np.asarray(
        Image.fromarray(flow[..., 1]).resize(shape[::-1], Image.Resampling.BILINEAR)
    )
    # Scale flow to new size
    flow_resized = np.stack(
        [flow_u * (shape[1] / w0), flow_v * (shape[0] / h0)], axis=-1
    )
    return flow_resized


def pack_triplet(
    flow_path: str, img1_path: str, img2_path: str, out_path: str, target_shape: tuple
) -> None:
    """Pack a flow and its two corresponding images into a .mat file.

    Args:
        flow_path: Path to the flow file.
        img1_path: Path to the first image file.
        img2_path: Path to the second image file.
        out_path: Where to save the packed .mat file.
        target_shape: Target shape (height, width) for resizing images and flow.
    """
    flow = read_flow(flow_path)
    I0 = read_img(img1_path)
    I1 = read_img(img2_path)
    flow = resize_flow(flow, target_shape)
    # Save as HDF5-based .mat (MATLAB v7.3)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("V", data=flow)
        f.create_dataset("I0", data=I0)
        f.create_dataset("I1", data=I1)
    print(f"Saved {out_path}")


def convert(dataset_dir: str, out_dir: str, target_shape: tuple) -> None:
    """Convert PIV dataset triples into packed .mat files.

    Args:
        dataset_dir: Directory containing flow and images.
        out_dir: Where to save packed .mat files.
        target_shape: Target shape (height, width) for resizing images and flow.
    """
    print(f"Packing dataset from {dataset_dir} to {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    # Recursively find all *_flow.* files
    for flow_path in sorted(
        glob.glob(os.path.join(dataset_dir, "**", "*_flow.*"), recursive=True)
    ):
        rel_dir = os.path.relpath(os.path.dirname(flow_path), dataset_dir)
        prefix = flow_path.rsplit("_flow", 1)[0]
        img1_path = prefix + "_img1.tif"
        img2_path = prefix + "_img2.tif"
        if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
            print(f"Missing image for {flow_path}, skipping.")
            continue
        # Preserve subfolder structure in output
        out_subdir = os.path.join(out_dir, rel_dir)
        os.makedirs(out_subdir, exist_ok=True)
        out_name = os.path.basename(prefix) + ".mat"
        out_path = os.path.join(out_subdir, out_name)
        try:
            pack_triplet(flow_path, img1_path, img2_path, out_path, target_shape)
        except Exception as e:
            print(f"Failed for {prefix}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pack PIV dataset triples into HDF5 .mat files."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing flow and images",
    )
    parser.add_argument(
        "--out_dir", type=str, required=True, help="Where to save packed .mat files"
    )
    args = parser.parse_args()
    convert(args.dataset_dir, args.out_dir, (256, 256))
