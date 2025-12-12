"""PIV Dataset Class 1 Builder.

This script provides a full end-to-end pipeline for preparing the
PIV Dataset (Class 1) from the public Google Drive repositories.

==============================================================
 PIV Dataset Class 1 Builder
==============================================================

The pipeline performs the following steps:

  1) Download  - Retrieves raw ZIP archives from two official
     Google Drive folders published by the dataset authors.

  2) Extract   - Recursively unpacks all ZIP files and flattens
     the directory structure into a unified raw/ directory.

  3) Convert   - Locates flow/image triplets
                     *_flow.(flo|mat), *_img1.tif, *_img2.tif
                 and packs each into a MATLAB v7.3 .mat file
                 containing:
                     V  : optical flow (HxWx2)
                     I0 : first image
                     I1 : second image
                 Flow is resized to a given target resolution
                 with proper vector scaling.

  4) Split     - Downloads the official train/test lists from
                 the dataset repository, generates val/tune
                 subsets, and organizes packed samples into
                 train/val/test/tune folders.

The output structure is:

    out_dir/
        raw_class1/
        packed_class1/
        splits/
            train.txt
            val.txt
            test.txt
            tune.txt
        data/
            train/
            val/
            test/
            tune/

This script is intended to reproduce the dataset organization
used in many modern deep-learning-based PIV benchmarking pipelines.
"""

import os
import glob
import numpy as np
import h5py
from PIL import Image
import struct
import zipfile
import argparse
from pathlib import Path
import shutil
import re
import gdown
import random
from utils import download_file, read_list, write_list


# --- Google Drive folders provided by the dataset authors ---
GDRIVE_FOLDERS = [
    # data_zip1
    "https://drive.google.com/drive/folders/1wP2kkeX4M7nCAsSIi52yMpO96RiT4NXq",
    # data_zip2
    "https://drive.google.com/drive/folders/1uJIHonOZGfhWtZcR-F0aGH7tnbLbCFn0",
]

# Base URL for the raw files in the repository
BASE_URL = "https://raw.githubusercontent.com/shengzesnail/PIV_dataset/master"


def fetch_splits(out_path: Path, split_seed: int, split_ratios: list[int]) -> None:
    """Fetch official train/test split files and generate val/tune splits.

    Args:
        out_path: Directory to save split text files.
        split_seed: Random seed for shuffling when generating splits.
        split_ratios: List of three integers representing the
                      percentages for train/val/tune splits.
    """
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Download train and test lists
    train_dest = out_path / "train.txt"
    test_dest = out_path / "test.txt"

    if not download_file(f"{BASE_URL}/FlowData_train.list", train_dest):
        print("CRITICAL: Failed to download train list.")
        return

    if not download_file(f"{BASE_URL}/FlowData_test.list", test_dest):
        print("WARNING: Failed to download test list.")
        # Proceeding anyway as train is most important

    # 2. Generate val.txt and tune.txt from train.txt
    if train_dest.exists():
        full_train = read_list(train_dest)
        total = len(full_train)
        print(f"Loaded {total} items from train list.")

        # Shuffle deterministically
        random.seed(split_seed)
        random.shuffle(full_train)

        n_val = int(total * split_ratios[1] / 100)
        n_tune = int(total * split_ratios[2] / 100)
        n_train = total - n_val - n_tune

        new_train = full_train[:n_train]
        val_split = full_train[n_train : n_train + n_val]
        tune_split = full_train[n_train + n_val :]

        print(
            f"Splitting into:\n  Train: {len(new_train)}\n  Val:   {len(val_split)}\n  Tune:  {len(tune_split)}"
        )

        # Write out the new splits
        write_list(train_dest, new_train)
        write_list(out_path / "val.txt", val_split)
        write_list(out_path / "tune.txt", tune_split)

        print("Created train.txt, val.txt, tune.txt")

    else:
        print("Skipping split generation (train.txt missing).")


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


def download_from_gdrive(raw_dir_path: Path) -> None:
    """Download all files from the two public Google Drive folders.

    Downloads the contents into raw_dir_path, then recursively unzip
    every .zip we find.

    Args:
        raw_dir_path: Directory to store downloaded raw data.
    """
    raw_dir_path.mkdir(parents=True, exist_ok=True)

    print("Press Ctrl+C to interrupt download safely and proceed to processing.")

    try:
        for idx, url in enumerate(GDRIVE_FOLDERS, start=1):
            dest = raw_dir_path / f"gdrive_folder_{idx}"
            dest.mkdir(parents=True, exist_ok=True)
            print(f"\nDownloading Google Drive folder {idx} → {dest}")
            try:
                # gdown handles folder recursion
                gdown.download_folder(
                    url=url,
                    output=str(dest),
                    quiet=False,
                    use_cookies=False,
                )
            except KeyboardInterrupt:
                print("\n\nInterrupted by user! Stopping download loop.")
                break
            except Exception as e:
                print(f"Error downloading folder {idx}: {e}")
                # Optional: continue to next folder or break?
                # Let's break to be safe if it's a major error
                break
    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")

    # Now unzip everything we just downloaded
    print("\nExtracting all .zip archives under", raw_dir_path)
    for zip_path in raw_dir_path.rglob("*.zip"):
        print(f"  Extracting {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                # Extract into the main raw directory to avoid gdrive_folder_X nesting
                # This flattens the structure so 'backstep' ends up in raw_dir_path/backstep
                z.extractall(raw_dir_path)
        except zipfile.BadZipFile:
            print(f"  WARNING: {zip_path} is not a valid zip, skipping.")
            continue

        # Optional: delete zip to save space
        # zip_path.unlink()

    print("\nDownload + extraction from Google Drive complete.")


def load_split_file(path: Path) -> set[str]:
    """Return list of .mat file names from the split file.

    Args:
        path: Path to the split file.

    Returns:
        Set of .mat filenames listed in the split file.
    """
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # The split file might contain multiple columns (img1 img2 flow)
            # We want to canonicalize to the expected .mat filename
            parts = line.split()
            base_token = parts[0]

            p = Path(base_token)
            stem = p.name

            if "_img1" in stem:
                core = stem.split("_img1")[0]
            elif "_img2" in stem:
                core = stem.split("_img2")[0]
            elif "_flow" in stem:
                core = stem.split("_flow")[0]
            else:
                # Fallback: just remove extension
                core = p.stem

            out.append(f"{core}.mat")
    return set(out)


def perform_split(packed_root: Path, split_root: Path) -> None:
    """Copy packed dataset into split folders according to split text files.

    Args:
        packed_root: Directory containing packed .mat files.
        split_root: Directory to save split datasets.
    """
    split_root.mkdir(parents=True, exist_ok=True)
    split_files_dir = packed_root.parent / "splits"

    splits = ["train", "val", "test", "tune"]
    lists = {}

    print("\nLoading split files...\n")
    for s in splits:
        txt = split_files_dir / f"{s}.txt"
        if not txt.exists():
            raise FileNotFoundError(f"Missing split file: {txt}")

        lists[s] = load_split_file(txt)
        print(f"{s}: loaded {len(lists[s])} targets")

    # Walk all packed .mat files and distribute
    print("\nAssigning packed files to splits...\n")
    for mat_path in packed_root.rglob("*.mat"):
        fname = mat_path.name

        # Find the split(s) it belongs to.
        # ALLOW MULTIPLE ASSIGNMENTS e.g. if tune is subset of val
        assigned = []
        for split in splits:
            if fname in lists[split]:
                assigned.append(split)

        if not assigned:
            print(f"WARNING: {fname} not found in any split file.")
            continue

        # Copy to ALL assigned splits
        for s in assigned:
            rel = mat_path.relative_to(packed_root)

            # Check for Reynolds number in filename (e.g., backstep_Re1000_...)
            re_match = re.search(r"_(Re\d+)_", fname)
            if re_match:
                re_folder = re_match.group(1)
                target_dir = split_root / s / rel.parent / re_folder
            else:
                target_dir = split_root / s / rel.parent

            target_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy(mat_path, target_dir / fname)
            print(f"{fname} → {s}/{rel.parent}/{re_folder if re_match else ''}")

    print("\nSplitting complete!")
    print(f"Split datasets saved under: {split_root}")


def main(out_dir: str, target_shape: str) -> None:
    """Main function to orchestrate the dataset preparation workflow.

    Args:
        out_dir: Directory to save raw, packed, and split datasets.
        target_shape: Target shape (HxW) for resizing images and flow, e.g., '256x256'.
    """
    out_dir_path = Path(out_dir)
    try:
        target_shape_tuple = tuple(map(int, target_shape.split("x")))
    except Exception as e:
        print(
            f"Target shape is in the wrong format: {target_shape}. Use HxW, e.g., '256x256'. Error: {e}"
        )
        return

    raw_dir_path = out_dir_path / "raw_class1"
    packed_dir = out_dir_path / "packed_class1"
    split_output_dir = out_dir_path

    raw_dir_path.mkdir(parents=True, exist_ok=True)
    packed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download + extract from Google Drive
    if not any(raw_dir_path.iterdir()):
        download_from_gdrive(raw_dir_path)
    else:
        print(f"{raw_dir_path} not empty — skipping download/extraction.")

    # 2. Convert all downloaded data (recursively) to .mat
    print("\nRunning packing script (convert)...\n")
    convert(str(raw_dir_path), str(packed_dir), target_shape=target_shape_tuple)

    # 3. Split according to train/val/test/tune.txt
    perform_split(packed_root=packed_dir, split_root=split_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Where to store raw, packed, and split datasets",
    )
    parser.add_argument(
        "--split-ratio",
        type=str,
        required=False,
        default="80/10/10",
        help="Split ratio for train/val/tune as percentages, e.g., '80/10/10'",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        required=False,
        default=42,  # TODO: which one was the seed we used?
        help="Random seed for shuffling when generating splits",
    )
    parser.add_argument(
        "--target-shape",
        type=str,
        required=False,
        default="256x256",
        help="Target shape (HxW) for resizing images and flow, e.g., '256x256'",
    )
    args = parser.parse_args()

    try:
        split_ratios = list(map(int, args.split_ratio.split("/")))
        if len(split_ratios) != 3 or sum(split_ratios) != 100:
            raise ValueError
        out_path_split = Path(args.out_dir) / "splits"
        fetch_splits(out_path_split, args.split_seed, split_ratios)
        main(args.out_dir, args.target_shape)
    except Exception as e:
        print(
            f"Split ratio is in the wrong format: {args.split_ratio}. Use '80/10/10' format summing to 100. Error: {e}"
        )
        exit(1)
