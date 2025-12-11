"""Download, extract, convert, and split the PIV dataset (class 1)."""

import zipfile
import argparse
from pathlib import Path
import shutil
import re
import gdown
from convert import convert


# --- Google Drive folders provided by the dataset authors ---
GDRIVE_FOLDERS = [
    # data_zip1
    "https://drive.google.com/drive/folders/1wP2kkeX4M7nCAsSIi52yMpO96RiT4NXq",
    # data_zip2
    "https://drive.google.com/drive/folders/1uJIHonOZGfhWtZcR-F0aGH7tnbLbCFn0",
]


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


def perform_split(packed_root: Path, split_root: Path, split_files_dir: Path) -> None:
    """Copy packed dataset into split folders according to split text files.

    Args:
        packed_root: Directory containing all packed .mat files.
        split_root: Directory where to create
            train/, val/, test/, tune/ subdirs.
        split_files_dir: Directory containing
            train.txt, val.txt, test.txt, tune.txt.
    """
    split_root.mkdir(parents=True, exist_ok=True)

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


def main(out_dir: str, split_dir: str) -> None:
    """Main function to orchestrate the dataset preparation workflow.

    Args:
        out_dir: Where to store raw, packed, and split datasets.
        split_dir: Directory containing train.txt, val.txt, test.txt, tune.txt.
    """
    out_dir_path = Path(out_dir)
    split_files_dir = Path(split_dir)

    raw_dir_path = out_dir_path / "raw_class1"
    packed_dir = out_dir_path / "packed_class1"
    split_output_dir = out_dir_path / "splits"

    raw_dir_path.mkdir(parents=True, exist_ok=True)
    packed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download + extract from Google Drive
    if not any(raw_dir_path.iterdir()):
        download_from_gdrive(raw_dir_path)
    else:
        print(f"{raw_dir_path} not empty — skipping download/extraction.")

    # 2. Convert all downloaded data (recursively) to .mat
    print("\nRunning packing script (convert)...\n")
    convert(str(raw_dir_path), str(packed_dir))

    # 3. Split according to train/val/test/tune.txt
    perform_split(
        packed_root=packed_dir,
        split_root=split_output_dir,
        split_files_dir=split_files_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Where to store raw, packed, and split datasets",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        required=True,
        help="Directory containing train.txt, val.txt, test.txt, tune.txt",
    )
    args = parser.parse_args()
    main(args.out_dir, args.split_dir)
