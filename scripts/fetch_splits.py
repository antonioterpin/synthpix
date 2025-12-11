"""Fetch PIV dataset splits and create val/tune subsets from train.

TODO: Explain what all this is.
TODO: Add parameters to allow custom split ratios.
TODO: Allow selection of the random seed, set at the beginning. Default is the one we use.
TODO: Check results are consistent with ADMM.
TODO: this should be piv_dataset_1... since it is only for class 1.
"""

import argparse
import random
from pathlib import Path
from utils import download_file, read_list, write_list

# Base URL for the raw files in the repository
BASE_URL = "https://raw.githubusercontent.com/shengzesnail/PIV_dataset/master"


def main(out_dir: str) -> None:
    """Main function to orchestrate the downloading and splitting.

    Args:
        out_dir: Directory to save split files.
    """
    out_path = Path(out_dir)
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
    # 80% train, 10% val, 10% tune
    if train_dest.exists():
        full_train = read_list(train_dest)
        total = len(full_train)
        print(f"Loaded {total} items from train list.")

        # Shuffle deterministically
        random.seed(42)
        random.shuffle(full_train)

        n_val = int(total * 0.1)
        n_tune = int(total * 0.1)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", required=True, help="Directory to save split files"
    )
    args = parser.parse_args()
    main(args.out_dir)
