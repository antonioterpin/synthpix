import argparse
import random
from pathlib import Path
import urllib.request

# Base URL for the raw files in the repository
BASE_URL = "https://raw.githubusercontent.com/shengzesnail/PIV_dataset/master"

def download_file(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print("Done.")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def read_list(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def write_list(path, items):
    with open(path, "w") as f:
        for item in items:
            f.write(f"{item}\n")

def main(out_dir):
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
    # User instruction: "val and tune are subsets of the original train list"
    if train_dest.exists():
        full_train = read_list(train_dest)
        total = len(full_train)
        print(f"Loaded {total} items from train list.")

        # Shuffle deterministically
        random.seed(42)
        random.shuffle(full_train)

        # Let's allocate:
        # 80% real train
        # 10% val
        # 10% tune
        n_val = int(total * 0.1)
        n_tune = int(total * 0.1)
        n_train = total - n_val - n_tune

        new_train = full_train[:n_train]
        val_split = full_train[n_train:n_train+n_val]
        tune_split = full_train[n_train+n_val:]

        print(f"Splitting into:\n  Train: {len(new_train)}\n  Val:   {len(val_split)}\n  Tune:  {len(tune_split)}")

        # Overwrite train with the reduced set? 
        # Usually 'train.txt' implies the set used for training. 
        # But if the user says "subsets of the original train list", maybe they mean Val/Tune are *taken out* of it.
        # I will overwrite train.txt to be safe so we don't leak data into val/tune.
        write_list(train_dest, new_train)
        write_list(out_path / "val.txt", val_split)
        write_list(out_path / "tune.txt", tune_split)
        
        print("Created train.txt, val.txt, tune.txt")

    else:
        print("Skipping split generation (train.txt missing).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, help="Directory to save split files")
    args = parser.parse_args()
    main(args.out_dir)
