"""Download and process the PIV class 2 dataset from Zenodo."""

import argparse
import zipfile
from pathlib import Path
import h5py
import numpy as np
import tensorflow as tf
from utils import download_file

# URL for the dataset
ZENODO_URL = "https://zenodo.org/records/4432496/files/Data_ProblemClass2_RAFT-PIV.zip?download=1"


def parse_proto(example_proto: tf.Tensor) -> dict:
    """Parse a single TFRecord example.

    Args:
        example_proto (tf.Tensor): A serialized TFRecord example.

    Returns:
        dict: Parsed features.
            - 'target': tf.string tensor - raw bytes of images
            - 'flow': tf.string tensor - raw bytes of optical flow
            - 'label': tf.string tensor (if exists) - raw bytes of labels
    """
    feature_description = {
        "target": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "flow": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "label": tf.io.FixedLenFeature([], tf.string, default_value=""),
    }
    return tf.io.parse_single_example(example_proto, feature_description)


def process_tfrecord(tfrecord_path: str, out_dir: str) -> None:
    """Process a TFRecord file and convert its contents to .mat files.

    TODO: check arguments types
    Args:
        tfrecord_path (str): Path to the TFRecord file.
        out_dir (str): Directory to save the converted .mat files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    print(f"Processing {tfrecord_path} -> {out_dir}")

    count = 0
    for raw_record in dataset:
        try:
            example = parse_proto(raw_record)

            target_raw = example["target"]
            flow_raw = example["flow"]

            if target_raw == b"" or flow_raw == b"":
                if count == 0:
                    print("Keys 'target' or 'flow' not found. Skipping...")
                continue

            # Decode target -> I0, I1
            try:
                target_flat = np.frombuffer(target_raw.numpy(), dtype=np.float32)
                if target_flat.size != 256 * 256 * 2:
                    print(f"Target size mismatch: {target_flat.size}")
                    continue
                target = target_flat.reshape(256, 256, 2)

                # Assume channel 0 is I0, channel 1 is I1
                I0 = target[..., 0]
                I1 = target[..., 1]
            except Exception as e:
                print(f"Error decoding target: {e}")
                continue

            # Decode flow
            try:
                flow_flat = np.frombuffer(flow_raw.numpy(), dtype=np.float32)
                if flow_flat.size != 256 * 256 * 2:
                    print(f"Flow size mismatch: {flow_flat.size}")
                    continue
                flow = flow_flat.reshape(256, 256, 2)
            except Exception as e:
                print(f"Error decoding flow: {e}")
                continue

            # Ensure shapes for .mat (H,W for images and H,W,2 for flow)
            # convert.py used I0 shape (256, 256)
            fname = f"sample_{count:05d}.mat"
            out_path = out_dir / fname

            with h5py.File(out_path, "w") as f:
                f.create_dataset("V", data=flow)
                f.create_dataset("I0", data=I0)
                f.create_dataset("I1", data=I1)

            count += 1
            if count % 100 == 0:
                print(f"Processed {count} records...", end="\r")

        except Exception as e:
            print(f"Error processing record {count}: {e}")
            continue

    print(f"\nFinished {tfrecord_path}: {count} records.")


def main(out_dir: str):
    """Main function to download, extract, and process the PIV class 2 dataset.

    Args:
        out_dir (str): Output directory for the processed dataset.
    """
    out_path = Path(out_dir)
    raw_dir = out_path / "raw_class2"
    packed_dir = out_path / "packed_class2"
    raw_dir.mkdir(parents=True, exist_ok=True)
    packed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download
    zip_path = raw_dir / "Data_ProblemClass2_RAFT-PIV.zip"
    if not zip_path.exists():
        print("Starting download... (This is 12GB, ensure you have stable connection)")
        if not download_file(ZENODO_URL, zip_path):
            print("Download failed.")
            return
    else:
        print("Zip file already exists, skipping download.")

    # 2. Extract
    print("Extracting...")
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(raw_dir)
    except zipfile.BadZipFile:
        print("CRITICAL: Bad zip file. The download might be incomplete or corrupted.")
        return
    except Exception as e:
        print(f"Extraction error: {e}")
        return

    # 3. Find TFRecords
    tfrecords = list(raw_dir.glob("*.tfrecord*"))
    print(f"Found {len(tfrecords)} TFRecord files.")

    for tfr in tfrecords:
        name = tfr.name
        if "RAFT256" not in name:
            continue

        print(f"Processing {name}")

        if "Training" in name:
            subdir = packed_dir / "train"
        elif "Validation" in name:
            subdir = packed_dir / "val"
        else:
            subdir = packed_dir / "other"

        process_tfrecord(tfr, subdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    main(args.out_dir)
