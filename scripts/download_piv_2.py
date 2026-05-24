"""Download and process the PIV class 2 dataset from Zenodo.

Optional flags ``--push-to``, ``--push-public``,
``--allow-public``, ``--push-token``, and
``--no-push-card`` push the resulting tree to a Hugging Face Hub dataset
repo via :func:`synthpix.hf.push_dataset` when ``--push-to`` is set.
Public pushes require the explicit ``--allow-public`` safety gate.
"""

import argparse
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from utils import download_file

# URL for the dataset
ZENODO_URL = "https://zenodo.org/records/4432496/files/Data_ProblemClass2_RAFT-PIV.zip?download=1"


def parse_proto(example_proto: tf.Tensor) -> dict[str, tf.Tensor]:
    """Parse a single TFRecord example.

    Args:
        example_proto: A serialized TFRecord example.

    Returns:
        Parsed features.
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


def process_tfrecord(tfrecord_path: str | Path, out_dir: str | Path) -> None:
    """Process a TFRecord file and convert its contents to .mat files.

    Args:
        tfrecord_path: Path to the TFRecord file.
        out_dir: Directory to save the converted .mat files.
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    dataset = tf.data.TFRecordDataset(tfrecord_path)  # pyright: ignore[reportArgumentType]
    print(f"Processing {tfrecord_path} -> {out_dir}")

    count = 0
    for raw_record in dataset:
        try:
            example = parse_proto(raw_record)

            target_raw = example.get("target")
            flow_raw = example.get("flow")

            if (
                target_raw is None
                or flow_raw is None
                or target_raw == b""
                or flow_raw == b""
            ):
                if count == 0:
                    print("'target' or 'flow' not found or empty. Skipping...")
                continue

            # Decode target -> I0, I1
            try:
                target_flat = np.frombuffer(
                    target_raw.numpy(), dtype=np.float32
                )
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
            out_path = out_dir_path / fname

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


def main(out_dir: str) -> None:
    """Main function to download, extract, and process the PIV class 2 dataset.

    Args:
        out_dir: Output directory for the processed dataset.
    """
    out_path = Path(out_dir)
    raw_dir = out_path / "raw_class2"
    packed_dir = out_path / "packed_class2"
    raw_dir.mkdir(parents=True, exist_ok=True)
    packed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download
    zip_path = raw_dir / "Data_ProblemClass2_RAFT-PIV.zip"
    if not zip_path.exists():
        print(
            "Starting download... \
                (This is 12GB, ensure you have stable connection)"
        )
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
        print(
            "CRITICAL: Bad zip file. \
                The download might be incomplete or corrupted."
        )
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


_PUSH_SOURCE_URL = "https://github.com/shengzesnail/PIV_dataset"
_PUSH_CITATION = (
    "Cai, S., Zhou, S., Xu, C., Gao, Q. (2019). "
    "Dense motion estimation of particle images via a convolutional "
    "neural network. Exp Fluids 60, 73.\n\n"
    "@article{cai2019dense,\n"
    "  title={Dense motion estimation of particle images via "
    "a convolutional neural network},\n"
    "  author={Cai, Shengze and Zhou, Shichao and Xu, Chuanqi "
    "and Gao, Qi},\n"
    "  journal={Experiments in Fluids},\n"
    "  volume={60},\n"
    "  number={4},\n"
    "  pages={73},\n"
    "  year={2019},\n"
    "  publisher={Springer}\n"
    "}"
)


def _default_card_meta(repo_id: str) -> "object":
    """Return a default ``DatasetCardMeta`` for the class-2 dataset.

    The import is deferred so the rest of this script keeps working
    without the ``[hf]`` extra installed.

    Args:
        repo_id: ``<owner>/<name>`` identifier on the Hub.

    Returns:
        DatasetCardMeta: A populated card metadata object.
    """
    from synthpix.hf import DatasetCardMeta  # noqa: PLC0415

    name = repo_id.split("/", 1)[-1]
    return DatasetCardMeta(
        name=name,
        source_url=_PUSH_SOURCE_URL,
        citation=_PUSH_CITATION,
        pretty_name=name,
        tags=("PIV", "synthetic", "optical-flow", "class-2"),
    )


def _maybe_push(args: argparse.Namespace, out_dir_path: Path) -> None:
    """Optionally push ``out_dir_path`` to the Hub based on CLI flags.

    Args:
        args: Parsed CLI namespace; ``args.push_to`` controls activation.
        out_dir_path: Local directory uploaded to the Hub.
    """
    if not getattr(args, "push_to", None):
        return

    from synthpix.hf import push_dataset  # noqa: PLC0415

    card_meta = None if args.no_push_card else _default_card_meta(args.push_to)
    sha = push_dataset(
        local_dir=out_dir_path,
        repo_id=args.push_to,
        private=not args.push_public,
        allow_public=args.allow_public,
        token=args.push_token,
        card_meta=card_meta,
    )
    print(sha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--push-to",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face Hub dataset repo id "
            "(<owner>/<name>) to push the built dataset to."
        ),
    )
    parser.add_argument(
        "--push-public",
        action="store_true",
        default=False,
        help=(
            "Push as a public repo (private by default). Requires "
            "--allow-public. Class-2 sources are research-only; do not "
            "redistribute publicly without explicit permission."
        ),
    )
    parser.add_argument(
        "--allow-public",
        action="store_true",
        default=False,
        help="Safety gate companion for --push-public.",
    )
    parser.add_argument(
        "--push-token",
        type=str,
        default=None,
        help="Explicit HF token; falls back to HF_TOKEN/cache.",
    )
    parser.add_argument(
        "--no-push-card",
        action="store_true",
        default=False,
        help="Skip dataset-card generation on push.",
    )
    args = parser.parse_args()

    if args.push_public and not args.allow_public:
        print(
            "--push-public requires --allow-public (safety gate).",
            file=sys.stderr,
        )
        sys.exit(2)

    main(args.out_dir)
    _maybe_push(args, Path(args.out_dir))
