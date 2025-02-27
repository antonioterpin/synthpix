"""Dataloader for sequences of images in a directory."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    """Dataset class to load sequences of images from a directory."""

    def __init__(self, root_dir: str, seq_length: int):
        """Initializes the dataset.

        Args:
            root_dir (str): Directory with all the images.
            seq_length (int): Number of consecutive images to load.
        """
        self.root_dir = root_dir
        self.seq_length = seq_length

        # List and sort the .jpg files (assuming names like 100.jpg, 101.jpg, etc.)
        self.image_files = sorted(
            [
                os.path.join(root_dir, f)
                for f in os.listdir(root_dir)
                if f.lower().endswith(".jpg")
            ]
        )

        # Ensure there are enough images for at least one sequence.
        if len(self.image_files) < self.seq_length:
            raise ValueError("Not enough images in the folder to form one sequence.")

    def __len__(self) -> int:
        """Returns the number of sequences that can be formed from the images.

        Returns:
            int: Number of sequences that can be formed.
        """
        # Each index gives a sequence of seq_length consecutive images.
        return len(self.image_files) - self.seq_length + 1

    def __getitem__(self, idx: int) -> np.ndarray:
        """Loads a sequence of images starting from index idx.

        Args:
            idx (int): Index of the first image in the sequence.

        Returns:
            np.ndarray: Sequence of images with shape (seq_length, H, W).
        """
        # Load seq_length consecutive images starting from index idx.
        imgs = []
        for i in range(idx, idx + self.seq_length):
            img_path = self.image_files[i]
            # Open image and convert to grayscale ('L')
            with Image.open(img_path) as img:
                img = img.convert("L")
                # Convert image to numpy array with type int8.
                img_np = np.array(img, dtype=np.uint8)
            imgs.append(img_np)

        # Stack images along a new first dimension, shape: (seq_length, H, W)
        imgs = np.stack(imgs, axis=0)
        return imgs


# Example usage: display each sequence of images in a plot and wait for a key press.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display sequences of images.")
    parser.add_argument(
        "--directory",
        type=str,
        default="src/analyses/image_data/images",
        help="Directory with images.",
    )
    parser.add_argument(
        "--seq-length", type=int, default=3, help="Number of images in each sequence."
    )
    args = parser.parse_args()

    directory = args.directory
    seq_length = args.seq_length

    # Create the dataset
    dataset = ImageDataset(directory, seq_length)

    # Create a DataLoader (batch_size is 1 to display one sequence at a time)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in dataloader:
        # Remove the batch dimension: sequence shape is (seq_length, H, W)
        sequence = batch[0].numpy()

        fig, axs = plt.subplots(1, seq_length, figsize=(seq_length * 3, 3))
        # In case seq_length is 1, make axs a list for consistency.
        if seq_length == 1:
            axs = [axs]

        for i in range(seq_length):
            # Display the image in grayscale.
            axs[i].imshow(sequence[i], cmap="gray", vmin=0, vmax=255)
            axs[i].axis("off")
            axs[i].set_title(f"Image {i+1}")

        plt.tight_layout()
        plt.show(block=False)
        plt.waitforbuttonpress()
        plt.close(fig)
