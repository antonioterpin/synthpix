"""Utility functions for file downloading and list handling.

TODO: add documentation about how to call download_piv_1.py and download_piv_2.py (in the readme)
"""

from pathlib import Path
import urllib.request


def download_file(url: str, output_path: Path) -> bool:
    """Download a file from a URL to the specified output path.

    Args:
        url: The URL to download from.
        output_path: The local file path to save the downloaded file.

    Returns:
        True if download succeeded, False otherwise.
    """
    print(f"Downloading {url} to {output_path}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print("Done.")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def read_list(path: Path) -> list[str]:
    """Read a list of items from a text file, one per line.

    Args:
        path: Path to the text file.

    Returns:
        List of items read from the file.
    """
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def write_list(path: Path, items: list[str]) -> None:
    """Write a list of items to a text file, one per line.

    Args:
        path: Path to the text file.
        items: List of items to write.
    """
    with open(path, "w") as f:
        for item in items:
            f.write(f"{item}\n")
