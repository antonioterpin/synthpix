"""Local dataset layout introspection for the synthpix HF integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_SPLITS = ("train", "val", "test", "tune")


@dataclass(frozen=True)
class LayoutSummary:
    """Summary of a local dataset directory layout.

    Attributes:
        splits: ``.mat`` file count per split that is present on disk.
        reynolds_by_split: Sorted unique names of immediate subdirectories
            grouped by split (typically Reynolds-number or scenario folders).
        total_bytes: Sum of the size of every regular file under ``root``.
        mat_files: Number of ``.mat`` files across all splits.
        extra_files: Number of non-``.mat`` regular files (dotfiles excluded).
    """

    splits: dict[str, int] = field(default_factory=dict)
    reynolds_by_split: dict[str, list[str]] = field(default_factory=dict)
    total_bytes: int = 0
    mat_files: int = 0
    extra_files: int = 0


def _is_hidden(path: Path) -> bool:
    return path.name.startswith(".")


def _scan_split(split_dir: Path) -> tuple[int, list[str], int, int, int]:
    """Walk a split directory and gather counts.

    Returns:
        tuple: ``(mat_count, reynolds_subdirs, extra_count, total_bytes,
            visited_files)``.
    """
    mat_count = 0
    extra_count = 0
    total_bytes = 0
    subdir_names: set[str] = set()

    for entry in split_dir.iterdir():
        if _is_hidden(entry):
            continue
        if entry.is_dir():
            subdir_names.add(entry.name)
            for nested in entry.rglob("*"):
                if _is_hidden(nested) or not nested.is_file():
                    continue
                total_bytes += nested.stat().st_size
                if nested.suffix == ".mat":
                    mat_count += 1
                else:
                    extra_count += 1
        elif entry.is_file():
            total_bytes += entry.stat().st_size
            if entry.suffix == ".mat":
                mat_count += 1
            else:
                extra_count += 1

    reynolds = sorted(subdir_names)
    return mat_count, reynolds, extra_count, total_bytes, 0


def inspect_local_layout(root: Path) -> LayoutSummary:
    """Inspect a local dataset directory and summarize its layout.

    Args:
        root: Path to the dataset root.

    Returns:
        LayoutSummary: A frozen snapshot of split counts, immediate
            subdirectory names, byte totals, and ``.mat``/extra file counts.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
        NotADirectoryError: If ``root`` is not a directory.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    splits: dict[str, int] = {}
    reynolds_by_split: dict[str, list[str]] = {}
    total_bytes = 0
    mat_files = 0
    extra_files = 0

    for split in _SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        mat_count, reynolds, extra_count, split_bytes, _ = _scan_split(
            split_dir
        )
        splits[split] = mat_count
        reynolds_by_split[split] = reynolds
        mat_files += mat_count
        extra_files += extra_count
        total_bytes += split_bytes

    return LayoutSummary(
        splits=splits,
        reynolds_by_split=reynolds_by_split,
        total_bytes=total_bytes,
        mat_files=mat_files,
        extra_files=extra_files,
    )
