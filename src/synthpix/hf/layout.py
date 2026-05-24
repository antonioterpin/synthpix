"""Local dataset layout introspection for the synthpix HF integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_SPLITS = ("train", "val", "test", "tune")


@dataclass(frozen=True)
class LayoutSummary:
    """Summary of a local dataset directory layout.

    The dataclass is declared ``frozen=True``, which prevents attribute
    reassignment, but the nested ``dict`` / ``list`` members are themselves
    mutable in place. Callers that share a ``LayoutSummary`` instance across
    threads or scopes must treat the inner containers as read-only.

    Attributes:
        splits: ``.mat`` file count per split that is present on disk.
        subdirs_by_split: Sorted unique names of immediate subdirectories
            grouped by split (Reynolds-number folders for class-2, scenario
            names like ``DNS_turbulence``/``cylinder`` for class-1).
        total_bytes: Sum of the size of every regular file under ``root``.
        mat_files: Number of ``.mat`` files across all splits.
        extra_files: Number of non-``.mat`` regular files (excludes hidden).
    """

    splits: dict[str, int] = field(default_factory=dict)
    subdirs_by_split: dict[str, list[str]] = field(default_factory=dict)
    total_bytes: int = 0
    mat_files: int = 0
    extra_files: int = 0


def _has_hidden_component(path: Path, root: Path) -> bool:
    """Return whether ``path`` lies inside any hidden directory under ``root``.

    Args:
        path: Candidate path to test.
        root: Base path; only components below ``root`` are considered.

    Returns:
        bool: ``True`` if any component of ``path`` (relative to ``root``)
            starts with ``.``.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return any(part.startswith(".") for part in rel.parts)


def _scan_split(split_dir: Path) -> tuple[int, list[str], int, int]:
    """Walk a split directory and gather counts.

    Hidden entries are skipped at every depth: a file like
    ``train/scene/.cache/x.mat`` is treated as hidden because one of its
    ancestors starts with ``.``, not just files whose own name is dotted.

    Args:
        split_dir: Directory for a single split (``train``/``val``/...).

    Returns:
        tuple: ``(mat_count, subdir_names, extra_count, total_bytes)``.
    """
    mat_count = 0
    extra_count = 0
    total_bytes = 0
    subdir_names: set[str] = set()

    for entry in split_dir.iterdir():
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            subdir_names.add(entry.name)
            for nested in entry.rglob("*"):
                if not nested.is_file():
                    continue
                if _has_hidden_component(nested, split_dir):
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

    return mat_count, sorted(subdir_names), extra_count, total_bytes


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
    subdirs_by_split: dict[str, list[str]] = {}
    total_bytes = 0
    mat_files = 0
    extra_files = 0

    for split in _SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        mat_count, subdirs, extra_count, split_bytes = _scan_split(split_dir)
        splits[split] = mat_count
        subdirs_by_split[split] = subdirs
        mat_files += mat_count
        extra_files += extra_count
        total_bytes += split_bytes

    return LayoutSummary(
        splits=splits,
        subdirs_by_split=subdirs_by_split,
        total_bytes=total_bytes,
        mat_files=mat_files,
        extra_files=extra_files,
    )
