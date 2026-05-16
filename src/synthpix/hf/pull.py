"""Download synthpix-hosted PIV datasets from the Hugging Face Hub.

The public entry point is :func:`pull_dataset`, a thin wrapper around
``huggingface_hub.snapshot_download`` that:

* lazy-imports ``huggingface_hub`` so the package keeps working without the
  ``[hf]`` extra installed;
* resolves a token via :func:`synthpix.hf.auth.resolve_token`;
* turns the ``splits`` shorthand into ``allow_patterns`` so callers can ask
  for a subset of a dataset without learning glob syntax;
* enables ``hf_transfer`` acceleration when available via
  :func:`synthpix.hf._transfer.enable_hf_transfer`.

The resulting on-disk tree mirrors the repo and is consumable directly by
the ``.mat`` schedulers without an extra symlink resolution step.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from synthpix.hf._transfer import enable_hf_transfer
from synthpix.hf.auth import resolve_token
from synthpix.utils import SYNTHPIX_SCOPE, get_logger

logger = get_logger(__name__, scope=SYNTHPIX_SCOPE)

_DEFAULT_IGNORE: tuple[str, ...] = (
    "**/.DS_Store",
    "**/__pycache__/**",
    "**/.git/**",
)


def _resolve_allow_patterns(
    splits: tuple[str, ...] | None,
    include_globs: tuple[str, ...] | None,
) -> list[str] | None:
    """Translate the user-facing filters into ``allow_patterns``.

    Explicit ``include_globs`` always win. When only ``splits`` is provided
    we synthesize ``"<split>/**"`` patterns and always append ``README.md``
    so the dataset card travels with the data.

    Args:
        splits: Split names to download, or ``None`` for no split filter.
        include_globs: Explicit allow-list glob patterns. Takes precedence
            over ``splits`` and triggers a warning when both are set.

    Returns:
        list[str] | None: The patterns to pass to
            ``snapshot_download``'s ``allow_patterns``, or ``None`` when
            both inputs are unset (download everything).
    """
    if include_globs is not None:
        if splits is not None:
            logger.warning(
                "Both `splits` and `include_globs` were provided; "
                "ignoring `splits` and honoring the explicit globs."
            )
        return list(include_globs)
    if splits is not None:
        return [f"{name}/**" for name in splits] + ["README.md"]
    return None


def pull_dataset(
    repo_id: str,
    local_dir: str | Path,
    *,
    revision: str = "main",
    token: str | None = None,
    splits: tuple[str, ...] | None = None,
    include_globs: tuple[str, ...] | None = None,
    ignore_globs: tuple[str, ...] = _DEFAULT_IGNORE,
    max_workers: int = 8,
) -> Path:
    """Download an HF Hub dataset repo into ``local_dir``.

    The transfer goes through ``huggingface_hub.snapshot_download`` and is
    therefore resumable and idempotent: an existing ``local_dir`` is reused
    rather than wiped. Symlinks are disabled so the resulting tree mirrors
    the repo and can be fed directly into ``.mat`` schedulers.

    Args:
        repo_id: ``<org>/<dataset>`` identifier on the Hub.
        local_dir: Destination directory; created if it does not exist.
            Tilde expansion is applied.
        revision: Git-style revision (branch, tag, or commit) to download.
        token: Explicit Hugging Face token. When ``None``, the standard
            resolution chain (env vars, cached token) is used. Public
            datasets work without a token.
        splits: Convenience filter for split-style layouts; when set,
            ``allow_patterns`` becomes ``("<split>/**", ..., "README.md")``.
            Ignored if ``include_globs`` is also given.
        include_globs: Explicit glob allow-list; takes precedence over
            ``splits``.
        ignore_globs: Glob deny-list applied after ``include_globs``.
        max_workers: Parallel download workers.

    Returns:
        Path: The resolved local path, equivalent to
            ``Path(local_dir).expanduser().resolve()``.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
        NotADirectoryError: If ``local_dir`` exists and is a regular file.
    """
    target = Path(local_dir).expanduser()
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(
            f"local_dir exists and is not a directory: {target}"
        )

    try:
        hub = importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for synthpix.hf.pull_dataset; "
            "install with `pip install synthpix[hf]`"
        ) from exc
    resolved_token = resolve_token(token)
    allow_patterns = _resolve_allow_patterns(splits, include_globs)

    with enable_hf_transfer():
        hub.snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=str(target),
            token=resolved_token,
            allow_patterns=allow_patterns,
            ignore_patterns=list(ignore_globs),
            max_workers=max_workers,
        )

    return target.resolve()
