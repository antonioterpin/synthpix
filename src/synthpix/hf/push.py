"""Upload synthpix-hosted PIV datasets to the Hugging Face Hub.

The public entry point is :func:`push_dataset`. It wraps
``huggingface_hub.HfApi.upload_folder`` with a small set of guarantees
beyond the upstream call:

* a default-private safety gate (``allow_public`` must be explicitly
  set before a public repo is created);
* lazy import of ``huggingface_hub`` so the package keeps working
  without the ``[hf]`` extra installed;
* token resolution via :func:`synthpix.hf.auth.resolve_token`;
* optional dataset-card generation right before the upload;
* ``hf_transfer`` acceleration when available via
  :func:`synthpix.hf._transfer.enable_hf_transfer`;
* a ``dry_run`` mode that compares the local file set against the
  files already in the remote repo.

Tokens are never logged.
"""

from __future__ import annotations

import importlib
import re
from fnmatch import fnmatchcase
from pathlib import Path

from synthpix.hf._transfer import enable_hf_transfer
from synthpix.hf.auth import resolve_token
from synthpix.hf.card import DatasetCardMeta, make_dataset_card
from synthpix.hf.layout import inspect_local_layout
from synthpix.utils import SYNTHPIX_SCOPE, get_logger

logger = get_logger(__name__, scope=SYNTHPIX_SCOPE)

_DEFAULT_INCLUDE: tuple[str, ...] = (
    "train/**",
    "val/**",
    "test/**",
    "tune/**",
    "splits/**",
    "README.md",
)
_DEFAULT_IGNORE: tuple[str, ...] = (
    "raw_*/**",
    "packed_*/**",
    "**/.DS_Store",
    "**/__pycache__/**",
    "**/.git/**",
)

_REPO_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$"
)

_PUBLIC_GATE_MESSAGE = (
    "Refusing to create a public HF Hub dataset without --allow-public. "
    "The PIV class-1 source is research-only / all rights reserved; "
    "redistribution under a public license is not permitted by default. "
    "Pass allow_public=True (CLI: --allow-public) to override."
)


def _validate_local_dir(local_dir: Path) -> None:
    if not local_dir.exists():
        raise ValueError(f"local_dir does not exist: {local_dir}")
    if not local_dir.is_dir():
        raise ValueError(f"local_dir is not a directory: {local_dir}")


def _validate_repo_id(repo_id: str) -> None:
    if not _REPO_ID_RE.match(repo_id):
        raise ValueError(
            f"repo_id must look like 'owner/name', got: {repo_id!r}"
        )


def _maybe_write_card(local_dir: Path, card_meta: DatasetCardMeta) -> None:
    layout = inspect_local_layout(local_dir)
    rendered = make_dataset_card(card_meta, layout)
    readme_path = local_dir / "README.md"
    if readme_path.exists():
        try:
            existing = readme_path.read_text()
        except OSError:
            existing = None
        if existing is not None and existing != rendered:
            logger.warning("overwriting existing README.md")
    readme_path.write_text(rendered)


def _iter_relative_files(local_dir: Path) -> list[str]:
    """Return repo-relative POSIX paths for every regular file under root.

    Args:
        local_dir: Directory whose tree is walked.

    Returns:
        list[str]: Repo-relative POSIX paths for every regular file.
    """
    files: list[str] = []
    for entry in local_dir.rglob("*"):
        if entry.is_file():
            files.append(entry.relative_to(local_dir).as_posix())
    return files


def _matches_any(rel_path: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        # Normalize ``**`` to ``*`` for ``fnmatch`` so directory-globs match.
        # ``fnmatch`` does not understand ``**`` natively; for our purposes
        # ``train/**`` should match ``train/anything/deep/file.mat``.
        if fnmatchcase(rel_path, pattern):
            return True
        # Translate ``a/**`` -> match anything below ``a/``.
        if "**" in pattern:
            translated = pattern.replace("/**", "/*").replace("**", "*")
            if fnmatchcase(rel_path, translated):
                return True
            # Also match arbitrary nesting: turn ``a/**`` into prefix match.
            prefix = pattern.split("/**", 1)[0]
            if pattern.endswith("/**") and (
                rel_path == prefix or rel_path.startswith(prefix + "/")
            ):
                return True
    return False


def _filter_local_files(
    local_dir: Path,
    include_globs: tuple[str, ...],
    ignore_globs: tuple[str, ...],
) -> list[str]:
    selected: list[str] = []
    for rel in _iter_relative_files(local_dir):
        if not _matches_any(rel, include_globs):
            continue
        if _matches_any(rel, ignore_globs):
            continue
        selected.append(rel)
    return sorted(selected)


def _dry_run_summary(
    local_dir: Path,
    repo_id: str,
    revision: str,
    include_globs: tuple[str, ...],
    ignore_globs: tuple[str, ...],
    token: str | None,
) -> str:
    """Print a per-file dry-run plan; return ``"dry-run"`` for the caller.

    Args:
        local_dir: Local directory used for the diff.
        repo_id: Remote ``<owner>/<name>`` identifier.
        revision: Git-style revision queried on the Hub.
        include_globs: ``allow_patterns`` for the diff.
        ignore_globs: ``ignore_patterns`` for the diff.
        token: Resolved Hugging Face token, if any.

    Returns:
        str: The literal ``"dry-run"`` sentinel.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
    """
    try:
        hub = importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for synthpix.hf.push_dataset; "
            "install with `pip install synthpix[hf]`"
        ) from exc

    try:
        errors_mod = importlib.import_module("huggingface_hub.errors")
        repo_not_found = errors_mod.RepositoryNotFoundError
    except (ImportError, AttributeError):
        try:
            utils_mod = importlib.import_module("huggingface_hub.utils")
            repo_not_found = utils_mod.RepositoryNotFoundError
        except (ImportError, AttributeError):
            repo_not_found = Exception  # type: ignore[assignment]

    api = hub.HfApi(token=token)
    try:
        remote_files = api.list_repo_files(
            repo_id, repo_type="dataset", revision=revision
        )
    except repo_not_found:
        remote_files = []
    remote_set = set(remote_files)

    local_files = _filter_local_files(local_dir, include_globs, ignore_globs)
    new = [f for f in local_files if f not in remote_set]
    unchanged = [f for f in local_files if f in remote_set]

    print(f"DRY RUN: {repo_id}@{revision}")
    print(f"  local_dir: {local_dir}")
    print(f"  new files: {len(new)}")
    for f in new:
        print(f"    + {f}")
    print(f"  unchanged files: {len(unchanged)}")
    for f in unchanged:
        print(f"    = {f}")
    return "dry-run"


def push_dataset(
    local_dir: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    allow_public: bool = False,
    token: str | None = None,
    revision: str = "main",
    commit_message: str | None = None,
    card_meta: DatasetCardMeta | None = None,
    include_globs: tuple[str, ...] = _DEFAULT_INCLUDE,
    ignore_globs: tuple[str, ...] = _DEFAULT_IGNORE,
    max_workers: int = 8,
    dry_run: bool = False,
) -> str:
    """Upload ``local_dir`` to an HF Hub dataset repository.

    The push is private by default. Creating a public repo requires
    ``allow_public=True`` (the CLI surfaces this as ``--allow-public``)
    so that ad-hoc redistribution of research-only sources is gated by a
    deliberate flag.

    Args:
        local_dir: Local directory whose contents are uploaded.
        repo_id: ``<owner>/<name>`` identifier on the Hub.
        private: Whether to create/keep the repo private. Defaults to
            ``True``.
        allow_public: Companion flag for ``private=False``. Must be set
            explicitly; otherwise the public push is refused.
        token: Explicit Hugging Face token. When ``None``, the standard
            resolution chain (env vars, cached token) is used.
        revision: Git-style revision (branch, tag, or commit) to upload to.
        commit_message: Custom commit message; defaults to
            ``"Upload via synthpix-hf"``.
        card_meta: Optional dataset-card metadata. When set, the README
            is (re)generated under ``local_dir`` right before the upload.
        include_globs: ``allow_patterns`` for ``upload_folder``.
        ignore_globs: ``ignore_patterns`` for ``upload_folder``.
        max_workers: Parallel upload workers.
        dry_run: When ``True``, compare local and remote file lists, print
            the plan, and return ``"dry-run"`` without touching the Hub.

    Returns:
        str: The commit sha returned by ``upload_folder``, or ``"dry-run"``
            when ``dry_run`` is set.

    Raises:
        PermissionError: When ``private=False`` is requested without
            ``allow_public=True``.
        ValueError: When ``local_dir`` is missing/not a directory,
            ``repo_id`` does not match the ``owner/name`` shape, or
            ``max_workers`` is below one.
        ImportError: If ``huggingface_hub`` is not installed.
    """
    if private is False and allow_public is not True:
        raise PermissionError(_PUBLIC_GATE_MESSAGE)

    local_path = Path(local_dir).expanduser()
    _validate_local_dir(local_path)
    _validate_repo_id(repo_id)
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got: {max_workers}")

    resolved_token = resolve_token(token)

    if card_meta is not None:
        _maybe_write_card(local_path, card_meta)

    logger.info(f"Pushing {local_path} to {repo_id} (private={private})")

    if dry_run:
        return _dry_run_summary(
            local_path,
            repo_id,
            revision,
            include_globs,
            ignore_globs,
            resolved_token,
        )

    try:
        hub = importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for synthpix.hf.push_dataset; "
            "install with `pip install synthpix[hf]`"
        ) from exc

    api = hub.HfApi(token=resolved_token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    # ``HfApi.upload_folder`` has no parallelism knob in huggingface_hub
    # 1.x (the old ``num_workers`` kwarg was removed and never existed on
    # this API). ``max_workers`` is kept on the public signature for API
    # stability but is a no-op here; passing it to ``upload_folder`` would
    # raise ``TypeError`` against the real Hub.
    with enable_hf_transfer():
        commit_info = api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(local_path),
            revision=revision,
            commit_message=commit_message or "Upload via synthpix-hf",
            allow_patterns=list(include_globs),
            ignore_patterns=list(ignore_globs),
        )

    # In huggingface_hub 1.x ``CommitInfo`` subclasses ``str`` and its
    # string value is the *commit URL*, not the sha. Prefer the explicit
    # ``oid`` attribute; only fall back to the string form when it is a
    # plain ``str`` with no ``oid``.
    oid = getattr(commit_info, "oid", None)
    if oid:
        commit_sha = oid
    else:
        commit_sha = str(commit_info)

    uploaded = _filter_local_files(local_path, include_globs, ignore_globs)
    logger.info(f"Pushed {repo_id}@{commit_sha} with {len(uploaded)} files")
    return commit_sha
