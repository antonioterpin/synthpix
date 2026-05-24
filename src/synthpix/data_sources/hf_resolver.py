"""Resolve ``hf://`` dataset URIs to local file paths.

The resolver materializes a Hugging Face Hub dataset repository into a local
cache directory and returns the list of files inside it. It is consumed
transparently by :class:`synthpix.data_sources.base.FileDataSource` so that
any concrete file-based data source (``.mat``, ``.h5``, ``.npy``) can accept
``hf://`` URIs alongside regular local paths.

URI grammar::

    hf://<owner>/<name>
    hf://<owner>/<name>@<revision>
    hf://<owner>/<name>:<subpath>
    hf://<owner>/<name>@<revision>:<subpath>

The resolver always pulls the full repository (so transfers stay resumable);
``<subpath>`` only narrows the returned file list. Caches live under
``cache_dir`` if given, otherwise ``SYNTHPIX_HF_CACHE`` if set, otherwise
``~/.cache/synthpix/hf``. Each revision of each repo lives under its own
``<owner>/<name>@<revision>/`` subdirectory.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from synthpix.hf.pull import pull_dataset
from synthpix.utils import SYNTHPIX_SCOPE, get_logger

logger = get_logger(__name__, scope=SYNTHPIX_SCOPE)

_HF_SCHEME = "hf://"
_DEFAULT_REVISION = "main"
_METADATA_FILES: frozenset[str] = frozenset({"README.md"})

# Strict URI shape:
#   owner/name (no slashes, no @ or : inside)
#   optional @revision (non-empty, no ':' inside)
#   optional :subpath (non-empty)
_URI_RE = re.compile(
    r"^(?P<owner>[^/@:]+)/(?P<name>[^/@:]+)"
    r"(?:@(?P<revision>[^:]+))?"
    r"(?::(?P<subpath>.+))?$"
)


def _sanitize_for_path(component: str, *, what: str, spec: str) -> str:
    """Reject path-traversal payloads and produce a fs-safe directory name.

    ``owner``/``name`` are already filtered by ``_URI_RE`` (no slashes,
    ``@``, or ``:``), but we still defensively reject ``.``/``..`` so a URI
    like ``hf://../bad`` (which the regex would let through as
    ``owner="..", name="bad"``) cannot escape the cache root.

    ``revision`` may legitimately contain slashes (``refs/pr/123``). We
    rewrite slashes to ``__`` for the directory name only — the original
    string is still passed verbatim to :func:`pull_dataset`.

    Args:
        component: The raw URI component (owner, name, or revision).
        what: Human-readable name for the component, used in errors.
        spec: Original URI, included verbatim in error messages.

    Returns:
        str: A version of ``component`` that is safe to splice into a
            filesystem path under the cache root.

    Raises:
        ValueError: If ``component`` is empty, equal to ``.``/``..``, or
            (for owner/name) contains a path separator.
    """
    if not component:
        raise ValueError(f"invalid hf:// URI: empty {what} in {spec!r}")
    # Block any ``..`` segment anywhere in the string, not just standalone.
    normalized = component.replace("\\", "/")
    segments = [seg for seg in normalized.split("/") if seg]
    if any(seg in {".", ".."} for seg in segments) or not segments:
        raise ValueError(
            f"invalid hf:// URI: {what} cannot be '.' or '..' in {spec!r}"
        )
    if what == "revision":
        # Refs like ``refs/pr/123`` are legal upstream — flatten for the
        # cache dir but keep the original for the network call.
        return normalized.replace("/", "__")
    if "/" in normalized:
        raise ValueError(
            f"invalid hf:// URI: {what} must not contain '/' in {spec!r}"
        )
    return normalized


@dataclass(frozen=True)
class _ParsedURI:
    """Parsed components of an ``hf://`` URI.

    Attributes:
        owner: Repository owner / org.
        name: Repository name.
        revision: Git ref to pull; defaults to ``"main"``.
        subpath: Sub-directory within the repo to narrow the result to, or
            ``None`` when no subpath was given.
    """

    owner: str
    name: str
    revision: str
    subpath: str | None

    @property
    def repo_id(self) -> str:
        """Return the ``owner/name`` slug consumed by ``huggingface_hub``."""
        return f"{self.owner}/{self.name}"


def _parse_uri(spec: str) -> _ParsedURI:
    """Parse an ``hf://`` URI into its components.

    Args:
        spec: The full URI, including the ``hf://`` prefix.

    Returns:
        _ParsedURI: The parsed URI components, with ``revision`` defaulted
            to ``"main"`` when the URI did not specify one.

    Raises:
        ValueError: If ``spec`` is malformed. The message names the
            offending component (scheme, owner, name, revision, or subpath).
    """
    if not isinstance(spec, str) or not spec.startswith(_HF_SCHEME):
        raise ValueError(
            f"invalid hf:// URI: missing '{_HF_SCHEME}' scheme in {spec!r}"
        )

    body = spec[len(_HF_SCHEME) :]
    if not body:
        raise ValueError("invalid hf:// URI: missing owner and repo name")

    if "/" not in body:
        raise ValueError(
            f"invalid hf:// URI: missing repo name in {spec!r} "
            "(expected hf://<owner>/<name>)"
        )

    # Pre-flight: catch dangling separators so we can name the broken piece.
    if body.endswith("@"):
        raise ValueError(
            f"invalid hf:// URI: empty revision in {spec!r} "
            "(drop the '@' or supply a branch/tag/sha)"
        )
    if body.endswith(":"):
        raise ValueError(f"invalid hf:// URI: empty subpath in {spec!r}")

    match = _URI_RE.match(body)
    if match is None:
        raise ValueError(f"invalid hf:// URI: malformed components in {spec!r}")

    owner = match.group("owner")
    name = match.group("name")
    revision = match.group("revision")
    subpath = match.group("subpath")

    if not owner:
        raise ValueError(f"invalid hf:// URI: missing owner in {spec!r}")
    if not name:
        raise ValueError(f"invalid hf:// URI: missing repo name in {spec!r}")

    if revision is not None and not revision.strip():
        raise ValueError(
            f"invalid hf:// URI: empty revision in {spec!r} "
            "(drop the '@' or supply a branch/tag/sha)"
        )

    if subpath is not None:
        if not subpath.strip():
            raise ValueError(f"invalid hf:// URI: empty subpath in {spec!r}")
        # Reject parent-directory traversal anywhere in the subpath.
        parts = [p for p in subpath.replace("\\", "/").split("/") if p]
        if any(p == ".." for p in parts):
            raise ValueError(
                f"invalid hf:// URI: subpath cannot contain '..' in {spec!r}"
            )
        if subpath.startswith("/"):
            raise ValueError(
                f"invalid hf:// URI: subpath must be relative in {spec!r}"
            )

    return _ParsedURI(
        owner=owner,
        name=name,
        revision=revision if revision is not None else _DEFAULT_REVISION,
        subpath=subpath,
    )


def _resolve_cache_root(cache_dir: Path | None) -> Path:
    """Resolve the on-disk cache root following the documented precedence.

    The order is: explicit ``cache_dir`` argument, ``SYNTHPIX_HF_CACHE``
    environment variable, then ``~/.cache/synthpix/hf``.

    Args:
        cache_dir: Caller-supplied cache directory, or ``None`` to fall
            back to environment / default.

    Returns:
        Path: The absolute, user-expanded cache root.
    """
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()

    env_cache = os.environ.get("SYNTHPIX_HF_CACHE")
    if env_cache:
        return Path(env_cache).expanduser().resolve()

    return (Path.home() / ".cache" / "synthpix" / "hf").resolve()


def _is_metadata_file(path: Path, repo_root: Path) -> bool:
    """Return True if ``path`` should be excluded from the result list.

    Files in any hidden directory (e.g. ``.git/config``, ``.cache/x``) are
    excluded, not just files whose own name starts with ``.``: VCS / admin
    payloads that the Hub sometimes mirrors must not leak into the consumer
    file list. ``README.md`` is excluded only at the repo root because the
    dataset card is metadata; nested ``README.md`` files inside a sub-folder
    are kept.

    Args:
        path: Candidate file path.
        repo_root: Absolute repo cache root used to detect "at the root".

    Returns:
        bool: True when the file is metadata and should be skipped.
    """
    try:
        rel = path.resolve().relative_to(repo_root)
    except ValueError:
        rel = Path(path.name)
    if any(part.startswith(".") for part in rel.parts):
        return True
    name = path.name
    if name in _METADATA_FILES and path.parent.resolve() == repo_root:
        return True
    return False


def _enumerate_files(repo_root: Path, subpath: str | None) -> list[str]:
    """Walk ``repo_root`` (optionally narrowed by ``subpath``) for data files.

    Args:
        repo_root: Absolute path of the freshly pulled repo cache directory.
        subpath: Sub-directory within ``repo_root`` to restrict to, or
            ``None`` to walk the whole tree.

    Returns:
        list[str]: Sorted absolute file paths, with hidden files and the
            root ``README.md`` excluded.

    Raises:
        FileNotFoundError: If ``subpath`` was given but the resolved
            directory does not exist after the pull.
    """
    walk_root = repo_root if subpath is None else repo_root / subpath
    if subpath is not None and not walk_root.exists():
        raise FileNotFoundError(
            f"hf:// subpath does not exist after pull: {walk_root} "
            f"(repo root: {repo_root})"
        )

    results: list[str] = []
    for path in walk_root.rglob("*"):
        if not path.is_file():
            continue
        if _is_metadata_file(path, repo_root):
            continue
        results.append(str(path.resolve()))
    results.sort()
    return results


def _pull_repo(
    spec: str,
    *,
    cache_dir: Path | None,
    token: str | None,
) -> tuple[Path, _ParsedURI]:
    """Pull the URI's repo into the cache and return (repo_root, parsed).

    Args:
        spec: The ``hf://`` URI to pull.
        cache_dir: Optional explicit cache root.
        token: Optional Hugging Face token forwarded to ``pull_dataset``.

    Returns:
        tuple[Path, _ParsedURI]: The absolute cache directory for this
            repo+revision and the parsed URI components.
    """
    parsed = _parse_uri(spec)
    cache_root = _resolve_cache_root(cache_dir)

    # Sanitize owner/name/revision *for the filesystem path* — keep the
    # original revision string for the network call so legitimate refs
    # like ``refs/pr/123`` still work upstream.
    safe_owner = _sanitize_for_path(parsed.owner, what="owner", spec=spec)
    safe_name = _sanitize_for_path(parsed.name, what="name", spec=spec)
    safe_revision = _sanitize_for_path(
        parsed.revision, what="revision", spec=spec
    )

    local_dir = cache_root / safe_owner / f"{safe_name}@{safe_revision}"
    logger.info(f"Resolving {spec} via cache {local_dir}")

    pull_dataset(
        parsed.repo_id,
        local_dir,
        revision=parsed.revision,
        token=token,
    )
    return local_dir.resolve(), parsed


def resolve_to_directory(
    spec: str,
    *,
    cache_dir: Path | None = None,
    token: str | None = None,
) -> Path:
    """Pull an ``hf://`` URI and return the directory to walk.

    This is the entry point consumed by :class:`FileDataSource`: it pulls
    the referenced dataset (idempotent, resumable) and returns the local
    path the caller can then glob exactly as if it had been a regular
    local directory. The ``:subpath`` qualifier — when present — is applied
    here so the returned path is already narrowed.

    Args:
        spec: The ``hf://`` URI to resolve.
        cache_dir: Optional explicit cache root. Falls back to
            ``SYNTHPIX_HF_CACHE`` then ``~/.cache/synthpix/hf``.
        token: Optional Hugging Face token, forwarded to
            :func:`synthpix.hf.pull.pull_dataset`.

    Returns:
        Path: Absolute path to the resolved directory (cache root + repo
            + revision, optionally narrowed by ``:subpath``).

    Raises:
        FileNotFoundError: If the URI specifies a ``:subpath`` that does
            not exist in the pulled tree.
        ValueError: If the resolved ``:subpath`` would escape the cache
            root (defensive check after symlink resolution).
    """
    repo_root, parsed = _pull_repo(spec, cache_dir=cache_dir, token=token)

    if parsed.subpath is None:
        return repo_root

    narrowed = (repo_root / parsed.subpath).resolve()
    if not narrowed.exists():
        raise FileNotFoundError(
            f"hf:// subpath does not exist after pull: {narrowed} "
            f"(repo root: {repo_root})"
        )
    # Defensive: ensure the resolved subpath did not escape the cache root.
    try:
        narrowed.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(
            f"invalid hf:// URI: subpath escapes the cache root in {spec!r}"
        ) from exc
    return narrowed


def resolve(
    spec: str,
    *,
    cache_dir: Path | None = None,
    token: str | None = None,
) -> list[str]:
    """Resolve an ``hf://`` dataset URI to a list of local file paths.

    Pulls the referenced HF Hub dataset (idempotent, resumable) into a
    cache directory and returns the resolved list of file paths.

    :class:`FileDataSource` does *not* call this helper — it uses
    :func:`resolve_to_directory` so the subclass-specific ``_file_pattern``
    glob still applies. ``resolve`` stays for programmatic callers who want
    the eagerly-enumerated file list (and the explicit metadata filter).

    Args:
        spec: The ``hf://`` URI to resolve. See the module docstring for
            the supported grammar.
        cache_dir: Optional explicit cache root.
        token: Optional Hugging Face token forwarded to ``pull_dataset``.

    Returns:
        list[str]: Sorted absolute paths of every data file under the
            resolved repo cache (narrowed by the URI subpath when given).
            ``ValueError`` is propagated from :func:`_parse_uri` if ``spec``
            is malformed, and ``FileNotFoundError`` from
            :func:`_enumerate_files` if the requested subpath is missing
            after the pull.
    """
    repo_root, parsed = _pull_repo(spec, cache_dir=cache_dir, token=token)
    files = _enumerate_files(repo_root, parsed.subpath)

    logger.info(f"Resolved {spec} -> {len(files)} files")
    return files
