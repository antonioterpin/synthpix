"""Hugging Face Hub authentication helpers.

The functions defined here read tokens from a small number of well-known
locations without ever persisting them. ``huggingface_hub`` is imported lazily
so that the rest of the ``synthpix.hf`` package remains importable without the
``[hf]`` extra installed.
"""

from __future__ import annotations

import os

from synthpix.utils import SYNTHPIX_SCOPE, get_logger

logger = get_logger(__name__, scope=SYNTHPIX_SCOPE)


def _cached_token() -> str | None:
    """Read the cached token via ``huggingface_hub`` if it is importable.

    Returns:
        str | None: The cached token, or ``None`` if the library is not
            installed or no token is cached.
    """
    try:
        import huggingface_hub  # noqa: PLC0415
    except ImportError:
        logger.debug(
            "huggingface_hub is not installed; skipping cached token lookup."
        )
        return None

    # huggingface_hub >= 1.0 removed ``HfFolder``; the cached CLI token is
    # now read via the top-level ``get_token`` helper. Guard for older
    # releases (and partially-stubbed installs) without re-raising.
    get_token = getattr(huggingface_hub, "get_token", None)
    if get_token is None:
        logger.debug(
            "huggingface_hub has no get_token(); skipping cached token lookup."
        )
        return None

    token: str | None = get_token()
    if token:
        return token
    return None


def resolve_token(explicit: str | None = None) -> str | None:
    """Resolve a Hugging Face token from the first available source.

    The lookup order is: explicit argument, ``HF_TOKEN``, ``HF_HUB_TOKEN``,
    and finally the cached token managed by ``huggingface_hub``. Empty strings
    are treated as missing values. The function never writes to disk.

    Args:
        explicit: Token passed directly by the caller, if any.

    Returns:
        str | None: The first non-empty token found, or ``None`` if nothing
            could be resolved.
    """
    if explicit:
        return explicit

    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")
    if env_token:
        return env_token

    return _cached_token()
