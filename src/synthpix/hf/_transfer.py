"""Internal helpers to toggle ``hf_transfer`` acceleration.

``hf_transfer`` is a Rust-backed download accelerator shipped via the
``huggingface_hub[hf_transfer]`` extra. When the library is importable we
opt in by setting ``HF_HUB_ENABLE_HF_TRANSFER=1`` for the duration of a
block. The probe uses :func:`importlib.util.find_spec` so we never pull
``hf_transfer`` into the module graph eagerly.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
from collections.abc import Iterator

from synthpix.utils import SYNTHPIX_SCOPE, get_logger

logger = get_logger(__name__, scope=SYNTHPIX_SCOPE)

_ENV_FLAG = "HF_HUB_ENABLE_HF_TRANSFER"


def _hf_transfer_available() -> bool:
    """Return whether ``hf_transfer`` is importable without importing it.

    Returns:
        bool: ``True`` when :func:`importlib.util.find_spec` locates the
            ``hf_transfer`` package, otherwise ``False``.
    """
    return importlib.util.find_spec("hf_transfer") is not None


@contextlib.contextmanager
def enable_hf_transfer() -> Iterator[None]:
    """Toggle ``HF_HUB_ENABLE_HF_TRANSFER`` for the duration of the block.

    The previous value of the environment variable (or its absence) is
    restored on exit. If ``hf_transfer`` is not installed, the helper logs a
    one-line warning and yields without mutating the environment, so the
    caller transparently falls back to ``huggingface_hub``'s plain
    downloader.

    Yields:
        None: The context payload; callers do not need the value.
    """
    if not _hf_transfer_available():
        logger.warning(
            "hf_transfer is not installed; falling back to the standard "
            "huggingface_hub downloader. Install with `pip install "
            "synthpix[hf]` for accelerated transfers."
        )
        yield
        return

    previous = os.environ.get(_ENV_FLAG)
    os.environ[_ENV_FLAG] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_ENV_FLAG, None)
        else:
            os.environ[_ENV_FLAG] = previous
