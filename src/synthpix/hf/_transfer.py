"""Internal helpers to toggle ``hf_transfer`` acceleration.

``hf_transfer`` is a Rust-backed download accelerator. When the library is
importable we opt in by setting ``HF_HUB_ENABLE_HF_TRANSFER=1`` for the
duration of a block. The probe uses :func:`importlib.util.find_spec` so we
never pull ``hf_transfer`` into the module graph eagerly.

The toggle is reentrant: nested ``enable_hf_transfer()`` blocks reference-
count the env mutation under a process-local lock, so overlapping callers
cannot leak the flag or restore it in the wrong order.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import threading
from collections.abc import Iterator

from synthpix.utils import SYNTHPIX_SCOPE, get_logger

logger = get_logger(__name__, scope=SYNTHPIX_SCOPE)

_ENV_FLAG = "HF_HUB_ENABLE_HF_TRANSFER"

# Reentrancy + thread-safety state. ``_state_lock`` serializes mutations of
# the env flag across overlapping ``enable_hf_transfer`` contexts in the same
# process. ``_active_count`` tracks how many contexts are currently inside
# the "set" branch so the outermost exit is the one that restores the prior
# value. ``_saved_value`` holds whatever ``os.environ`` had when the first
# context entered (``None`` means "absent").
_state_lock = threading.Lock()
_active_count = 0
_saved_value: str | None = None
_saved_present = False


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
    restored when the outermost enclosing context exits. Nested calls in the
    same process share the same saved value, so an interior exit does not
    pop the flag out from under a still-active outer context.

    When ``hf_transfer`` is not installed the helper logs a one-line warning
    and *clears* any stale ``HF_HUB_ENABLE_HF_TRANSFER=1`` inherited from
    the environment for the duration of the block. This guarantees the
    documented "transparently fall back to the standard downloader"
    behavior: ``huggingface_hub`` cannot then attempt the accelerated path
    and fail with a confusing ``ImportError`` from inside the download.

    Yields:
        None: The context payload; callers do not need the value.
    """
    # Module-level state is intentional: the saved env value must survive
    # across nested context entries/exits in the same process.
    global _active_count, _saved_value, _saved_present  # noqa: PLW0603

    if not _hf_transfer_available():
        logger.warning(
            "hf_transfer is not installed; falling back to the standard "
            "huggingface_hub downloader. Install with "
            "`pip install synthpix[hf]` for accelerated transfers."
        )
        with _state_lock:
            previous = os.environ.pop(_ENV_FLAG, None)
        try:
            yield
        finally:
            with _state_lock:
                if previous is not None:
                    os.environ[_ENV_FLAG] = previous
        return

    with _state_lock:
        if _active_count == 0:
            _saved_present = _ENV_FLAG in os.environ
            _saved_value = os.environ.get(_ENV_FLAG)
            os.environ[_ENV_FLAG] = "1"
        _active_count += 1

    try:
        yield
    finally:
        with _state_lock:
            _active_count -= 1
            if _active_count == 0:
                if _saved_present:
                    os.environ[_ENV_FLAG] = _saved_value or ""
                else:
                    os.environ.pop(_ENV_FLAG, None)
                _saved_value = None
                _saved_present = False
