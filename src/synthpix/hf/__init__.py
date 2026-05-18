"""Hugging Face Hub integration for the ``synthpix`` package.

This subpackage exposes only metadata and layout helpers in PR1. Network
operations (push/pull) are deferred to subsequent PRs. ``huggingface_hub`` is
imported lazily inside :mod:`synthpix.hf.auth`, so this module remains usable
without the ``[hf]`` optional extra installed.
"""

from synthpix.hf.auth import resolve_token
from synthpix.hf.card import DatasetCardMeta, make_dataset_card
from synthpix.hf.layout import LayoutSummary, inspect_local_layout

__all__ = [
    "DatasetCardMeta",
    "LayoutSummary",
    "inspect_local_layout",
    "make_dataset_card",
    "resolve_token",
]
