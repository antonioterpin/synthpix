"""Hugging Face Hub integration for the ``synthpix`` package.

This subpackage exposes metadata helpers, layout introspection, and the
``pull_dataset`` / ``push_dataset`` transports. ``huggingface_hub`` is
imported lazily inside :mod:`synthpix.hf.auth`, :mod:`synthpix.hf.pull`,
and :mod:`synthpix.hf.push`, so this module remains usable without the
``[hf]`` optional extra installed.
"""

from synthpix.hf.auth import resolve_token
from synthpix.hf.card import DatasetCardMeta, make_dataset_card
from synthpix.hf.layout import LayoutSummary, inspect_local_layout
from synthpix.hf.pull import pull_dataset
from synthpix.hf.push import push_dataset

__all__ = [
    "DatasetCardMeta",
    "LayoutSummary",
    "inspect_local_layout",
    "make_dataset_card",
    "pull_dataset",
    "push_dataset",
    "resolve_token",
]
