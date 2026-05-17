"""Grain DataSources for synthpix.

``hf_resolver`` is intentionally *not* re-exported eagerly: pulling it in
would force ``huggingface_hub`` (and the rest of the ``[hf]`` chain) into
the module graph for every consumer of the file/MAT/HDF5/Numpy data
sources, defeating the lazy-import goal. Use the explicit
``from synthpix.data_sources.hf_resolver import resolve`` form for
programmatic access; ``FileDataSource`` does the lazy import itself when
it sees an ``hf://`` URI.
"""

from __future__ import annotations

from .base import FileDataSource
from .episodic import EpisodicDataSource
from .hdf5 import HDF5DataSource
from .mat import MATDataSource
from .numpy import NumpyDataSource


def __getattr__(name: str):
    """Lazy back-compat: ``synthpix.data_sources.resolve`` still works.

    Pre-existing callers that imported ``resolve`` from the package root
    keep working; the resolver module (and ``huggingface_hub``) only loads
    on first access.

    Args:
        name: Attribute name being looked up.

    Returns:
        Any: The resolver's ``resolve`` callable when ``name == "resolve"``.

    Raises:
        AttributeError: For any other attribute name.
    """
    if name == "resolve":
        from .hf_resolver import resolve  # noqa: PLC0415

        return resolve
    raise AttributeError(
        f"module 'synthpix.data_sources' has no attribute {name!r}"
    )


__all__ = [
    "EpisodicDataSource",
    "FileDataSource",
    "HDF5DataSource",
    "MATDataSource",
    "NumpyDataSource",
    "resolve",
]
