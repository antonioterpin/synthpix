"""Tests for the shutdown method in GrainSchedulerAdapter.

These tests verify that `GrainSchedulerAdapter.shutdown` correctly closes
the underlying Grain iterator and handles exceptions gracefully.
"""
from unittest.mock import MagicMock

import grain.python as grain
import pytest

from synthpix.data_sources.adapter import GrainSchedulerAdapter


def test_grain_adapter_shutdown_sets_none():
    """Test that shutdown sets the iterator to None."""
    loader = MagicMock(spec=grain.DataLoader)
    mock_iterator = MagicMock()
    loader.__iter__.side_effect = lambda: mock_iterator

    adapter = GrainSchedulerAdapter(loader)
    
    # Verify initial state (iterator created)
    assert adapter.grain_iterator is not None
    
    adapter.shutdown()
    
    # Verify iterator is None
    assert adapter.grain_iterator is None
