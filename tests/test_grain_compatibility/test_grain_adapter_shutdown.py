"""Tests for the shutdown method in GrainSchedulerAdapter.

These tests verify that `GrainSchedulerAdapter.shutdown` correctly closes
the underlying Grain iterator and handles exceptions gracefully.
"""
from unittest.mock import MagicMock

import grain.python as grain
import pytest

from synthpix.data_sources.adapter import GrainSchedulerAdapter


def test_grain_adapter_shutdown_calls_close():
    """Test that shutdown calls close on the iterator."""
    loader = MagicMock(spec=grain.DataLoader)
    mock_iterator = MagicMock()
    loader.__iter__.side_effect = lambda: mock_iterator

    adapter = GrainSchedulerAdapter(loader)
    
    # Verify initial state (iterator created)
    loader.__iter__.assert_called_once()
    
    adapter.shutdown()
    
    # Verify close was called
    mock_iterator.close.assert_called_once()


def test_grain_adapter_shutdown_suppresses_exception(caplog):
    """Test that shutdown suppresses exceptions raised by close and logs warning."""
    loader = MagicMock(spec=grain.DataLoader)
    mock_iterator = MagicMock()
    # Configure close to raise an exception
    mock_iterator.close.side_effect = Exception("Close failed")
    loader.__iter__.side_effect = lambda: mock_iterator

    adapter = GrainSchedulerAdapter(loader)
    
    # Should not raise exception
    with caplog.at_level("WARNING"):
        adapter.shutdown()
    
    # Verify close was still called
    mock_iterator.close.assert_called_once()
    
    # Verify warning was logged
    assert "Failed to close iterator: Close failed" in caplog.text
