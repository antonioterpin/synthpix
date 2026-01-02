from unittest.mock import MagicMock

import grain.python as grain
import numpy as np
import pytest

from synthpix.data_sources.adapter import (GrainEpisodicAdapter,
                                           GrainSchedulerAdapter)
from synthpix.data_sources.episodic import EpisodicDataSource


def test_grain_adapter_empty_loader_error():
    """Test that empty loader raises StopIteration on shape check."""
    loader = MagicMock(spec=grain.DataLoader)
    loader.__iter__.return_value = iter([])

    adapter = GrainSchedulerAdapter(loader)

    with pytest.raises(StopIteration):
        adapter.get_flow_fields_shape()


def test_grain_adapter_missing_images_error(tmp_path):
    """Test that missing images raise KeyError when include_images is True."""
    batch = {"flow_fields": np.zeros((1, 64, 64, 2))}

    loader = MagicMock(spec=grain.DataLoader)
    loader.__iter__.return_value = iter([batch])

    # Mock data source with include_images = True
    mock_ds = MagicMock()
    mock_ds.include_images = True
    loader._data_source = mock_ds

    adapter = GrainSchedulerAdapter(loader)

    with pytest.raises(KeyError, match="Images expected but not found"):
        adapter.get_batch(1)


def test_grain_episodic_missing_metadata_error():
    """Test that missing _timestep metadata raises KeyError in next_episode."""
    batch = {"flow_fields": np.zeros((1, 64, 64, 2))}

    loader = MagicMock(spec=grain.DataLoader)
    loader.__iter__.return_value = iter([batch])

    # We need a MockDS that is an instance of EpisodicDataSource
    # To avoid complex __init__ logic of EpisodicDataSource, we can use
    # MagicMock with spec
    mock_eds = MagicMock(spec=EpisodicDataSource)
    mock_eds.episode_length = 10
    mock_eds.include_images = False

    loader._data_source = mock_eds

    # GrainEpisodicAdapter checks isinstance(loader._data_source, EpisodicDataSource)
    # MagicMock(spec=EpisodicDataSource) passes this check.

    adapter = GrainEpisodicAdapter(loader)

    # Manually trigger next_episode which should fail because batch lacks _timestep
    # We set _current_timestep to something > -1 to trigger the skip loop
    adapter._current_timestep = 0

    with pytest.raises(
        KeyError, match="Batch missing required '_timestep' metadata"
    ):
        adapter.next_episode()
