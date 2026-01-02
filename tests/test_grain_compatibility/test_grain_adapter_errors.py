import pytest
import numpy as np
import grain.python as grain
from synthpix.data_sources.adapter import GrainSchedulerAdapter, GrainEpisodicAdapter
from synthpix.data_sources.episodic import EpisodicDataSource

def test_grain_adapter_empty_loader_error():
    """Test that empty loader raises StopIteration on shape check."""
    
    class MockLoader(grain.DataLoader):
        def __init__(self):
            pass 
        def __iter__(self):
            return iter([])
    
    adapter = GrainSchedulerAdapter(MockLoader())
    
    with pytest.raises(StopIteration):
        adapter.get_flow_fields_shape()

def test_grain_adapter_missing_images_error(tmp_path):
    """Test that missing images raise KeyError when include_images is True."""
    batch = {
        "flow_fields": np.zeros((1, 64, 64, 2))
    }
    
    class MockLoader(grain.DataLoader):
        def __init__(self):
            pass
        def __iter__(self):
            yield batch
        @property
        def _data_source(self):
            class MockDS:
                include_images = True
            return MockDS()

    adapter = GrainSchedulerAdapter(MockLoader())
    
    with pytest.raises(KeyError, match="Images expected but not found"):
        adapter.get_batch(1)

def test_grain_episodic_missing_metadata_error():
    """Test that missing _timestep metadata raises KeyError in next_episode."""
    batch = {
        "flow_fields": np.zeros((1, 64, 64, 2))
    }
    
    class MockLoader(grain.DataLoader):
        def __init__(self):
            pass
        def __iter__(self):
            yield batch
        @property
        def _data_source(self):
            class MockDS(EpisodicDataSource):
                 def __init__(self):
                     pass
                 @property
                 def episode_length(self):
                     return 10
            return MockDS()

    adapter = GrainEpisodicAdapter(MockLoader())
    
    with pytest.raises(KeyError, match="Batch missing required '_timestep' metadata"):
        adapter.next_episode()
