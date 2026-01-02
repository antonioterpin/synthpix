"""Tests for Grain Adapters."""

import pytest
import numpy as np
import grain.python as grain
from synthpix.data_sources.adapter import GrainSchedulerAdapter, GrainEpisodicAdapter
from synthpix.scheduler.protocol import EpisodeEnd
from synthpix.data_sources import EpisodicDataSource, FileDataSource

class MockDataSource(grain.RandomAccessDataSource):
    """Mock source returning dicts with metadata."""
    def __init__(self, length=10, episode_len=5):
        self._length = length
        self.episode_length = episode_len
        self.include_images = False
        
    def __len__(self):
        return self._length
        
    def __getitem__(self, idx):
        # Simulate episodes: 0,1,2,3,4 | 0,1,2,3,4
        t = idx % self.episode_length
        return {
            "flow_fields": np.zeros((32, 32, 2)),
            "_chunk_id": int(idx // self.episode_length),
            "_timestep": t,
            "_is_last_step": (t == self.episode_length - 1)
        }

class MockFileSource(FileDataSource):
    def __init__(self, files, include_images=False):
        self._file_list = files
        self._include_images = include_images
        
    @property
    def include_images(self):
        return self._include_images

    @include_images.setter
    def include_images(self, value):
        self._include_images = value

    def load_file(self, f): return {}

def test_scheduler_adapter_basic():
    """Test basic non-episodic adapter."""
    ds = MockDataSource(length=10)
    # Batch size 2
    loader = grain.DataLoader(
        data_source=ds,
        sampler=grain.IndexSampler(
            num_records=len(ds), shuffle=False, shard_options=grain.NoSharding(), num_epochs=1
        ), 
        operations=[grain.Batch(batch_size=2)]
    )
    
    adapter = GrainSchedulerAdapter(loader)
    
    # First batch: indices 0, 1 -> t=0, t=1
    batch = adapter.get_batch(2)
    assert batch.flow_fields.shape == (2, 32, 32, 2)
    assert batch.mask.shape == (2,)
    assert np.all(batch.mask)
    
    # Check shape inference (cached)
    shape = adapter.get_flow_fields_shape()
    assert shape == (32, 32, 2)
    
    # Reset
    adapter.reset()
    batch_rst = adapter.get_batch(2)
    assert batch_rst.flow_fields.shape == (2, 32, 32, 2)

from unittest.mock import MagicMock

def test_scheduler_adapter_padding():
    """Test padding when requested batch size is larger than data batch side."""
    # Data returns batch size 2 (from Grain), but we ask for 4.
    ds = MockDataSource(length=2)
    loader = grain.DataLoader(
        data_source=ds,
        sampler=grain.IndexSampler(
            num_records=len(ds), shuffle=False, shard_options=grain.NoSharding(), num_epochs=1
        ),
        operations=[grain.Batch(batch_size=2)]
    )
    adapter = GrainSchedulerAdapter(loader)
    
    batch = adapter.get_batch(4)
    assert batch.flow_fields.shape == (4, 32, 32, 2)
    # First 2 valid, last 2 invalid
    assert np.sum(batch.mask) == 2
    assert batch.mask[0] == True
    assert batch.mask[2] == False # Padded

def test_episodic_adapter_exhaustion():
    """Test EpisodeEnd exception."""
    ds = MockDataSource(length=2, episode_len=2) # 0, 1
    loader = grain.DataLoader(
        data_source=ds, 
        sampler=grain.IndexSampler(
            num_records=len(ds), shuffle=False, shard_options=grain.NoSharding(), num_epochs=1
        ), 
        operations=[grain.Batch(batch_size=1)]
    )
    # Monkey patch source
    loader._data_source = MagicMock(spec=EpisodicDataSource)
    loader._data_source.episode_length = 2
    loader._data_source.include_images = False
    
    adapter = GrainEpisodicAdapter(loader)
    
    _ = adapter.get_batch(1) # t=0
    _ = adapter.get_batch(1) # t=1 (last)
    
    # Next call -> StopIteration from loader -> EpisodeEnd
    with pytest.raises(EpisodeEnd):
        adapter.get_batch(1)

def test_get_flow_fields_shape_empty():
    """Test behavior when loader is empty."""
    # Mock loader directly since grain.DataLoader doesn't like empty sources easily
    loader = MagicMock(spec=grain.DataLoader)
    loader.__iter__.return_value = iter([]) # Empty iterator
    
    adapter = GrainSchedulerAdapter(loader)
    
    with pytest.raises(StopIteration):
        adapter.get_flow_fields_shape()

def test_adapter_shutdown():
    """Test shutdown method."""
    ds = MockDataSource(length=1)
    loader = grain.DataLoader(data_source=ds, sampler=grain.IndexSampler(1, grain.NoSharding()), operations=[])
    adapter = GrainSchedulerAdapter(loader)
    adapter.shutdown() # Should do nothing but exist

def test_adapter_properties():
    """Test include_images and file_list properties."""
    ds = MockFileSource(files=["a"], include_images=True)

    
    # We need to ensure loader._data_source is accessible.
    # Grain wraps it.
    loader = grain.DataLoader(data_source=ds, sampler=grain.IndexSampler(1, grain.NoSharding()), operations=[])
    
    # Grain might not expose _data_source easily in all versions, but our adapter relies on it.
    # If the adapter implementation uses `loader._data_source`, we assume it's there.
    # Ensure our MockDataSource is attached.
    # grain.DataLoader usually sets self._data_source = data_source
    
    adapter = GrainSchedulerAdapter(loader)
    assert adapter.include_images is True
    assert adapter.file_list == ["a"]
    
    # Test setter error
    with pytest.raises(NotImplementedError):
        adapter.file_list = ["b"]

def test_adapter_images_and_files_padding():
    """Test valid handling and padding of images and files."""
    # Source returning images and file
    class ImageSource(grain.RandomAccessDataSource):
        def __len__(self): return 1
        def __getitem__(self, idx):
            return {
                "flow_fields": np.zeros((32, 32, 2)),
                "images1": np.zeros((32, 32, 3)),
                "images2": np.zeros((32, 32, 3)),
                "file": "file_path"
            }
            
    loader = grain.DataLoader(
        data_source=ImageSource(), 
        sampler=grain.IndexSampler(1, grain.NoSharding()), 
        operations=[grain.Batch(batch_size=1)]
    )
    # Mock property inheritance manually if needed, but ImageSource doesn't have it.
    # Adapter checks loader._data_source.include_images.
    # We set it on the source instance.
    loader._data_source.include_images = True
    
    adapter = GrainSchedulerAdapter(loader)
    
    # Request batch 2 (pad 1)
    batch = adapter.get_batch(2)
    assert batch.images1.shape == (2, 32, 32, 3)
    assert batch.images2.shape == (2, 32, 32, 3)
    assert batch.files == ("file_path", "")
    assert batch.mask[1] == False # Padded

def test_episodic_next_episode_logic():
    """Test next_episode advances correctly."""
    # Create a loader that yields:
    # 0: {ts: 0}
    # 1: {ts: 1}
    # 2: {ts: 2} (end of ep)
    # 3: {ts: 0} (next ep)
    
    # Custom iterator mock to strictly control output sequence
    loader = MagicMock(spec=grain.DataLoader)
    
    # Define source properties on the mock loader
    # Using a MagicMock for the data source
    mock_source = MagicMock(spec=EpisodicDataSource) # Use spec ensures checking
    mock_source.episode_length = 3
    mock_source.include_images = False
    
    # Attach to loader
    loader._data_source = mock_source
    
    # Iterator sequence
    batch0 = {"_timestep": np.array([0]), "flow_fields": np.zeros((1, 32, 32, 2))}
    batch1 = {"_timestep": np.array([1]), "flow_fields": np.zeros((1, 32, 32, 2))}
    batch2 = {"_timestep": np.array([2]), "flow_fields": np.zeros((1, 32, 32, 2))}
    batch_next = {"_timestep": np.array([0]), "flow_fields": np.zeros((1, 32, 32, 2))}
    
    # Iterator returns: t0, t1, t2, t0...
    loader.__iter__.return_value = iter([batch0, batch1, batch2, batch_next])
    
    adapter = GrainEpisodicAdapter(loader)
    
    # 1. Get t0
    b = adapter.get_batch(1)
    assert adapter._current_timestep == 0
    assert adapter.steps_remaining() == 2 # 3 - 1
    
    # 2. Skip remaining (t1, t2)
    # logic: while steps_remaining > 0: next(...)
    # It should consume batch1 (t1 -> rem 1), batch2 (t2 -> rem 0 -> break)
    adapter.next_episode()
    
    # 3. Next call should be batch_next (t0)
    # Adapter sets _current_timestep = -1 after skip loop
    assert adapter._current_timestep == -1
    
    b_new = adapter.get_batch(1)
    # b_new is SchedulerData, batch_next is dict
    assert np.array_equal(b_new.flow_fields, batch_next["flow_fields"])
    assert adapter._current_timestep == 0

def test_adapter_init_errors():
    """Test validation errors during initialization."""
    with pytest.raises(ValueError, match="must be a grain.DataLoader"):
        GrainSchedulerAdapter("not_a_loader")
        
    class BadLoader:
        pass # No __iter__
    
    # We need to trick isinstance check or strictly pass Grain DataLoader? 
    # Current impl checks isinstance(loader, grain.DataLoader) first.
    # So to test HASATTR check, we need an object that IS a DataLoader but HAS NO __iter__?
    # Grain DataLoader always has __iter__. So that check might be unreachable with type checking.
    # However, if we mock it:
    loader = MagicMock(spec=grain.DataLoader)
    del loader.__iter__ # Mock object allows deleting attrs
    # But MagicMock might recreate it on access if not careful.
    # Let's try:
    
    # Actually, simpler: bypass isinstance check if possible? No.
    # If the user passes a subclass that explicitly removes __iter__? Unlikely.
    # But coverage requires hitting it.
    
    # If we pass a Mock that specifies `spec=grain.DataLoader`, isinstance passes.
    # But if we make sure it doesn't have __iter__:
    loader = MagicMock(spec=grain.DataLoader)
    # MagicMock has everything by default.
    # We must explicitly set it to raise AttributeError or delete it.
    del loader.__iter__
    
    with pytest.raises(ValueError, match="loader must be iterable"):
        GrainSchedulerAdapter(loader)

def test_adapter_missing_data_source_property():
    """Test fallback when underlying loader has no _data_source."""
    loader = MagicMock(spec=grain.DataLoader)
    loader.__iter__.return_value = iter([])
    # Ensure no _data_source attr
    del loader._data_source 
    
    adapter = GrainSchedulerAdapter(loader)
    assert adapter.include_images is False
    assert adapter.file_list == []
    
    # Test setting file_list still raises
    with pytest.raises(NotImplementedError):
        adapter.file_list = []

def test_adapter_batch_truncation():
    """Test that batch is truncated if larger than target_batch_size."""
    # This covers `elif B > target_batch_size:` branches.
    ds = MockDataSource(length=4)
    loader = grain.DataLoader(data_source=ds, sampler=grain.IndexSampler(4, grain.NoSharding()), operations=[grain.Batch(batch_size=4)])
    adapter = GrainSchedulerAdapter(loader)
    
    # Ask for 2, get 4 from grain
    batch = adapter.get_batch(2)
    assert batch.flow_fields.shape == (2, 32, 32, 2)
    assert batch.mask.shape == (2,)

def test_adapter_images_pad_branch():
    """Test explicit padding branches for images."""
    # To hit `if pad_size > 0: ... images1 = np.pad`
    # We need a batch with images, smaller than target.
    class ImageSource(grain.RandomAccessDataSource):
        def __len__(self): return 1
        def __getitem__(self, idx):
            return {
                "flow_fields": np.zeros((32, 32, 2)),
                "images1": np.zeros((32, 32, 3)),
                "images2": np.zeros((32, 32, 3)),
                "file": "f"
            }
    
    loader = grain.DataLoader(data_source=ImageSource(), sampler=grain.IndexSampler(1, grain.NoSharding()), operations=[grain.Batch(1)])
    
    loader._data_source.include_images = True
    
    adapter = GrainSchedulerAdapter(loader)
    batch = adapter.get_batch(2) # Pad 1
    assert batch.images1 is not None
    assert batch.images2 is not None
    assert batch.images1.shape == (2, 32, 32, 3)

def test_adapter_images_truncate_branch():
    """Test explicit truncation branches for images."""
    # To hit `elif B > target_batch_size: ... images1 = ...`
    class ImageSource(grain.RandomAccessDataSource):
        def __len__(self): return 2
        def __getitem__(self, idx):
            return {
                "flow_fields": np.zeros((32, 32, 2)),
                "images1": np.zeros((32, 32, 3)),
                "images2": np.zeros((32, 32, 3)),
                "file": "f"
            }
            
    loader = grain.DataLoader(data_source=ImageSource(), sampler=grain.IndexSampler(2, grain.NoSharding()), operations=[grain.Batch(2)])
    loader._data_source.include_images = True
    
    adapter = GrainSchedulerAdapter(loader)
    batch = adapter.get_batch(1) # Truncate to 1
    assert batch.images1.shape == (1, 32, 32, 3)

    with pytest.raises(NotImplementedError):
        adapter.file_list = ["b"]

class MissingImageSource(MockFileSource):
    def load_file(self, f):
        return {"flow_fields": np.zeros((32, 32, 2))}

def test_adapter_missing_images_error():
    """Test KeyError if include_images is True but batch lacks them."""
    ds = MissingImageSource(files=["f"], include_images=True)
    loader = grain.DataLoader(data_source=ds, sampler=grain.IndexSampler(1, grain.NoSharding()), operations=[grain.Batch(1)])
    adapter = GrainSchedulerAdapter(loader)
    
    # Debug: Check property directly
    assert adapter.include_images is True, "Adapter.include_images should be True"
    
    with pytest.raises(KeyError, match="Images expected but not found"):
        adapter.get_batch(1)

def test_adapter_exhaustion():
    """Test StopIteration with custom message when exhausted."""
    loader = MagicMock(spec=grain.DataLoader)
    loader.__iter__.return_value = iter([]) # Empty iterator raises StopIteration immediately
    adapter = GrainSchedulerAdapter(loader)
    
    # Next should raise
    with pytest.raises(StopIteration, match="Grain DataLoader exhausted"):
        adapter.get_batch(1)

def test_adapter_invalid_types_validation():
    """Test Validation Errors for non-ndarray outputs."""
    class BadTypeSource(grain.RandomAccessDataSource):
        def __len__(self): return 1
        def __getitem__(self, idx):
            return {
                "flow_fields": [0], # List, not ndarray
                "file": "f"
            }
    
    # Mocking the iterator is safer to produce exact bad batch.
    loader = MagicMock(spec=grain.DataLoader)
    loader.__iter__.return_value = iter([{
        "flow_fields": "not_an_array",
        "file": ("f",)
    }])
    
    adapter = GrainSchedulerAdapter(loader)
    with pytest.raises(ValueError, match="Flow fields must be a np.ndarray"):
        adapter.get_batch(1)

def test_adapter_invalid_images_validation():
    """Test Validation Errors for non-ndarray images."""
    loader = MagicMock(spec=grain.DataLoader)
    loader.__iter__.return_value = iter([{
        "flow_fields": np.zeros((32, 32, 2)),
    }])
    # This should pass.
    # But let's verify invalid images type.
    
    batch_bad_img = {
        "flow_fields": np.zeros((1, 32, 32, 2)),
        "images1": "not_array",
        "images2": np.zeros((1, 32, 32, 3)),
        "file": ("f",)
    }
    loader.__iter__.return_value = iter([batch_bad_img])
    loader.include_images = True # Mocking property access on loader proxy? No, on _data_source.
    # But adapter `include_images` property checks `loader._data_source`. 
    # We can mock adapter.include_images to True on the instance to skip setup?
    # Or set valid property.
    loader._data_source = MagicMock()
    loader._data_source.include_images = True
    
    adapter = GrainSchedulerAdapter(loader)
    with pytest.raises(ValueError, match="Images1 must be a np.ndarray"):
        adapter.get_batch(1)

    # Test images2 invalid
    loader.__iter__.return_value = iter([{
        "flow_fields": np.zeros((32, 32, 2)),
        "images2": "not_an_array",
        "file": ("f",)
    }])
    # create new adapter to pick up new iterator
    adapter = GrainSchedulerAdapter(loader)
    # Ensure include_images is False to avoid KeyError for missing images1
    loader._data_source.include_images = False
    
    with pytest.raises(ValueError, match="Images2 must be a np.ndarray"):
        adapter.get_batch(1)

def test_episodic_adapter_init_error():
    """Test ValueError if source is not episodic."""
    loader = MagicMock(spec=grain.DataLoader)
    loader._data_source = MagicMock() # Not EpisodicDataSource
    # To make check fail, ensure not instance. 
    # MagicMock is not instance of EpisodicDataSource unless spec is set.
    
    with pytest.raises(ValueError, match="Data source is not an EpisodicDataSource"):
        GrainEpisodicAdapter(loader)

def test_episodic_truncated_episode():
    """Test next_episode break behavior when stream ends prematurely."""
    loader = MagicMock(spec=grain.DataLoader)
    # Mock iterator that is empty
    loader.__iter__.return_value = iter([])
    
    # Mock data source
    eds = MagicMock(spec=EpisodicDataSource)
    eds.episode_length = 5
    loader._data_source = eds
    
    adapter = GrainEpisodicAdapter(loader)
    adapter._current_timestep = 0 # Simulate we are in middle (steps_remaining = 4)
    
    # next_episode will call next(self._iterator) which raises StopIteration
    # This should hit line 256 (break) in adapter.py
    adapter.next_episode()
    
    assert adapter._current_timestep == -1

def test_episodic_missing_metadata_in_next_episode():
    """Test KeyError if batch missing _timestep in next_episode loop."""
    loader = MagicMock(spec=grain.DataLoader)
    loader._data_source = MagicMock(spec=EpisodicDataSource)
    loader._data_source.episode_length = 5
    
    # Batch missing _timestep
    batch = {"flow_fields": np.zeros((1, 32, 32, 2))}
    loader.__iter__.return_value = iter([batch])
    
    adapter = GrainEpisodicAdapter(loader)
    # Manually set steps remaining > 0
    adapter._current_timestep = 0 
    
    with pytest.raises(KeyError, match="Batch missing required '_timestep'"):
        adapter.next_episode()

def test_episodic_adapter_unknown_length():
    """Test property episode_length when None (should raise)."""
    loader = MagicMock(spec=grain.DataLoader)
    src = MagicMock(spec=EpisodicDataSource)
    src.episode_length = None # If source somehow has None
    loader._data_source = src
    
    adapter = GrainEpisodicAdapter(loader)
    # The init reads src.episode_length. If it is None, it sets self._episode_length = None.
    
    with pytest.raises(ValueError, match="Episode length unknown"):
        _ = adapter.episode_length

def test_episodic_adapter_property_access():
    """Test valid episode_length property access."""
    loader = MagicMock(spec=grain.DataLoader)
    src = MagicMock(spec=EpisodicDataSource)
    src.episode_length = 5
    loader._data_source = src
    
    adapter = GrainEpisodicAdapter(loader)
    assert adapter.episode_length == 5

def test_adapter_file_list_error():
    """Test that setting file_list raises NotImplementedError."""
    ds = MockDataSource(length=1)
    loader = grain.DataLoader(data_source=ds, sampler=grain.IndexSampler(1, grain.NoSharding()), operations=[])
    adapter = GrainSchedulerAdapter(loader)
    
    with pytest.raises(NotImplementedError):
        adapter.file_list = ["a"]

