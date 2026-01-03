"""Tests for Grain Adapters including GrainSchedulerAdapter and GrainEpisodicAdapter.

These adapters are responsible for bridging the Grain DataLoader (which handles 
data loading, sampling, and batching) with the SynthPix internal protocols. 
The tests cover basic loading, padding, truncation, epoch-aware behavior, 
episodic handling (including episode exhaustion), and various error conditions.
"""

from unittest.mock import MagicMock

import grain.python as grain
import numpy as np
import pytest

from synthpix.data_sources import EpisodicDataSource, FileDataSource
from synthpix.data_sources.adapter import (GrainEpisodicAdapter,
                                           GrainSchedulerAdapter)
from synthpix.scheduler.protocol import EpisodeEndError


class MockDataSource(grain.RandomAccessDataSource):
    """Mock RandomAccessDataSource that returns predictable episodic metadata.

    Simulates a sequence of items with '_chunk_id', '_timestep', and 
    '_is_last_step' keys, allowing verification of how adapters handle 
    episodic boundaries.
    """

    def __init__(self, length=10, episode_len=5):
        super().__init__()
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
            "_is_last_step": (t == self.episode_length - 1),
        }


class MockFileSource(FileDataSource):
    def __init__(self, files, include_images=False):
        super().__init__(dataset_path=files)
        self._include_images = include_images

    @property
    def include_images(self):
        return self._include_images

    @include_images.setter
    def include_images(self, value):
        self._include_images = value

    def load_file(self, f):
        return {}


def test_scheduler_adapter_basic():
    """Test standard non-episodic adapter behavior.

    Verifies that GrainSchedulerAdapter correctly retrieves batches from a 
    Grain DataLoader, maintains the expected flow field shape, and can be 
       reset.
    """
    ds = MockDataSource(length=10)
    # Batch size 2
    loader = grain.DataLoader(
        data_source=ds,
        sampler=grain.IndexSampler(
            num_records=len(ds),
            shuffle=False,
            shard_options=grain.NoSharding(),
            num_epochs=1,
        ),
        operations=[grain.Batch(batch_size=2)],
    )

    adapter = GrainSchedulerAdapter(loader)

    # First batch: indices 0, 1 -> t=0, t=1
    batch = adapter.get_batch(2)
    assert batch.flow_fields.shape == (2, 32, 32, 2), f"Flow field shape mismatch. Expected (2, 32, 32, 2), got {batch.flow_fields.shape}"
    assert batch.mask is not None, "Batch mask should not be None"
    assert batch.mask.shape == (2,), f"Mask shape mismatch. Expected (2,), got {batch.mask.shape}"
    assert np.all(batch.mask), f"All items in mask should be True, but got {batch.mask}"

    # Check shape inference (cached)
    shape = adapter.get_flow_fields_shape()
    assert shape == (32, 32, 2), f"Inferred shape mismatch. Expected (32, 32, 2), got {shape}"

    # Reset
    adapter.reset()
    batch_rst = adapter.get_batch(2)
    assert batch_rst.flow_fields.shape == (2, 32, 32, 2), f"Reset flow field shape mismatch. Expected (2, 32, 32, 2), got {batch_rst.flow_fields.shape}"


def test_scheduler_adapter_padding():
    """Test padding when the requested batch size exceeds the DataLoader's batch size.

    GrainSchedulerAdapter should structurally pad the returned batch to 
    match the requested size, marking the padded items as invalid in the mask.
    """
    # Data returns batch size 2 (from Grain), but we ask for 4.
    ds = MockDataSource(length=2)
    loader = grain.DataLoader(
        data_source=ds,
        sampler=grain.IndexSampler(
            num_records=len(ds),
            shuffle=False,
            shard_options=grain.NoSharding(),
            num_epochs=1,
        ),
        operations=[grain.Batch(batch_size=2)],
    )
    adapter = GrainSchedulerAdapter(loader)

    batch = adapter.get_batch(4)
    assert batch.flow_fields.shape == (4, 32, 32, 2), f"Padded flow field shape mismatch. Expected (4, 32, 32, 2), got {batch.flow_fields.shape}"
    # First 2 valid, last 2 invalid
    assert batch.mask is not None, "Batch mask should not be None"
    assert np.sum(batch.mask) == 2, f"Expected 2 valid items in mask, got {np.sum(batch.mask)}"
    assert batch.mask[0], "First item in mask should be True"
    assert batch.mask[2] == False, "Third item in mask should be False (padded)"


def test_episodic_adapter_exhaustion():
    """Test that GrainEpisodicAdapter raises EpisodeEndError when the loader is exhausted.

    In an episodic context, reaching the end of the DataLoader stream 
    specifically signifies the end of an episode sequence.
    """
    ds = MockDataSource(length=2, episode_len=2)  # 0, 1
    loader = grain.DataLoader(
        data_source=ds,
        sampler=grain.IndexSampler(
            num_records=len(ds),
            shuffle=False,
            shard_options=grain.NoSharding(),
            num_epochs=1,
        ),
        operations=[grain.Batch(batch_size=1)],
    )
    # Monkey patch source
    loader._data_source = MagicMock(spec=EpisodicDataSource)
    loader._data_source.episode_length = 2
    loader._data_source.include_images = False

    adapter = GrainEpisodicAdapter(loader)

    _ = adapter.get_batch(1)  # t=0
    _ = adapter.get_batch(1)  # t=1 (last)

    # Next call -> StopIteration from loader -> EpisodeEnd
    with pytest.raises(EpisodeEndError):
        adapter.get_batch(1)


def test_get_flow_fields_shape_empty():
    """Test behavior when loader is empty."""
    # Mock loader directly since grain.DataLoader doesn't like empty sources
    # easily
    loader = MagicMock(spec=grain.DataLoader)
    loader.__iter__.return_value = iter([])  # Empty iterator

    adapter = GrainSchedulerAdapter(loader)

    with pytest.raises(StopIteration):
        adapter.get_flow_fields_shape()


def test_adapter_shutdown():
    """Test that the shutdown method can be called without error.

    The adapter delegates shutdown to the underlying loader if supported.
    """
    ds = MockDataSource(length=1)
    loader = grain.DataLoader(
        data_source=ds,
        sampler=grain.IndexSampler(1, grain.NoSharding()),
        operations=[],
    )
    adapter = GrainSchedulerAdapter(loader)
    adapter.shutdown()  # Should do nothing but exist


def test_adapter_properties():
    """Test delegation of 'include_images' and 'file_list' properties to the source.

    The adapter should expose metadata from the underlying FileDataSource.
    """
    ds = MockFileSource(files=["a"], include_images=True)

    # We need to ensure loader._data_source is accessible.
    # Grain wraps it.
    loader = grain.DataLoader(
        data_source=ds,
        sampler=grain.IndexSampler(1, grain.NoSharding()),
        operations=[],
    )

    # Grain might not expose _data_source easily in all versions, but our adapter relies on it.
    # If the adapter implementation uses `loader._data_source`, we assume it's there.
    # Ensure our MockDataSource is attached.
    # grain.DataLoader usually sets self._data_source = data_source

    adapter = GrainSchedulerAdapter(loader)
    assert adapter.include_images is True, "include_images should be True as delegated to source"
    assert adapter.file_list == ["a"], f"file_list mismatch. Expected ['a'], got {adapter.file_list}"

    # Test setter error
    with pytest.raises(NotImplementedError):
        adapter.file_list = ["b"]


def test_adapter_images_and_files_padding():
    """Test correct handling and structural padding of paired images and file metadata.

    When structural padding is applied, all batch fields (flow_fields, 
    images1, images2, files) must be padded consistently to the target batch size.
    """

    # Source returning images and file
    class ImageSource(grain.RandomAccessDataSource):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "flow_fields": np.zeros((32, 32, 2)),
                "images1": np.zeros((32, 32, 3)),
                "images2": np.zeros((32, 32, 3)),
                "file": "file_path",
            }

    loader = grain.DataLoader(
        data_source=ImageSource(),
        sampler=grain.IndexSampler(1, grain.NoSharding()),
        operations=[grain.Batch(batch_size=1)],
    )
    # Mock property inheritance manually if needed, but ImageSource doesn't have it.
    # Adapter checks loader._data_source.include_images.
    # We set it on the source instance.
    loader._data_source.include_images = True  # type: ignore

    adapter = GrainSchedulerAdapter(loader)

    # Request batch 2 (pad 1)
    batch = adapter.get_batch(2)
    assert batch.images1 is not None, "Batch images1 should not be None"
    assert batch.images1.shape == (2, 32, 32, 3), f"Images1 shape mismatch. Expected (2, 32, 32, 3), got {batch.images1.shape}"
    assert batch.images2 is not None, "Batch images2 should not be None"
    assert batch.images2.shape == (2, 32, 32, 3), f"Images2 shape mismatch. Expected (2, 32, 32, 3), got {batch.images2.shape}"
    assert batch.files == ("file_path", ""), f"Files metadata mismatch. Expected ('file_path', ''), got {batch.files}"
    assert batch.mask is not None, "Batch mask should not be None"
    assert batch.mask[1] == False, "Second item in mask should be False (padded)"


def test_episodic_next_episode_logic():
    """Test that next_episode() correctly advances the DataLoader to the next sequence.

    It should consume items until a new episode (timestep 0) is encountered 
    or the stream is exhausted.
    """
    # Create a loader that yields:
    # 0: {ts: 0}
    # 1: {ts: 1}
    # 2: {ts: 2} (end of ep)
    # 3: {ts: 0} (next ep)

    # Custom iterator mock to strictly control output sequence
    loader = MagicMock(spec=grain.DataLoader)

    # Define source properties on the mock loader
    # Using a MagicMock for the data source
    mock_source = MagicMock(
        spec=EpisodicDataSource
    )  # Use spec ensures checking
    mock_source.episode_length = 3
    mock_source.include_images = False

    # Attach to loader
    loader._data_source = mock_source

    # Iterator sequence
    batch0 = {
        "_timestep": np.array([0]),
        "flow_fields": np.zeros((1, 32, 32, 2)),
    }
    batch1 = {
        "_timestep": np.array([1]),
        "flow_fields": np.zeros((1, 32, 32, 2)),
    }
    batch2 = {
        "_timestep": np.array([2]),
        "flow_fields": np.zeros((1, 32, 32, 2)),
    }
    batch_next = {
        "_timestep": np.array([0]),
        "flow_fields": np.zeros((1, 32, 32, 2)),
    }

    # Iterator returns: t0, t1, t2, t0...
    loader.__iter__.return_value = iter([batch0, batch1, batch2, batch_next])

    adapter = GrainEpisodicAdapter(loader)

    # 1. Get t0
    batch = adapter.get_batch(1)
    assert adapter._current_timestep == 0, f"Expected current timestep 0, got {adapter._current_timestep}"
    assert adapter.steps_remaining() == 2, f"Expected 2 steps remaining, got {adapter.steps_remaining()}"

    # 2. Skip remaining (t1, t2)
    # logic: while steps_remaining > 0: next(...)
    # It should consume batch1 (t1 -> rem 1), batch2 (t2 -> rem 0 -> break)
    adapter.next_episode()

    # 3. Next call should be batch_next (t0)
    # Adapter sets _current_timestep = -1 after skip loop
    assert adapter._current_timestep == -1, f"Expected current timestep -1 after skipping episode, got {adapter._current_timestep}"

    b_new = adapter.get_batch(1)
    # b_new is SchedulerData, batch_next is dict
    assert b_new.flow_fields is not None, "New batch flow fields should not be None"
    assert np.array_equal(b_new.flow_fields, batch_next["flow_fields"]), "New batch flow fields do not match expected data"
    assert adapter._current_timestep == 0, f"Expected current timestep 0 after new episode, got {adapter._current_timestep}"


def test_adapter_init_errors():
    """Test validation errors during initialization."""
    with pytest.raises(ValueError, match="must be a grain.DataLoader"):
        GrainSchedulerAdapter("not_a_loader")  # type: ignore

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
    assert adapter.include_images is False, "include_images should be False if source is missing"
    assert adapter.file_list == [], f"file_list should be empty if source is missing, got {adapter.file_list}"

    # Test setting file_list still raises
    with pytest.raises(NotImplementedError):
        adapter.file_list = []


def test_adapter_batch_truncation():
    """Test that the batch is truncated if the DataLoader returns more items than requested.

    This scenario can occur if Grain's batching configuration differs from the 
    adapter's retrieval request.
    """
    # This covers `elif B > target_batch_size:` branches.
    ds = MockDataSource(length=4)
    loader = grain.DataLoader(
        data_source=ds,
        sampler=grain.IndexSampler(4, grain.NoSharding()),
        operations=[grain.Batch(batch_size=4)],
    )
    adapter = GrainSchedulerAdapter(loader)

    # Ask for 2, get 4 from grain
    batch = adapter.get_batch(2)
    assert batch.flow_fields.shape == (2, 32, 32, 2), f"Truncated flow field shape mismatch. Expected (2, 32, 32, 2), got {batch.flow_fields.shape}"
    assert batch.mask is not None, "Batch mask should not be None"
    assert batch.mask.shape == (2,), f"Mask shape mismatch. Expected (2,), got {batch.mask.shape}"


def test_adapter_images_pad_branch():
    """Test that images are correctly padded when structural padding is applied.

    Verifies that the image arrays are zero-padded to match the target batch size.
    """

    # To hit `if pad_size > 0: ... images1 = np.pad`
    # We need a batch with images, smaller than target.
    class ImageSource(grain.RandomAccessDataSource):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "flow_fields": np.zeros((32, 32, 2)),
                "images1": np.zeros((32, 32, 3)),
                "images2": np.zeros((32, 32, 3)),
                "file": "f",
            }

    loader = grain.DataLoader(
        data_source=ImageSource(),
        sampler=grain.IndexSampler(1, grain.NoSharding()),
        operations=[grain.Batch(1)],
    )

    loader._data_source.include_images = True

    adapter = GrainSchedulerAdapter(loader)
    batch = adapter.get_batch(2)  # Pad 1
    assert batch.images1 is not None, "Batch images1 should not be None"
    assert batch.images2 is not None, "Batch images2 should not be None"
    assert batch.images1.shape == (2, 32, 32, 3), f"Padded images1 shape mismatch. Expected (2, 32, 32, 3), got {batch.images1.shape}"
    assert batch.images2 is not None, "Batch images2 should not be None"
    assert batch.images2.shape == (2, 32, 32, 3), f"Padded images2 shape mismatch. Expected (2, 32, 32, 3), got {batch.images2.shape}"


def test_adapter_images_truncate_branch():
    """Test that images are correctly truncated when the batch is larger than requested.

    Verifies that the image arrays are sliced to match the target batch size.
    """

    # To hit `elif B > target_batch_size: ... images1 = ...`
    class ImageSource(grain.RandomAccessDataSource):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {
                "flow_fields": np.zeros((32, 32, 2)),
                "images1": np.zeros((32, 32, 3)),
                "images2": np.zeros((32, 32, 3)),
                "file": "f",
            }

    loader = grain.DataLoader(
        data_source=ImageSource(),
        sampler=grain.IndexSampler(2, grain.NoSharding()),
        operations=[grain.Batch(2)],
    )
    loader._data_source.include_images = True

    adapter = GrainSchedulerAdapter(loader)
    batch = adapter.get_batch(1)  # Truncate to 1
    assert batch.images1 is not None, "Batch images1 should not be None"
    assert batch.images1.shape == (1, 32, 32, 3), f"Truncated images1 shape mismatch. Expected (1, 32, 32, 3), got {batch.images1.shape}"

    with pytest.raises(NotImplementedError):
        adapter.file_list = ["b"]


class MissingImageSource(MockFileSource):
    def load_file(self, f):
        return {"flow_fields": np.zeros((32, 32, 2))}


def test_adapter_missing_images_error():
    """Test KeyError if include_images is True but batch lacks them."""
    ds = MissingImageSource(files=["f"], include_images=True)
    loader = grain.DataLoader(
        data_source=ds,
        sampler=grain.IndexSampler(1, grain.NoSharding()),
        operations=[grain.Batch(1)],
    )
    adapter = GrainSchedulerAdapter(loader)

    # Debug: Check property directly
    assert adapter.include_images is True, (
        "Adapter.include_images should be True"
    )

    with pytest.raises(KeyError, match="Images expected but not found"):
        adapter.get_batch(1)


def test_adapter_exhaustion():
    """Test that StopIteration is raised with a descriptive message when the loader is exhausted.

    In a non-episodic context, this signals that all data has been consumed.
    """
    loader = MagicMock(spec=grain.DataLoader)
    loader.__iter__.return_value = iter(
        []
    )  # Empty iterator raises StopIteration immediately
    adapter = GrainSchedulerAdapter(loader)

    # Next should raise
    with pytest.raises(StopIteration, match="Grain DataLoader exhausted"):
        adapter.get_batch(1)


def test_adapter_invalid_types_validation():
    """Test that the adapter validates the type of returned data.

    It should raise a ValueError if 'flow_fields' is not a NumPy array.
    """

    # Mocking the iterator is safer to produce exact bad batch.
    loader = MagicMock(spec=grain.DataLoader)
    loader.__iter__.return_value = iter(
        [{"flow_fields": "not_an_array", "file": ("f",)}]
    )

    adapter = GrainSchedulerAdapter(loader)
    with pytest.raises(ValueError, match="Flow fields must be a np.ndarray"):
        adapter.get_batch(1)


def test_adapter_invalid_images_validation():
    """Test Validation Errors for non-ndarray images."""
    loader = MagicMock(spec=grain.DataLoader)
    loader.__iter__.return_value = iter(
        [
            {
                "flow_fields": np.zeros((32, 32, 2)),
            }
        ]
    )
    # This should pass.
    # But let's verify invalid images type.

    batch_bad_img = {
        "flow_fields": np.zeros((1, 32, 32, 2)),
        "images1": "not_array",
        "images2": np.zeros((1, 32, 32, 3)),
        "file": ("f",),
    }
    loader.__iter__.return_value = iter([batch_bad_img])
    loader.include_images = (
        True  # type: ignore
    )
    # But adapter `include_images` property checks `loader._data_source`.
    # We can mock adapter.include_images to True on the instance to skip setup?
    # Or set valid property.
    loader._data_source = MagicMock()
    loader._data_source.include_images = True

    adapter = GrainSchedulerAdapter(loader)
    with pytest.raises(ValueError, match="Images1 must be a np.ndarray"):
        adapter.get_batch(1)

    # Test images2 invalid
    loader.__iter__.return_value = iter(
        [
            {
                "flow_fields": np.zeros((32, 32, 2)),
                "images2": "not_an_array",
                "file": ("f",),
            }
        ]
    )
    # create new adapter to pick up new iterator
    adapter = GrainSchedulerAdapter(loader)
    # Ensure include_images is False to avoid KeyError for missing images1
    loader._data_source.include_images = False

    # When include_images is False, images1 and  
    with pytest.raises(ValueError, match="Images2 must be a np.ndarray"):
        adapter.get_batch(1)


def test_episodic_adapter_init_error():
    """Test that GrainEpisodicAdapter requires an EpisodicDataSource.

    It should raise a ValueError if the underlying loader's source does not 
    implement episodic features.
    """
    loader = MagicMock(spec=grain.DataLoader)
    loader._data_source = MagicMock()  # Not EpisodicDataSource
    # To make check fail, ensure not instance.
    # MagicMock is not instance of EpisodicDataSource unless spec is set.

    with pytest.raises(
        ValueError, match="Data source is not an EpisodicDataSource"
    ):
        GrainEpisodicAdapter(loader)


def test_episodic_truncated_episode():
    """Test that next_episode() handles premature stream exhaustion gracefully.

    If the DataLoader runs out of items while skipping to the next episode, 
    the adapter should reset its state correctly.
    """
    loader = MagicMock(spec=grain.DataLoader)
    # Mock iterator that is empty
    loader.__iter__.return_value = iter([])

    # Mock data source
    eds = MagicMock(spec=EpisodicDataSource)
    eds.episode_length = 5
    loader._data_source = eds

    adapter = GrainEpisodicAdapter(loader)
    adapter._current_timestep = (
        0  # Simulate we are in middle (steps_remaining = 4)
    )

    # next_episode will call next(self._iterator) which raises StopIteration
    # This should hit line 256 (break) in adapter.py
    adapter.next_episode()

    assert adapter._current_timestep == -1, f"Expected current timestep -1 after stream exhaustion, got {adapter._current_timestep}"


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
    """Test that accessing episode_length raises ValueError if the source doesn't provide it.

    In some cases, the episodic length might not be determinable from the source.
    """
    loader = MagicMock(spec=grain.DataLoader)
    src = MagicMock(spec=EpisodicDataSource)
    src.episode_length = None  # If source somehow has None
    loader._data_source = src

    adapter = GrainEpisodicAdapter(loader)
    # The init reads src.episode_length. If it is None, it sets
    # self._episode_length = None.

    with pytest.raises(ValueError, match="Episode length unknown"):
        _ = adapter.episode_length


def test_episodic_adapter_property_access():
    """Test valid episode_length property access."""
    loader = MagicMock(spec=grain.DataLoader)
    src = MagicMock(spec=EpisodicDataSource)
    src.episode_length = 5
    loader._data_source = src

    adapter = GrainEpisodicAdapter(loader)
    assert adapter.episode_length == 5, f"Expected episode length 5, got {adapter.episode_length}"


def test_adapter_file_list_error():
    """Test that setting file_list raises NotImplementedError."""
    ds = MockDataSource(length=1)
    loader = grain.DataLoader(
        data_source=ds,
        sampler=grain.IndexSampler(1, grain.NoSharding()),
        operations=[],
    )
    adapter = GrainSchedulerAdapter(loader)

    with pytest.raises(NotImplementedError):
        adapter.file_list = ["a"]

def test_adapter_explicit_padding():
    """Test that adapter respects _is_padding flag."""
    # Batch size 2.
    # Flow shape (2, 10, 10, 2)
    # Item 0: Valid, Item 1: Padding
    
    flow = np.ones((2, 10, 10, 2), dtype=np.float32)
    images1 = np.ones((2, 10, 10, 3), dtype=np.float32)
    images2 = np.ones((2, 10, 10, 3), dtype=np.float32)
    files = ("f1", "f2")
    
    # is_padding array
    is_padding = np.array([False, True], dtype=bool)
    
    batch = {
        "flow_fields": flow,
        "images1": images1,
        "images2": images2,
        "file": files,
        "_is_padding": is_padding
    }
    
    # Mock Grain DataLoader using MagicMock as per existing tests
    loader = MagicMock(spec=grain.DataLoader)
    loader.__iter__.return_value = iter([batch])
    # Ensure include_images checks pass if needed
    loader._data_source = MagicMock()
    loader._data_source.include_images = True  # type: ignore
    # Ensure epoch logic treats this as last epoch (or single epoch)
    loader._sampler = MagicMock()
    loader._sampler.num_epochs = 1
    loader._data_source.__len__.return_value = 2

    adapter = GrainSchedulerAdapter(loader)
    
    # Act
    data = adapter.get_batch(batch_size=2)
    
    # Assert
    # Item 0 should be intact (ones)
    assert np.all(data.flow_fields[0] == 1.0), "First item in batch should not be zeroed out"
    assert data.mask is not None, "Batch mask should not be None"
    assert data.mask[0] == True, "First item in mask should be True"
    
    # Item 1 should be zeroed out
    assert np.all(data.flow_fields[1] == 0.0), f"Padded flow field item should be zeroed out, but got values like {data.flow_fields[1].flatten()[0]}"
    assert data.images1 is not None, "Batch images1 should not be None"
    assert np.all(data.images1[1] == 0.0), f"Padded images1 item should be zeroed out, but got values like {data.images1[1].flatten()[0]}"
    assert data.images2 is not None, "Batch images2 should not be None"
    assert np.all(data.images2[1] == 0.0), f"Padded images2 item should be zeroed out, but got values like {data.images2[1].flatten()[0]}"
    assert data.mask[1] == False, "Second item in mask should be False (padded)"

class MockFileDataSource(FileDataSource):
    """Mock source returning custom files."""
    def __init__(self, files):
        self.files = files
        super().__init__(dataset_path=self.files)

    @property
    def file_list(self):
        return self.files

    def load_file(self, file_path):
        # Return dummy data
        return {
            "flow_fields": np.zeros((10, 10, 2), dtype=np.float32),
             # Optional: track file ID to verify wrapping
            "file_id": 999 
        }

def test_epoch_aware_wrapping_and_padding(tmp_path):
    """Verify GrainSchedulerAdapter handles epoch-aware wrapping and padding.

    Test scenario:
    - 3 files, Episode length 1.
    - Batch size 2.
    - Total Epochs: 3.

    Calculations:
    - Files: 3. Starts: 3.
    - Chunks per epoch = ceil(3/2) = 2 chunks.
    - Chunk 0 covers items 0, 1. (Batch 0)
    - Chunk 1 covers items 2, 0 (wrapped). (Batch 1)

    Expectation:
    - Non-Final Epochs (Epoch 0, 1):
      Data should seamlessly wrap around. The wrapped item (index 0 reused)
      should NOT be padded (mask=True). This ensures continuity.
    - Final Epoch (Epoch 2):
      The wrapped item should be padded (mask=False) to indicate end of
      dataset training without starting a partial new epoch.
    """
    d = tmp_path / "data"
    d.mkdir()
    files = []
    for i in range(3):
        f = d / f"f{i}.mat"
        f.touch()
        files.append(str(f))
    
    source = MockFileDataSource(files)
    # Ensure stable order logic inside EpisodicDataSource
    ds = EpisodicDataSource(source, batch_size=2, episode_length=1, seed=42)
    
    # Check ds length: 2 chunks * 1 step = 2 items per epoch * 2 batch size = 4 items
    assert len(ds) == 4, f"Expected 4 items in EpisodicDataSource, got {len(ds)}"
    
    loader = grain.DataLoader(
        data_source=ds,
        sampler=grain.IndexSampler(
            num_records=len(ds),
            shuffle=False,
            shard_options=grain.NoSharding(),
            num_epochs=3, # 3 Epochs
        ),
        operations=[grain.Batch(batch_size=2)],
    )
    
    adapter = GrainSchedulerAdapter(loader)
    
    batches = []
    try:
        while True:
            batches.append(adapter.get_batch(2))
    except (StopIteration, Exception):
        pass
        
    assert len(batches) == 6, f"Expected 6 batches across 3 epochs (2 batches/epoch), got {len(batches)}"
    
    # Epoch 1
    b1 = batches[1]
    assert b1.mask[1] == True, f"Epoch 1 wrap should be valid, but mask[1] is {b1.mask[1]}"
    
    # Epoch 2
    b3 = batches[3]
    assert b3.mask[1] == True, f"Epoch 2 wrap should be valid, but mask[1] is {b3.mask[1]}"

    # Epoch 3
    b5 = batches[5]
    assert b5.mask[1] == False, f"Epoch 3 wrap should be padded (final epoch), but mask[1] is {b5.mask[1]}"

import numpy as np
import pytest
from unittest.mock import MagicMock, PropertyMock
import grain.python as grain
from synthpix.data_sources.adapter import GrainSchedulerAdapter
from synthpix.data_sources.base import FileDataSource

class CrashingLoader(grain.DataLoader):
    """Loader that crashes during attribute inspection."""
    def __init__(self):
        pass
    def __iter__(self):
        return iter([])
    @property
    def sampler(self):
        raise RuntimeError("Crash on access")
    @property
    def _sampler(self):
        raise RuntimeError("Crash on access")
    @property
    def _data_source(self):
        raise RuntimeError("Crash on access")
    @property
    def _dataset(self):
        raise RuntimeError("Crash on access")

def test_adapter_init_exception_fallback():
    """Test fallback when loader inspection raises exception."""
    loader = CrashingLoader()
    # Should not raise, just warn and default to unsafe epoch logic
    adapter = GrainSchedulerAdapter(loader)
    assert adapter._can_determine_epoch is False, "Epoch logic should be disabled on inspection crash"

def test_cached_shape():
    """Test get_flow_fields_shape uses cache."""
    # Mock loader
    batch = {
        "flow_fields": np.zeros((2, 10, 10, 2)),
    }
    loader = MagicMock(spec=grain.DataLoader)
    loader.__iter__.return_value = iter([batch])
    
    adapter = GrainSchedulerAdapter(loader)
    
    # First call - computes shape
    shape1 = adapter.get_flow_fields_shape()
    assert shape1 == (10, 10, 2), f"Expected shape (10, 10, 2), got {shape1}"
    
    # Second call - should hit cache (missing line 92)
    # We can verify by exhausting the iterator or mocking it
    # If it called iter() again, it would be fine, but coverage checks if the return path is taken.
    shape2 = adapter.get_flow_fields_shape()
    assert shape2 == (10, 10, 2), f"Expected cached shape (10, 10, 2), got {shape2}"

def test_file_list_setter():
    """Test setting file_list raises NotImplementedError."""
    loader = MagicMock(spec=grain.DataLoader)
    adapter = GrainSchedulerAdapter(loader)
    
    with pytest.raises(NotImplementedError):
        adapter.file_list = ["foo"]

def test_infinite_epoch_logic():
    """Test fallback when dataset len is known but num_epochs is None (Infinite)."""
    # Cover line 188-189
    loader = MagicMock(spec=grain.DataLoader)
    loader.__iter__.return_value = iter([])
    
    # Configure so dataset len is known, but sampler num_epochs missing
    loader._data_source = MagicMock()
    loader._data_source.include_images = False # Disable images to avoid KeyError
    loader._data_source.__len__.return_value = 100
    # No sampler
    loader.sampler = None 
    loader._sampler = None
    
    adapter = GrainSchedulerAdapter(loader)
    
    assert adapter._can_determine_epoch is True, "Epoch logic should be enabled when dataset len is known"
    assert adapter._num_epochs is None, f"Expected infinite epochs (None), got {adapter._num_epochs}"
    
    # To hit line 189 inside _to_scheduler_data, we need to call get_batch
    # And have is_padding present
    
    batch = {
        "flow_fields": np.zeros((2, 10, 10, 2)),
        "_is_padding": np.array([False, True]) # Triggers logic
    }
    loader.__iter__.return_value = iter([batch])
    
    # Reset iterator
    adapter.reset()
    
    data = adapter.get_batch(2)
    # If respect_padding is False (expected), then data masked should be True for both
    # because user wants infinite stream
    assert data.mask is not None, "Batch mask should not be None"
    assert np.all(data.mask), f"Infinite epoch should not respect padding, but got mask: {data.mask}"

def test_structural_and_explicit_padding_combination():
    """Test combination of structural padding (pad_size > 0) AND explicit _is_padding flag."""
    # Cover lines 265-266
    
    # Batch has 1 item. Target is 2.
    # Item is marked as Padding.
    # Structural padding adds 1 item (implicitly invalid).
    # We want to ensure _is_padding is concatenated correctly.
    
    batch = {
        "flow_fields": np.zeros((1, 10, 10, 2)),
        "_is_padding": np.array([True]) # The existing item is padding
    }
    
    loader = MagicMock(spec=grain.DataLoader)
    
    # Configure epoch awareness so respect_padding=True
    # Num epochs 1. Dataset len 1.
    loader._sampler = MagicMock()
    loader._sampler.num_epochs = 1
    loader._data_source = MagicMock()
    loader._data_source.include_images = False # Disable images
    loader._data_source.__len__.return_value = 1
    
    loader.__iter__.return_value = iter([batch])
    
    adapter = GrainSchedulerAdapter(loader)
    
    # Get batch size 2
    data = adapter.get_batch(2)
    
    # Result:
    # Item 0: is_padding=True -> Mask=False, Zeroed.
    # Item 1: Structural Pad -> Mask=False, Zeroed.
    # Both should be masked out.
    assert data.mask is not None, "Batch mask should not be None"
    assert data.mask[0] == False, "First item (padding) should be masked"
    assert data.mask[1] == False, "Second item (structural pad) should be masked"
    # data.flow_fields[0] should be 0
    assert np.all(data.flow_fields[0] == 0.0), f"Padded item should be zeroed out, but got values like {data.flow_fields[0].flatten()[0]}"


def test_truncation_with_padding_flag():
    """Test truncation (target < current) with explicit _is_padding flag."""
    # Ensure line 253 is covered
    
    # Batch has 3 items. Target is 2.
    batch = {
        "flow_fields": np.zeros((3, 10, 10, 2)),
        "_is_padding": np.array([False, False, True]) # Last item is padding, but should be truncated
    }
    
    loader = MagicMock(spec=grain.DataLoader)
    loader._sampler = MagicMock()
    loader._sampler.num_epochs = 1
    loader._data_source = MagicMock()
    loader._data_source.include_images = False
    loader._data_source.__len__.return_value = 3
    
    loader.__iter__.return_value = iter([batch])
    
    adapter = GrainSchedulerAdapter(loader)
    
    # Get batch size 2
    data = adapter.get_batch(2)
    
    # Verify truncation
    assert data.flow_fields.shape[0] == 2, f"Expected 2 items after truncation, got {data.flow_fields.shape[0]}"
    assert data.mask is not None, "Batch mask should not be None"
    assert data.mask.shape[0] == 2, f"Expected mask size 2, got {data.mask.shape[0]}"
    # Ensure is_padding logic worked (mask should be True for first 2 items)
    assert np.all(data.mask), f"Expected mask values to be True, got {data.mask}"
    
    # We don't strictly assert the internal is_padding truncation happened 
    # but execution path guarantees it.
