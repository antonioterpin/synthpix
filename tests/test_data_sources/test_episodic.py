"""Tests for EpisodicDataSource."""

import os
import numpy as np
import pytest
import grain.python as grain
from synthpix.data_sources import EpisodicDataSource, FileDataSource

class MockDataSource(FileDataSource):
    """Simple source that returns the filepath."""
    def __init__(self, file_list, include_images=False):
        self._file_list = file_list
        self._include_images = include_images
        
    def load_file(self, file_path):
        return {"file": file_path}

    @property
    def include_images(self) -> bool:
        return self._include_images

def test_episodic_interleaving(tmp_path):
    # Setup mock file system: 2 dirs, 10 files each
    # d1: [f0..f9], d2: [f0..f9]
    # Episode length = 5
    # Batch size = 2
    
    files = []
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    
    for i in range(10):
        # Create Dummy files so os.path.isfile checks pass if any
        # But FileDataSource usually just takes a list. 
        # Our Mock passes list directly.
        f1 = d1 / f"f_{i:02d}.mat"
        f2 = d2 / f"f_{i:02d}.mat"
        files.append(str(f1))
        files.append(str(f2))
    
    # Sort files to ensure stable order for test expectations
    files.sort()
    
    source = MockDataSource(files)
    
    # Init Episodic Wrapper
    # Starts per dir: 10 - 5 + 1 = 6 starts (0..5)
    # Total starts = 12
    # Batch size 2 means 12 // 2 = 6 chunks of episodes
    ds = EpisodicDataSource(source, batch_size=2, episode_length=5, seed=42)
    
    # Total items should be 6 chunks * 2 episodes/chunk * 5 steps = 60 items
    assert len(ds) == 60
    
    # Check item 0 and 1 -> Should be t=0 of different episodes
    item0 = ds[0]
    item1 = ds[1]
    
    assert item0["_timestep"] == 0
    assert item1["_timestep"] == 0
    assert item0["_chunk_id"] == 0
    assert item1["_chunk_id"] == 0
    
    # Check item 2 -> Should be t=1 of first episode in chunk
    item2 = ds[2]
    assert item2["_timestep"] == 1
    assert item2["_chunk_id"] == 0
    
    # Ensure files are actually sequential in time
    f0 = item0["file"]
    f2 = item2["file"]
    
    # Extract index from filename "f_XX.mat"
    def extract_idx(f):
        return int(os.path.basename(f).split("_")[1].split(".")[0])
        
    idx0 = extract_idx(f0)
    idx2 = extract_idx(f2)
    assert idx2 == idx0 + 1
    assert os.path.dirname(f0) == os.path.dirname(f2) # Same directory

def test_episodic_grain_integration():
    """Verify it actually works efficiently with Grain DataLoader."""
    # We construct files such that they are strictly ordered in one dir for simplicity,
    # or multiple directories.
    files = [f"/d/f_{i:02d}.mat" for i in range(20)]
    source = MockDataSource(files)
    
    # 20 files, len 5 -> 16 starts
    # batch 4 -> 4 chunks
    ds = EpisodicDataSource(source, batch_size=4, episode_length=5)
    
    # Use simple sequential sampler
    loader = grain.DataLoader(
        data_source=ds,
        sampler=grain.IndexSampler(
            len(ds), shuffle=False, shard_options=grain.NoSharding(), num_epochs=1
        ),
        operations=[grain.Batch(batch_size=4)],
        worker_count=0
    )
    
    batch = next(iter(loader))
    
    # Batch should contain 4 items.
    timesteps = batch["_timestep"] # expected shape (4,)
    assert np.all(timesteps == 0)
    
    # Next batch
    it = iter(loader)
    next(it) # t=0
    batch_t1 = next(it) # t=1
    assert np.all(batch_t1["_timestep"] == 1)

def test_episodic_type_validation():
    """Test that TypeError is raised if source is not FileDataSource."""
    with pytest.raises(TypeError, match="must be an instance of FileDataSource"):
        EpisodicDataSource(source="not_a_source", batch_size=2, episode_length=5)

def test_episodic_remainder_handling():
    """Test that data is dropped if it doesn't fit into a full batch of episodes."""
    # 1 directory, 7 files. Episode length 5.
    # Starts: 0, 1, 2 (Total 3 starts).
    # Batch size 2.
    # 3 // 2 = 1 chunk.
    # Remainder (1 start) should be dropped.
    files = [f"/d/f_{i}.mat" for i in range(7)]
    source = MockDataSource(files)
    ds = EpisodicDataSource(source, batch_size=2, episode_length=5)
    
    # 1 chunk * 2 episodes * 5 steps = 10 items
    assert len(ds) == 10

def test_episodic_delegation():
    """Test that include_images and file_list are delegated to source."""
    files = ["a", "b"]
    source = MockDataSource(files, include_images=True)
    ds = EpisodicDataSource(source, batch_size=1, episode_length=1)
    
    assert ds.include_images is True
    assert ds.file_list == files
