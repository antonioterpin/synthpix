"""Tests for EpisodicDataSource.

EpisodicDataSource wraps a FileDataSource and provides an episodic view of the data. 
It supports interleaving multiple episodes into mini-batches and ensures that 
timesteps within an episode are yielded sequentially.
"""

import os

import grain.python as grain
import numpy as np
import pytest

from synthpix.data_sources import EpisodicDataSource, FileDataSource


class MockDataSource(FileDataSource):
    """Simple source that returns the filepath."""

    def __init__(self, file_list, include_images=False):
        super().__init__(dataset_path=file_list)
        self._include_images = include_images

    def load_file(self, file_path):
        return {"file": file_path}

    @property
    def include_images(self) -> bool:
        return self._include_images


def test_episodic_interleaving(tmp_path):
    """Test that multiple episodes are correctly interleaved in mini-batches.

    This test sets up a mock file system with two directories (representing two 
    sequences), and verifies that:
    1. The data source length accounts for all possible episode starts and padding.
    2. Items at the same index across a 'chunk' represent the same timestep (t=0) 
       across different episodes.
    3. Sequential indices within a 'chunk' follow the temporal order of a single 
       episode.
    4. Metadata like '_timestep' and '_chunk_id' are correctly populated.
    """
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
    assert len(ds) == 60, f"Expected 60 total items, got {len(ds)}"

    # Check item 0 and 1 -> Should be t=0 of different episodes
    item0 = ds[0]
    item1 = ds[1]

    assert item0["_timestep"] == 0, f"Initial timestep for item0 should be 0, got {item0['_timestep']}"
    assert item1["_timestep"] == 0, f"Initial timestep for item1 should be 0, got {item1['_timestep']}"
    assert item0["_chunk_id"] == 0, f"Initial chunk ID for item0 should be 0, got {item0['_chunk_id']}"
    assert item1["_chunk_id"] == 0, f"Initial chunk ID for item1 should be 0, got {item1['_chunk_id']}"

    # Check item 2 -> Should be t=1 of first episode in chunk
    item2 = ds[2]
    assert item2["_timestep"] == 1, f"Timestep for item2 should be 1, got {item2['_timestep']}"
    assert item2["_chunk_id"] == 0, f"Chunk ID for item2 should be 0, got {item2['_chunk_id']}"

    # Ensure files are actually sequential in time
    f0 = item0["file"]
    f2 = item2["file"]

    # Extract index from filename "f_XX.mat"
    def extract_idx(f):
        return int(os.path.basename(f).split("_")[1].split(".")[0])

    idx0 = extract_idx(str(f0))
    idx2 = extract_idx(str(f2))
    assert idx2 == idx0 + 1, f"Files in episode should be sequential. Expected {idx0 + 1}, got {idx2}"
    assert os.path.dirname(str(f0)) == os.path.dirname(str(f2)), "Sequential items within a chunk should come from the same directory"


def test_episodic_grain_integration(tmp_path):
    """Verify it actually works efficiently with Grain DataLoader."""
    d = tmp_path / "d"
    d.mkdir()
    files = []
    for i in range(20):
        p = d / f"f_{i:02d}.mat"
        p.touch()
        files.append(str(p))

    source = MockDataSource(files)

    # 20 files, len 5 -> 16 starts
    # batch 4 -> 4 chunks
    ds = EpisodicDataSource(source, batch_size=4, episode_length=5)

    # Use simple sequential sampler
    loader = grain.DataLoader(
        data_source=ds,
        sampler=grain.IndexSampler(
            len(ds),
            shuffle=False,
            shard_options=grain.NoSharding(),
            num_epochs=1,
        ),
        operations=[grain.Batch(batch_size=4)],
        worker_count=0,
    )

    batch = next(iter(loader))

    # Batch should contain 4 items.
    timesteps = batch["_timestep"]  # expected shape (4,)
    assert np.all(timesteps == 0), f"First batch should have all timesteps=0, got {timesteps}"

    # Next batch
    it = iter(loader)
    next(it)  # t=0
    batch_t1 = next(it)  # t=1
    assert np.all(batch_t1["_timestep"] == 1), f"Second batch should have all timesteps=1, got {batch_t1['_timestep']}"


def test_episodic_type_validation():
    """Test that TypeError is raised if source is not FileDataSource."""
    with pytest.raises(
        TypeError, match="must be an instance of FileDataSource"
    ):
        EpisodicDataSource(
            source="not_a_source", batch_size=2, episode_length=5  # type: ignore
        )


def test_episodic_wrap_around_padding(tmp_path):
    """Test that data is wrapped around if it doesn't fit into a full batch."""
    # 1 directory, 7 files. Episode length 5.
    d = tmp_path / "d"
    d.mkdir()
    files = []
    for i in range(7):
        p = d / f"f_{i}.mat"
        p.touch()
        files.append(str(p))
    
    source = MockDataSource(files)
    ds = EpisodicDataSource(source, batch_size=2, episode_length=5)

    # 2 chunks * 2 episodes * 5 steps = 20 items
    assert len(ds) == 20, f"Expected 20 items after wrap-around padding, got {len(ds)}"


def test_episodic_delegation(tmp_path):
    """Test that include_images and file_list are delegated to source."""
    d = tmp_path / "d"
    d.mkdir()
    f1 = d / "a"
    f2 = d / "b"
    f1.touch()
    f2.touch()
    files = [str(f1), str(f2)]
    
    source = MockDataSource(files, include_images=True)
    ds = EpisodicDataSource(source, batch_size=1, episode_length=1)

    assert ds.include_images is True, "include_images should be delegated to source"
    assert ds.file_list == files, "file_list should be delegated to source"


def test_episodic_no_valid_episodes(tmp_path):
    """Test that ValueError is raised if no valid episodes are found."""
    # 3 files, episode length 5 -> No valid starts
    d = tmp_path / "d"
    d.mkdir()
    files = []
    for i in range(3):
        p = d / f"f_{i}.mat"
        p.touch()
        files.append(str(p))
    
    source = MockDataSource(files)

    with pytest.raises(ValueError, match="No valid episodes found"):
        EpisodicDataSource(source, batch_size=2, episode_length=5)


def test_episodic_invalid_directory():
    """Test that ValueError is raised if a file path points to a non-existent directory.

    EpisodicDataSource groups files by their parent directory to identify 
    sequences. If a file path is provided whose directory cannot be resolved, 
    it should raise a ValueError.
    """
    files = ["/non_existent_dir/f_0.mat"]
    source = MockDataSource(files)

    with pytest.raises(ValueError, match="Directory not found"):
        EpisodicDataSource(source, batch_size=1, episode_length=1)


# ──────────────────────────────────────────────────────────────────────────────
# Stress Tests (Corner Cases)
# ──────────────────────────────────────────────────────────────────────────────

class MockFileSource(FileDataSource):
    """Simple source that returns metadata for verification."""

    def __init__(self, file_list):
        super().__init__(dataset_path=file_list)

    def load_file(self, file_path):
        return {"file": file_path}

    @property
    def include_images(self) -> bool:
        return False


def create_mock_dataset(tmp_path, folder_structure):
    """Helper to create a nested directory structure and empty mock files.
    
    `folder_structure` is a dict mapping subfolder names to the number of 
    `.mat` files to create in each. Returns a sorted list of absolute paths.
    """
    files = []
    for folder_name, num_files in folder_structure.items():
        d = tmp_path / folder_name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(num_files):
            f = d / f"f_{i:04d}.mat"
            f.touch()
            files.append(str(f))
    return sorted(files)


def test_padding_flag_accuracy(tmp_path):
    """Verify that `_is_padding` is correctly flagged for interleaving chunks.

    When the total number of episodes isn't a multiple of the batch size, 
    the `EpisodicDataSource` should fill the remainder with padded episodes 
    marked with `_is_padding=True`.
    """
    # 1 directory, 3 files, episode_length 2 -> 2 possible starts (0, 1)
    # batch_size 3 -> needs 2 chunks of 3 episodes = 6 episodes total.
    # 2 real episodes, 4 padded episodes.
    files = create_mock_dataset(tmp_path, {"seq": 3})
    source = MockFileSource(files)
    
    batch_size = 3
    episode_length = 2
    ds = EpisodicDataSource(source, batch_size=batch_size, episode_length=episode_length, seed=42)
    
    # 1 chunk * 3 ep * 2 steps = 6 items
    assert len(ds) == 6, f"Expected 6 items for 1 chunk of size 3 and length 2, got {len(ds)}"
    
    loader = grain.DataLoader(
        data_source=ds,
        sampler=grain.IndexSampler(len(ds), shuffle=False, shard_options=grain.NoSharding(), num_epochs=1),
        operations=[grain.Batch(batch_size=batch_size)],
    )
    
    all_is_padding = []
    for batch in loader:
        all_is_padding.append(batch["_is_padding"])
    
    # Expected: 1 chunk of 2 batches.
    # Chunk 0:
    # Batch 0 (t=0): [Ep0, Ep1, Padded_Ep0]
    # Batch 1 (t=1): [Ep0, Ep1, Padded_Ep0]
    
    is_padding_flat = np.concatenate(all_is_padding)
    # 2 batches * 3 elements per batch = 6
    
    # Each real episode has 2 steps. We have 2 real episodes.
    assert np.sum(~is_padding_flat) == 2 * episode_length, f"Expected 4 non-padding steps, got {np.sum(~is_padding_flat)}"
    # Each padded episode has 2 steps. We have 1 padded episode.
    assert np.sum(is_padding_flat) == 1 * episode_length, f"Expected 2 padding steps, got {np.sum(is_padding_flat)}"


def test_is_last_step_precision(tmp_path):
    """Verify that `_is_last_step` is only True at the conclusion of an episode.

    Checks the logic for tracking timesteps and ensuring the signal for 
    episode completion is accurate for all batches.
    """
    files = create_mock_dataset(tmp_path, {"seq": 5})
    source = MockFileSource(files)
    episode_length = 3
    ds = EpisodicDataSource(source, batch_size=1, episode_length=episode_length)
    
    loader = grain.DataLoader(
        data_source=ds,
        sampler=grain.IndexSampler(len(ds), shuffle=False, shard_options=grain.NoSharding(), num_epochs=1),
        operations=[grain.Batch(batch_size=1)],
    )
    
    for batch in loader:
        t = batch["_timestep"][0]
        last = batch["_is_last_step"][0]
        if t == episode_length - 1:
            assert last == True, f"Expected _is_last_step to be True at timestep {t} (end of length {episode_length} episode)"
        else:
            assert last == False, f"Expected _is_last_step to be False before end of episode (timestep {t})"


def test_asymmetric_folder_lengths(tmp_path):
    """Verify that folders with different file counts contribute properly to starts.

    The number of possible episode starts depends on the length of each 
    sequence relative to the required `episode_length`.
    """
    folder_structure = {
        "long": 10,  # 10 - 5 + 1 = 6 starts
        "short": 5   # 5 - 5 + 1 = 1 start
    }
    files = create_mock_dataset(tmp_path, folder_structure)
    source = MockFileSource(files)
    ds = EpisodicDataSource(source, batch_size=1, episode_length=5)
    
    # Total starts should be 7
    assert len(ds._starts) == 7, f"Expected 7 total starts, got {len(ds._starts)}"
    
    # Count occurrences of каждой dir in starts
    dirs = [s[0] for s in ds._starts]
    assert dirs.count(str(tmp_path / "long")) == 6, f"Expected 6 starts from 'long' folder, got {dirs.count(str(tmp_path / 'long'))}"
    assert dirs.count(str(tmp_path / "short")) == 1, f"Expected 1 start from 'short' folder, got {dirs.count(str(tmp_path / 'short'))}"


def test_extreme_padding(tmp_path):
    """Verify behavior when the batch size exceeds the total available episodes.

    The source should handle large amounts of padding gracefully, 
    filling almost the entire chunk with marked padded episodes.
    """
    files = create_mock_dataset(tmp_path, {"seq": 5})
    source = MockFileSource(files)
    # 5 files, length 5 -> 1 episode start
    batch_size = 8
    ds = EpisodicDataSource(source, batch_size=batch_size, episode_length=5)
    
    # 1 started episode, needs 8 to fill one chunk.
    # 1 chunk * 8 elements * 5 steps = 40
    assert len(ds) == 40, f"Expected 40 items (1 chunk of 8 episodes * 5 steps), got {len(ds)}"
    
    batch = ds[0] # t=0
    assert batch["_is_padding"] == False, "First episode in chunk should not be padded"
    
    batch_padded = ds[1] # t=0 of second (padded) episode
    assert batch_padded["_is_padding"] == True, "Second episode in chunk should be padded"


def test_reproducibility(tmp_path):
    """Verify that the random interleaving is deterministic given a seed.

    Multiple instances of `EpisodicDataSource` initialized with the same 
    parameters and seed must yield the exact same sequence of files and 
    chunk assignments.
    """
    files = create_mock_dataset(tmp_path, {"seq1": 5, "seq2": 5})
    source = MockFileSource(files)
    
    ds1 = EpisodicDataSource(source, batch_size=2, episode_length=3, seed=123)
    ds2 = EpisodicDataSource(source, batch_size=2, episode_length=3, seed=123)
    
    assert len(ds1) == len(ds2), "Lengths of identical EpisodicDataSources should match"
    for i in range(len(ds1)):
        assert ds1[i]["file"] == ds2[i]["file"], f"File mismatch at index {i} between identical seeds"
        assert ds1[i]["_chunk_id"] == ds2[i]["_chunk_id"], f"Chunk ID mismatch at index {i} between identical seeds"


def test_multiple_epochs_total_count(tmp_path):
    """Verify that the total number of batches matches the epoch-chunk product.

    Ensures that Grain correctly handles multiple passes over the 
    `EpisodicDataSource` while respecting the underlying chunk structure.
    """
    files = create_mock_dataset(tmp_path, {"seq1": 5})
    source = MockFileSource(files)
    # 1 episode start (5 files, len 5)
    # batch_size 2 -> 1 chunk (padded to 2 episodes)
    # 1 chunk * 2 elements * 5 steps = 10 items
    ds = EpisodicDataSource(source, batch_size=2, episode_length=5)
    
    num_epochs = 3
    loader = grain.DataLoader(
        data_source=ds,
        sampler=grain.IndexSampler(len(ds), shuffle=False, shard_options=grain.NoSharding(), num_epochs=num_epochs),
        operations=[grain.Batch(batch_size=2)],
    )
    
    batches = list(loader)
    # 1 chunk per epoch * 5 steps per chunk = 5 batches per epoch
    # 3 epochs -> 15 batches
    assert len(batches) == 15, f"Expected 15 total batches for 3 epochs of 5 steps, got {len(batches)}"
