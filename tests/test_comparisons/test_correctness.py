"""Correctness comparison between Legacy and Grain schedulers."""

import h5py
import grain.python as grain
import numpy as np
import pytest

from synthpix.data_sources.adapter import GrainEpisodicAdapter
from synthpix.data_sources.episodic import EpisodicDataSource
from synthpix.data_sources.mat import MATDataSource
from synthpix.scheduler.episodic import EpisodicFlowFieldScheduler
from synthpix.scheduler.mat import MATFlowFieldScheduler


@pytest.mark.parametrize(
    "batch_size, episode_length",
    [
        (1, 10),    # Standard
        (1, 1),     # Single-frame episodes
        (1, 7),     # Odd episode length
    ],
)
def test_legacy_vs_grain_equality_simple(
    tmp_path, mat_test_dims, batch_size, episode_length
):
    """Verify that Legacy and Grain stacks yield identical results for a simple case.

    Generates a single episode of data and confirms that both the old 
    `EpisodicFlowFieldScheduler` and the new `GrainEpisodicAdapter` 
    produce bit-identical flow field batches.
    """
    # Create exactly episode_length files in one directory
    num_files = episode_length
    h, w = mat_test_dims["height"], mat_test_dims["width"]
    
    # We create a subdirectory to ensure it's treated as an episode
    episode_dir = tmp_path / "episode_0"
    episode_dir.mkdir()

    for t in range(1, num_files + 1):
        mat_path = episode_dir / f"flow_{t:04d}.mat"
        with h5py.File(mat_path, "w", libver="latest", userblock_size=512) as f:
            f.create_dataset("I0", data=np.full((h, w), t, dtype=np.uint8))
            f.create_dataset("I1", data=np.full((h, w), t + 1, dtype=np.uint8))
            f.create_dataset("V", data=np.full((h, w, 2), t / 10.0, dtype=np.float32))
        
        # Fake MATLAB header
        with open(mat_path, "r+b") as fp:
            fp.write(b"MATLAB 7.3 MAT-file".ljust(512, b" "))

    # Legacy Stack
    legacy_base = MATFlowFieldScheduler(
        file_list=[str(tmp_path)],
        include_images=True,
        output_shape=(h, w),
        randomize=False,
    )
    legacy_episodic = EpisodicFlowFieldScheduler(
        scheduler=legacy_base,
        batch_size=batch_size,
        episode_length=episode_length,
    )

    # Grain Stack
    grain_source = MATDataSource(
        dataset_path=str(tmp_path),
        include_images=True,
        output_shape=(h, w),
    )
    grain_episodic_ds = EpisodicDataSource(
        source=grain_source,
        batch_size=batch_size,
        episode_length=episode_length,
        seed=42,
    )

    grain_loader = grain.DataLoader(
        data_source=grain_episodic_ds,
        sampler=grain.IndexSampler(
            num_records=len(grain_episodic_ds),
            shuffle=False,
            shard_options=grain.NoSharding(),
            num_epochs=1,
        ),
        operations=[grain.Batch(batch_size=batch_size, drop_remainder=False)],
    )

    grain_adapter = GrainEpisodicAdapter(grain_loader)

    legacy_episodic.reset()
    grain_adapter.reset()

    for i in range(episode_length):
        legacy_batch = legacy_episodic.get_batch(batch_size)
        grain_batch = grain_adapter.get_batch(batch_size)

        np.testing.assert_allclose(
            legacy_batch.flow_fields,
            grain_batch.flow_fields,
            err_msg=f"Step {i}: Flow fields mismatch",
        )


@pytest.mark.parametrize("mock_mat_files", [20], indirect=True)
def test_grain_ordering(tmp_path, mock_mat_files):
    """Verify that the Grain adapter maintains strict temporal order of data.

    In single-worker mode (`worker_count=0`), batches produced by Grain 
    should follow the expected sequence of timesteps within each 
    interleaved chunk, ensuring no frame shuffling occurs.
    """
    dataset_dir = tmp_path
    episode_length = 5
    batch_size = 2

    source = MATDataSource(str(dataset_dir), include_images=False)
    episodic_ds = EpisodicDataSource(
        source,
        batch_size=batch_size,
        episode_length=episode_length,
        seed=42,
    )
    loader = grain.DataLoader(
        data_source=episodic_ds,
        sampler=grain.IndexSampler(
            num_records=len(episodic_ds),
            shuffle=False,
            shard_options=grain.NoSharding(),
            num_epochs=1,
        ),
        operations=[
            grain.Batch(batch_size=batch_size, drop_remainder=False)
        ],
        worker_count=0,
    )
    
    trace = []
    for batch in loader:
        t_vals = batch["_timestep"]
        chunk_vals = batch["_chunk_id"]
        # Batch elements have the same timestep in EpisodicDataSource
        trace.append((int(chunk_vals[0]), int(t_vals[0])))

    # Verify Order of batches
    for i in range(1, len(trace)):
        prev_c, prev_t = trace[i - 1]
        curr_c, curr_t = trace[i]

        if prev_c == curr_c:
            expected_t = prev_t + 1
            assert curr_t == expected_t, f"Order violation at batch {i}: Chunk {prev_c} t={prev_t} -> t={curr_t}"
        else:
            assert curr_t == 0, f"New chunk {curr_c} should start at t=0, got t={curr_t}"
