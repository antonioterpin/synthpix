"""Integration tests for the synthetic image sampler using a
Grain-based backend.

These tests verify that `SyntheticImageSampler` can correctly
interact with `GrainEpisodicAdapter` to retrieve flow fields and
manage episodic transitions when using Google Grain's `DataLoader`.
"""

from __future__ import annotations

from pathlib import Path

import grain.python as grain
import pytest

from synthpix.data_sources.adapter import GrainEpisodicAdapter
from synthpix.data_sources.episodic import EpisodicDataSource
from synthpix.data_sources.mat import MATDataSource
from synthpix.sampler.synthetic import SyntheticImageSampler
from synthpix.scheduler.protocol import EpisodicSchedulerProtocol
from synthpix.types import ImageGenerationSpecification


@pytest.mark.parametrize("mock_mat_files", [10], indirect=True)
def test_sampler_with_grain_integration(
    tmp_path: Path, mock_mat_files: tuple[list[str], dict[str, int]]
) -> None:
    """Verify full-stack integration between Grain, the episodic adapter,
    and the sampler.

    Tests that the sampler correctly retrieves batches from a Grain
    loader and that `next_episode()` properly resets the sequence within
    the sampler loop.

    Args:
        tmp_path: Temporary directory for this test.
        mock_mat_files: Temporary MATLAB file paths and dimensions.
    """
    num_files = 10
    batch_size = 2
    episode_length = 5

    _, _ = mock_mat_files
    dataset_dir = tmp_path

    ds = MATDataSource(str(dataset_dir), include_images=False)
    assert len(ds) == num_files, (
        f"Expected {num_files} files in MATDataSource, got {len(ds)}"
    )

    episodic_ds = EpisodicDataSource(
        ds, batch_size=batch_size, episode_length=episode_length
    )

    num_episodes = num_files - episode_length + 1
    num_chunks = num_episodes // batch_size
    expected_len = num_chunks * batch_size * episode_length
    assert len(episodic_ds) == expected_len, (
        f"EpisodicDataSource length mismatch. Expected "
        f"{expected_len}, got {len(episodic_ds)}"
    )

    loader = grain.DataLoader(
        data_source=episodic_ds,
        sampler=grain.IndexSampler(
            num_records=len(episodic_ds),
            shuffle=False,
            shard_options=grain.NoSharding(),
            num_epochs=1,
        ),
        operations=[grain.Batch(batch_size=batch_size)],
    )

    adapter = GrainEpisodicAdapter(loader)

    gs = ImageGenerationSpecification(batch_size=batch_size)

    sampler = SyntheticImageSampler(
        scheduler=adapter,
        batches_per_flow_batch=1,
        flow_fields_per_batch=2,
        flow_field_size=(200.0, 200.0),
        resolution=32,
        velocities_per_pixel=1,
        seed=0,
        max_speed_x=1.0,
        max_speed_y=1.0,
        min_speed_x=0.0,
        min_speed_y=0.0,
        output_units="pixels",
        device_ids=[0],
        generation_specification=gs,
    )

    batch_data = next(sampler)
    assert batch_data is not None, "Sampler returned None instead of batch data"
    assert batch_data.flow_fields.shape == (
        batch_size,
        256,
        256,
        2,
    ), (
        f"Batch flow fields shape mismatch. Expected "
        f"{(batch_size, 256, 256, 2)}, got "
        f"{batch_data.flow_fields.shape}"
    )

    assert isinstance(sampler.scheduler, EpisodicSchedulerProtocol), (
        "Sampler scheduler must implement EpisodicSchedulerProtocol"
    )
    assert sampler.scheduler.steps_remaining() == episode_length - 1, (
        f"Expected {episode_length - 1} steps remaining, got "
        f"{sampler.scheduler.steps_remaining()}"
    )

    sampler.next_episode()

    batch_data_2 = next(sampler)
    assert batch_data_2.flow_fields.shape == (
        batch_size,
        256,
        256,
        2,
    ), (
        f"Second batch flow fields shape mismatch. Expected "
        f"{(batch_size, 256, 256, 2)}, got "
        f"{batch_data_2.flow_fields.shape}"
    )
