"""Integration tests for SyntheticImageSampler with Grain Adapter."""

import pytest
import numpy as np
import grain.python as grain
from synthpix.data_sources.adapter import GrainEpisodicAdapter
from synthpix.sampler.synthetic import SyntheticImageSampler
from synthpix.data_sources.mat import MATDataSource
from synthpix.scheduler.protocol import EpisodicSchedulerProtocol

@pytest.mark.parametrize("mock_mat_files", [10], indirect=True)
def test_sampler_with_grain_integration(tmp_path, mock_mat_files):
    """Test standard sampler loop with Grain backend."""
    num_files = 10
    batch_size = 2
    episode_length = 5

    _, _ = mock_mat_files 
    dataset_dir = tmp_path
    
    ds = MATDataSource(str(dataset_dir), include_images=False)
    assert len(ds) == num_files
    
    from synthpix.data_sources.episodic import EpisodicDataSource
    
    episodic_ds = EpisodicDataSource(ds, batch_size=batch_size, episode_length=episode_length)
    
    num_episodes = num_files - episode_length + 1
    num_chunks = num_episodes // batch_size
    expected_len = num_chunks * batch_size * episode_length 
    assert len(episodic_ds) == expected_len
    
    loader = grain.DataLoader(
        data_source=episodic_ds,
        sampler=grain.IndexSampler(
            num_records=len(episodic_ds),
            shuffle=False,
            shard_options=grain.NoSharding(),
            num_epochs=1
        ),
        operations=[grain.Batch(batch_size=batch_size)]
    )
    
    adapter = GrainEpisodicAdapter(loader)
    
    from synthpix.types import ImageGenerationSpecification
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
    assert batch_data is not None
    assert batch_data.flow_fields.shape == (batch_size, 256, 256, 2)
    
    assert isinstance(sampler.scheduler, EpisodicSchedulerProtocol)  
    assert sampler.scheduler.steps_remaining() == episode_length - 1
    
    sampler.next_episode() 
    
    batch_data_2 = next(sampler)
    assert batch_data_2.flow_fields.shape == (batch_size, 256, 256, 2)
