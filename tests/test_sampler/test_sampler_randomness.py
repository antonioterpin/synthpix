"""Unit tests for the hybrid randomness logic in SyntheticImageSampler.

These tests verify that the sampler correctly handles deterministic randomness
from both Grain-provided seeds (modern path) and its own internal RNG (legacy path).
They also ensure that the sampler's state (RNG and step counter) is correctly
managed for checkpointing reproducibility using the high-level `make` API.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import jax

from typing import cast, Any
from synthpix import make, save_checkpoint
from synthpix.sampler import SyntheticImageSampler
from synthpix.scheduler import SchedulerProtocol
from synthpix.types import SchedulerData, ImageGenerationSpecification

@pytest.fixture
def dummy_data(tmp_path):
    """Creates a dummy .npy dataset for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create 2 files, 64x64x2
    file1 = data_dir / "file1.npy"
    file2 = data_dir / "file2.npy"
    np.save(file1, np.random.randn(64, 64, 2).astype(np.float32))
    np.save(file2, np.random.randn(64, 64, 2).astype(np.float32))
    
    return [str(file1), str(file2)]


@pytest.fixture
def base_config(dummy_data):
    """Base configuration for tests."""
    return {
        "scheduler_class": ".npy",
        "file_list": dummy_data,
        "batch_size": 2,
        "image_shape": (32, 32),
        "flow_field_size": (64, 64),
        "resolution": 1.0,
        "velocities_per_pixel": 1.0,
        "seed": 42,
        "max_speed_x": 0.0,
        "max_speed_y": 0.0,
        "min_speed_x": 0.0,
        "min_speed_y": 0.0,
        "output_units": "pixels",
        "batches_per_flow_batch": 1,
        "flow_fields_per_batch": 2, # Match batch_size for simplicity
        "img_offset": (0, 0),
        "seeding_density_range": (0.01, 0.01),
        "p_hide_img1": 0.0,
        "p_hide_img2": 0.0,
        "diameter_ranges": [(1.0, 1.0)],
        "diameter_var": 0.0,
        "intensity_ranges": [(100, 100)],
        "intensity_var": 0.0,
        "rho_ranges": [(0.0, 0.0)],
        "rho_var": 0.0,
        "dt": 1.0,
        "noise_uniform": 0.0,
        "noise_gaussian_mean": 0.0,
        "noise_gaussian_std": 0.0,
        "randomize": False,
        "loop": True,
    }


def test_sampler_grain_randomness_path(base_config):
    """Verify that the sampler uses Grain-provided seeds when available."""
    # Initialize with Grain scheduler
    sampler = cast(SyntheticImageSampler, make(base_config, use_grain_scheduler=True))
    
    # 1. Generate batch with Grain seeds
    batch1 = next(sampler)
    assert batch1.seeds is not None, "Batch should have seeds if provided by scheduler"
    assert sampler._batches_generated == 1, f"Expected 1 batch generated, got {sampler._batches_generated}"
    
    # 2. Verify that another batch with SAME seeds results in SAME images 
    # (if we reset the repetition counter manually, simulating flow reuse logic)
    # Note: make() returns a fully configured sampler. To test internal logic:
    sampler._batches_generated = 0
    batch2 = next(sampler)
    
    assert jnp.allclose(batch1.images1, batch2.images1), "Images should be identical if seeds and repetition index match"
    
    # 3. Verify that changing the repetition counter results in DIFFERENT images
    sampler.batches_per_flow_batch = 10 
    sampler._batches_generated = 5
    batch3 = next(sampler)
    assert not jnp.allclose(batch1.images1, batch3.images1), "Images should be different if repetition index changes"


def test_sampler_legacy_randomness_path(base_config):
    """Verify that the sampler falls back to internal RNG if using legacy scheduler."""
    # Initialize with Legacy scheduler
    sampler = cast(SyntheticImageSampler, make(base_config, use_grain_scheduler=False))
    
    # Generate two batches
    batch1 = next(sampler)
    batch2 = next(sampler)
    
    # Legacy path usually doesn't provide seeds in batch (it's None)
    
    assert batch1.seeds is None or jnp.all(batch1.seeds == None), "Batch should have None seeds from legacy scheduler"
    assert not jnp.allclose(batch1.images1, batch2.images1), "Images should be different (randomized by internal RNG)"


def test_sampler_checkpoint_state_consistency(base_config, tmp_path):
    """Verify that get_state (via make) preserves state structure."""
    sampler = cast(SyntheticImageSampler, make(base_config, use_grain_scheduler=True))
    
    # Run a few steps
    _ = next(sampler)
    _ = next(sampler)
    
    state = sampler.state
    # Check keys expected for checkpointing
    assert "step" in state
    assert "rng" in state
    assert "batches_generated" in state
    assert "current_flows" in state
    assert "scheduler_state" in state


def test_sampler_bit_perfect_reproducibility(base_config, tmp_path):
    """Verify bit-perfect reproducibility using make API."""
    checkpoint_dir = tmp_path / "checkpoints_repro"
    
    # 1. Ground Truth Run
    sampler = cast(SyntheticImageSampler, make(base_config, use_grain_scheduler=True))
    _ = next(sampler)
    _ = next(sampler)
    gt_batch = next(sampler)
    gt_val = gt_batch.images1.mean()
    gt_step = sampler._step
    
    # 2. Resettable Run (Simulate Checkpoint)
    # Re-init sampler to start from 0
    sampler = make(base_config, use_grain_scheduler=True)
    _ = next(sampler)
    _ = next(sampler)
    
    # Save Checkpoint using API
    save_checkpoint(checkpoint_dir, sampler, step=2)
    
    # 3. Restore using API
    resumed_sampler = cast(SyntheticImageSampler, make(base_config, use_grain_scheduler=True, load_from=checkpoint_dir))
    
    # The next batch should be IDENTICAL to gt_batch
    resumed_batch = next(resumed_sampler)
    resumed_val = resumed_batch.images1.mean()
    
    assert jnp.allclose(gt_batch.images1, resumed_batch.images1), "Images should be bit-perfectly identical after restore"
    assert float(resumed_val) == pytest.approx(float(gt_val)), f"Mean value mismatch"
    assert resumed_sampler._step == gt_step


def test_sampler_repetition_logic_across_resumes(base_config, tmp_path):
    """Verify repetition index restoration using make API."""
    checkpoint_dir = tmp_path / "checkpoints_reps"
    
    # Force reuse config
    config = base_config.copy()
    config["batches_per_flow_batch"] = 4
    
    sampler = cast(SyntheticImageSampler, make(config, use_grain_scheduler=True))
    
    # Step 0: Batch 0 (Rep 0)
    _ = next(sampler)
    assert sampler._batches_generated == 1
    
    # Save using API
    save_checkpoint(checkpoint_dir, sampler, step=1)
    
    # Resume using API
    new_sampler = cast(SyntheticImageSampler, make(config, use_grain_scheduler=True, load_from=checkpoint_dir))
    
    assert new_sampler._current_flows is not None, "Cache should be restored"
    assert new_sampler._batches_generated == 1, "Repetition counter restored"
    
    # Step 1: Batch 1 (Rep 1)
    # Should continue using same flow field
    # We can check flow field equality if we captured it before save
    # But here we verify behavior and counter.
    
    # Capture original flow BEFORE save (hacky access for test)
    original_flows = sampler._current_flows
    
    batch = next(new_sampler)
    assert new_sampler._batches_generated == 2
    
    # Verify flows match original
    assert original_flows is not None
    assert new_sampler._current_flows is not None
    assert jnp.array_equal(new_sampler._current_flows, original_flows), "Flow fields should match across resume within repetition block"


class MockScheduler(SchedulerProtocol):
    def __init__(self, flow_fields_per_batch, flow_shape=(20, 20, 2)):
        self.flow_fields_per_batch = flow_fields_per_batch
        self.flow_shape = flow_shape
        self._file_list = []
        self._state = {}

    def get_flow_fields_shape(self):
        # returns (H, W, 2)
        return self.flow_shape

    def get_batch(self, batch_size):
        # Returns SchedulerData
        # Flow fields: (B, H, W, 2)
        flows = np.zeros((batch_size, *self.flow_shape))
        
        # jax_seed: (B,) scalar seeds
        # We start with different seeds to ensure base randomness, 
        # but the test logic relies on tiling to create duplicates.
        jax_seeds = np.arange(batch_size, dtype=np.uint32)
        
        # Or returns keys to verify key support
        # jax_seeds = jax.random.split(jax.random.PRNGKey(0), batch_size) 
        
        return SchedulerData(
            flow_fields=flows,
            jax_seed=jax_seeds,
        )

    def shutdown(self) -> None:
        pass

    def reset(self) -> None:
        pass

    @property
    def file_list(self) -> list[str]:
        return self._file_list

    @file_list.setter
    def file_list(self, value: list[str]) -> None:
        self._file_list = value

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._state = value

    @property
    def grain_iterator(self) -> Any | None:
        return None

def test_jax_seeds_uniqueness():
    """Test each image and jax seed are different""" 
    # flow_fields_per_batch = 1
    # batch_size = 4
    # This implies that the single flow field and its corresponding seed 
    # will be tiled/repeated 4 times in the batch expansion.
    # We want to verify that despite using the same seed due to tiling,
    # the generated output (specifically random parameters) differs across the batch.
    
    batch_size = 4
    flow_fields_per_batch = 1
    
    scheduler = MockScheduler(flow_fields_per_batch, flow_shape=(20, 20, 2))
    
    spec = ImageGenerationSpecification(
        batch_size=batch_size,
        image_shape=(10, 10),
        img_offset=(0.2, 0.2), 
        dt=0.1,
        # Ensure some randomness in parameters to check
        intensity_var=1.0, 
        diameter_var=1.0,
    )
    
    sampler = SyntheticImageSampler(
        scheduler=scheduler,
        batches_per_flow_batch=1,
        flow_fields_per_batch=flow_fields_per_batch,
        flow_field_size=(20.0, 20.0), 
        resolution=1.0,
        velocities_per_pixel=1.0,
        seed=42,
        max_speed_x=1.0,
        max_speed_y=1.0,
        min_speed_x=0.0,
        min_speed_y=0.0,
        output_units="pixels",
        generation_specification=spec,
        # Use single device logic - sharding is handled internally but with 1 device it's just batch
        device_ids=[0],
    )
    
    # Get a batch
    batch = sampler._get_next()
    
    # flow_fields should all be identical (tiled)
    assert np.allclose(batch.flow_fields[0], batch.flow_fields[1]), "Flow fields should be identical due to tiling"
    
    params = batch.params
    # Seeding densities: sampled per image.
    seeding_densities = params.seeding_densities
    
    # Verify they are NOT all identical
    # If keys were identical, these random samples would be identical.
    print(f"Seeding densities: {seeding_densities}")
    assert not np.allclose(seeding_densities[0], seeding_densities[1]), "Seeding densities should differ if keys are unique"
    
    # Also verify that we support keys as seeds
    # Modify scheduler to return keys
    scheduler_keys = MockScheduler(flow_fields_per_batch, flow_shape=(20, 20, 2))
    def get_batch_keys(bs):
        flows = np.zeros((bs, 20, 20, 2))
        jax_seeds = jax.random.split(jax.random.PRNGKey(0), bs)
        return SchedulerData(flow_fields=flows, jax_seed=jax_seeds)
    scheduler_keys.get_batch = get_batch_keys
    
    sampler_keys = SyntheticImageSampler(
        scheduler=scheduler_keys,
        batches_per_flow_batch=1,
        flow_fields_per_batch=flow_fields_per_batch,
        flow_field_size=(20.0, 20.0), 
        resolution=1.0,
        velocities_per_pixel=1.0,
        seed=42,
        max_speed_x=1.0,
        max_speed_y=1.0,
        min_speed_x=0.0,
        min_speed_y=0.0,
        output_units="pixels",
        generation_specification=spec,
        device_ids=[0],
    )
    
    # Should not crash
    batch_keys = sampler_keys._get_next()
    print("Successfully generated batch using keys as seeds")

