"""Unit tests for the hybrid randomness logic in SyntheticImageSampler.

These tests verify that the sampler correctly handles deterministic randomness
from both Grain-provided seeds (modern path) and its own internal RNG (legacy path).
They also ensure that the sampler's state (RNG and step counter) is correctly
managed for checkpointing reproducibility using the high-level `make` API.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pathlib import Path

from typing import cast
from synthpix import make, save_checkpoint
from synthpix.sampler import SyntheticImageSampler


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
    assert batch1.jax_seed is not None, "Batch should have jax_seed if provided by scheduler"
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
    
    # Legacy path usually doesn't provide jax_seed in batch (it's None)
    
    assert batch1.jax_seed is None or jnp.all(batch1.jax_seed == None), "Batch should have None jax_seed from legacy scheduler"
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
