"""Integration tests for the unified randomness and checkpointing strategy.

These tests verify that the entire pipeline (Grain DataLoader + SynthPix Sampler) 
can be correctly checkpointed and restored using Orbax, achieving 
bit-perfect reproducibility.
"""

import jax.numpy as jnp
import pytest
import numpy as np
import scipy.io
from typing import cast

from synthpix import make, save_checkpoint
from synthpix.sampler import SyntheticImageSampler


def test_full_pipeline_checkpoint_reproducibility(mock_mat_files, tmp_path):
    """Verify bit-perfect reproducibility of the whole pipeline after Orbax restore.
    
    This test simulates a training loop where we:
    1. Initialize the pipeline (Grain + Sampler).
    2. Step for a few iterations.
    3. Save a checkpoint (Orbax).
    4. Step for a few more iterations (Ground Truth for resumption).
    5. Initialize a NEW pipeline and restore from checkpoint.
    6. Step and verify that resumed output matches Ground Truth.
    """
    paths, _ = mock_mat_files # 2 files of 64x64
    checkpoint_dir = tmp_path / "checkpoints"
    
    # Configuration
    config = {
        "scheduler_class": ".mat",
        "file_list": paths,
        "batch_size": 2,
        "flow_fields_per_batch": 1,
        "batches_per_flow_batch": 2, # Force repetitions
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
        "img_offset": (10, 10),
        "seeding_density_range": (0.01, 0.02),
        "p_hide_img1": 0.01,
        "p_hide_img2": 0.01,
        "diameter_ranges": [(1.0, 2.0)],
        "diameter_var": 0.1,
        "intensity_ranges": [(100, 200)],
        "intensity_var": 0.1,
        "rho_ranges": [(0.0, 0.0)],
        "rho_var": 0.0,
        "dt": 1.0,
        "noise_uniform": 0.0,
        "noise_gaussian_mean": 0.0,
        "noise_gaussian_std": 0.0,
        "randomize": False,
        "loop": True,
    }
    
    # 1. Initialize Pipeline
    sampler = cast(SyntheticImageSampler, make(config, use_grain_scheduler=True))
    assert isinstance(sampler, SyntheticImageSampler), "Expected SyntheticImageSampler"
    
    # 2. Run for 2 steps (Flow 0, Rep 0 and Rep 1)
    _ = next(sampler)
    _ = next(sampler)
    
    # 3. Save Checkpoint
    save_checkpoint(checkpoint_dir, sampler, step=2)
    
    # 4. Ground Truth for Resumption (Flow 1, Rep 0)
    gt_batch = next(sampler)
    gt_img_mean = float(gt_batch.images1.mean())
    gt_step = sampler._step
    
    # 5. New Pipeline Resumption
    new_sampler = cast(SyntheticImageSampler, make(config, use_grain_scheduler=True, load_from=checkpoint_dir))
    
    # 6. Verify Reproducibility
    resumed_batch = next(new_sampler)
    resumed_img_mean = float(resumed_batch.images1.mean())
    
    assert new_sampler._step == gt_step, f"Resumed step mismatch. Expected {gt_step}, got {new_sampler._step}"
    assert resumed_img_mean == pytest.approx(gt_img_mean), f"Resumed image mean mismatch. Expected {gt_img_mean}, got {resumed_img_mean}"
    assert jnp.allclose(resumed_batch.images1, gt_batch.images1), "Resumed images are not bit-perfectly identical to ground truth"
    
    # 7. Verify jax_seed and epoch preservation
    if gt_batch.seeds is not None:
        assert resumed_batch.seeds is not None, "seeds should not be None after restore"
        assert jnp.array_equal(resumed_batch.seeds, gt_batch.seeds), "seeds mismatch after restore"
    else:
        assert resumed_batch.seeds is None, "seeds should be None after restore"
    assert jnp.allclose(resumed_batch.epoch, gt_batch.epoch), f"Epoch mismatch after restore. Expected {gt_batch.epoch}, got {resumed_batch.epoch}"


def test_empty_dataset_handling(tmp_path):
    """Verify that the pipeline handles empty datasets gracefully or with clear errors."""
    # Create empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    config = {
        "scheduler_class": ".mat",
        "file_list": [], # Empty list
        "batch_size": 2,
        "flow_fields_per_batch": 1,
        "batches_per_flow_batch": 1,
        "image_shape": (32, 32),
        "flow_field_size": (64, 64),
        "randomize": False,
        "loop": False,
    }
    
    # make() should probably fail early if no files are found
    with pytest.raises((ValueError, RuntimeError), match=".*found.*|.*empty.*"):
        sampler = cast(SyntheticImageSampler, make(config, use_grain_scheduler=True))


def test_real_sampler_checkpointing(tmp_path):
    """Verify detailed checkpointing flow with generic Sampler support."""
    checkpoint_dir = tmp_path / "checkpoints_real"
    
    # 0. Setup Configuration (same as other tests)
    # We use .mat as a representative data source
    config = {
        "scheduler_class": ".mat",
        "batch_size": 2,
        "include_images": True,
        "flow_fields_per_batch": 2,
        # We need a file list for real data sources
        "file_list": ["dummy.mat"],
        "image_shape": (64, 64),
        "seed": 42
    }
    
    dummy_data = {
        "V": np.zeros((64, 64, 2)),
        "I0": np.zeros((64, 64)),
        "I1": np.zeros((64, 64)),
    }
    scipy.io.savemat(tmp_path / "dummy.mat", dummy_data)
    config["file_list"] = [str(tmp_path / "dummy.mat")]
    
    # Initialize generic sampler via make
    sampler = make(config, use_grain_scheduler=True)
    
    # Consume batches
    _ = next(sampler)
    _ = next(sampler)
    
    # Save using top-level API
    save_checkpoint(checkpoint_dir, sampler, step=2)
    
    # Resume
    resumed = make(config, use_grain_scheduler=True, load_from=checkpoint_dir)
    
    # Verify Scheduler State Match
    s1_state = sampler.scheduler.state
    s2_state = resumed.scheduler.state
    
    assert type(s1_state) == type(s2_state), "Scheduler states are not structurally compatible"
    assert s1_state == s2_state, "Scheduler states do not match after restore"
