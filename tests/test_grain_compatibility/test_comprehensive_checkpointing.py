"""Tests for comprehensive checkpointing functionality.

These tests check the edge cases of checkpointing and resuming from a Grain iterator.
"""

import os
import shutil

import h5py
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.io

from synthpix import make, save_checkpoint


def get_valid_config(
    paths, source_type, include_images=False, episode_length=0
):
    """Returns a valid configuration dictionary for testing."""
    config = {
        "scheduler_class": source_type,
        "file_list": paths,
        "batch_size": 1,
        "flow_fields_per_batch": 1,
        "batches_per_flow_batch": 1,
        "include_images": include_images,
        "image_shape": (64, 64),
        "flow_field_size": (128, 128),  # Larger than needed to be safe
        "seed": 42,
        "loop": True,
        "img_offset": (10, 10),
        "rho_ranges": [(0.0, 0.0)],
        "rho_var": 0.0,
        "intensity_ranges": [(100, 200)],
        "intensity_var": 0.1,
        "diameter_ranges": [(1.0, 2.0)],
        "diameter_var": 0.1,
        "seeding_density_range": (0.01, 0.02),
        "p_hide_img1": 0.0,
        "p_hide_img2": 0.0,
        "min_speed_x": 0.0,
        "max_speed_x": 0.0,
        "min_speed_y": 0.0,
        "max_speed_y": 0.0,
        "noise_uniform": 0.0,
        "noise_gaussian_mean": 0.0,
        "noise_gaussian_std": 0.0,
        "output_units": "pixels",
        "dt": 1.0,
        "resolution": 1.0,
        "velocities_per_pixel": 1.0,
    }
    if episode_length > 0:
        config["episode_length"] = episode_length
    return config


@pytest.mark.parametrize("source_type", [".h5", ".mat", ".npy"])
@pytest.mark.parametrize("mode", ["non-episodic", "episodic"])
@pytest.mark.parametrize("include_images", [True, False])
def test_checkpoint_matrix(source_type, mode, include_images, tmp_path):
    """Verify bit-perfect reproducibility across data sources and modes."""
    # 1. Constraints Check
    if source_type == ".h5" and include_images:
        pytest.skip("HDF5DataSource does not support images yet.")
    if source_type == ".npy" and include_images:
        pytest.skip(
            "NumpyDataSource image extraction is not yet supported in make.py."
        )

    checkpoint_dir = tmp_path / "checkpoints"

    # 2. Prepare Data
    # 10 episodes of 10 files each = 100 files
    num_files = 100
    paths = []

    if source_type == ".h5":
        for i in range(num_files):
            p = tmp_path / f"file_{i}.h5"
            with h5py.File(p, "w") as f:
                f.create_dataset(
                    "flow", data=np.zeros((64, 64, 2), dtype=np.float32)
                )
            paths.append(str(p))
    elif source_type == ".mat":
        for i in range(num_files):
            p = tmp_path / f"file_{i}.mat"
            scipy.io.savemat(
                p,
                {
                    "V": np.zeros((64, 64, 2)),
                    "I0": np.zeros((64, 64)),
                    "I1": np.zeros((64, 64)),
                },
            )
            paths.append(str(p))
    else:  # .npy
        for i in range(num_files):
            p = tmp_path / f"flow_{i}.npy"
            np.save(p, np.zeros((64, 64, 2)))
            paths.append(str(p))

    # Organize for episodic
    if mode == "episodic":
        for i in range(10):  # 10 episodes
            epi_dir = tmp_path / f"epi{i}"
            epi_dir.mkdir(exist_ok=True)
            for j in range(10):  # 10 files per episode
                idx = i * 10 + j
                src = paths[idx]
                dst = epi_dir / os.path.basename(src)
                shutil.copy(src, dst)
        # Scan for files in subdirs
        import glob

        paths = glob.glob(str(tmp_path / "epi*/*") + source_type)

    # 3. Configuration
    config = get_valid_config(
        paths, source_type, include_images, 10 if mode == "episodic" else 0
    )

    # 4. Create Sampler and run
    sampler = make(config, use_grain_scheduler=True)

    # Run a few steps
    batch0 = next(sampler)
    batch1 = next(sampler)

    save_checkpoint(checkpoint_dir, sampler, step=2)

    # Ground truth for resumption
    gt_batch = next(sampler)

    # 5. Resume
    resumed_sampler = make(
        config, use_grain_scheduler=True, load_from=checkpoint_dir
    )
    resumed_batch = next(resumed_sampler)

    # 6. Verify
    assert jnp.allclose(resumed_batch.flow_fields, gt_batch.flow_fields)
    if include_images:
        assert jnp.allclose(resumed_batch.images1, gt_batch.images1)
        assert jnp.allclose(resumed_batch.images2, gt_batch.images2)

    assert resumed_batch.epoch == gt_batch.epoch

    if gt_batch.seeds is not None:
        assert jnp.array_equal(resumed_batch.seeds, gt_batch.seeds)
    else:
        assert resumed_batch.seeds is None


def test_legacy_scheduler_exclusion(mock_mat_files, tmp_path):
    """Verify that legacy scheduler (Grain=False) does NOT support checkpointing."""
    paths, _ = mock_mat_files
    checkpoint_dir = tmp_path / "checkpoints_legacy"

    # Use helper for valid config
    config = get_valid_config(paths, ".mat")

    # 1. Restoration failure
    # First create a valid checkpoint using Grain (we need a dir with checkpoints)
    grain_sampler = make(config, use_grain_scheduler=True)
    _ = next(grain_sampler)
    save_checkpoint(checkpoint_dir, grain_sampler, step=1)

    # Try to load into legacy sampler
    with pytest.raises(
        ValueError,
        match="Sampler does not provide access to Grain iterator|does not expose Grain iterator",
    ):
        make(config, use_grain_scheduler=False, load_from=checkpoint_dir)

    # 2. Save failure
    legacy_sampler = make(config, use_grain_scheduler=False)
    with pytest.raises(
        ValueError, match="Sampler does not provide access to Grain iterator"
    ):
        save_checkpoint(tmp_path / "fail", legacy_sampler, step=1)


def test_edge_case_modified_file_list(mock_mat_files, tmp_path):
    """Verify that restoring with a different file list fails or warns (Grain detects mismatch)."""
    paths, _ = mock_mat_files
    checkpoint_dir = tmp_path / "checkpoints_drift"

    config = get_valid_config(paths, ".mat")

    # Save checkpoint
    sampler = make(config, use_grain_scheduler=True)
    _ = next(sampler)
    save_checkpoint(checkpoint_dir, sampler, step=1)

    # Change file list in config
    new_config = dict(config)
    new_config["file_list"] = paths[:1]  # Drop one file

    # Grain validation should catch this because the DataSource or Sampler repr will change
    with pytest.raises(
        ValueError,
        match="match dataloader sampler|DataSource in checkpoint does not match",
    ):
        make(new_config, use_grain_scheduler=True, load_from=checkpoint_dir)


def test_edge_case_missing_files(mock_mat_files, tmp_path):
    """Verify behavior when files are deleted after checkpointing."""
    paths, _ = mock_mat_files
    checkpoint_dir = tmp_path / "checkpoints_missing"

    config = get_valid_config(paths, ".mat")
    config["loop"] = False  # Ensure we hit the end

    # Save checkpoint
    sampler = make(config, use_grain_scheduler=True)
    _ = next(sampler)  # Consumes flow 0
    save_checkpoint(checkpoint_dir, sampler, step=1)

    # Delete the next file (flow 1)
    os.remove(paths[1])

    # Restore
    resumed_sampler = make(
        config, use_grain_scheduler=True, load_from=checkpoint_dir
    )

    # next() should fail because it tries to load a non-existent file
    with pytest.raises(
        (FileNotFoundError, ValueError),
        match=".*not found.*|.*Failed to load.*",
    ):
        _ = next(resumed_sampler)


def test_edge_case_config_drift(mock_mat_files, tmp_path):
    """Verify behavior when critical config parameters (like seed) change."""
    paths, _ = mock_mat_files
    checkpoint_dir = tmp_path / "checkpoints_drift_params"

    config = get_valid_config(paths, ".mat")

    # Save
    sampler = make(config, use_grain_scheduler=True)
    _ = next(sampler)
    save_checkpoint(checkpoint_dir, sampler, step=1)

    # Change seed (this definitely changes IndexSampler repr)
    new_config = dict(config)
    new_config["seed"] = 99

    # Grain validation should catch the sampler repr mismatch
    with pytest.raises(
        ValueError,
        match="Sampler in checkpoint does not match dataloader sampler",
    ):
        make(new_config, use_grain_scheduler=True, load_from=checkpoint_dir)
