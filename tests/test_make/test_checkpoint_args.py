"""Tests for the checkpoint_args helper in synthpix.make."""

from unittest.mock import MagicMock

import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from synthpix.make import checkpoint_args
from synthpix.sampler import Sampler


def test_checkpoint_args_save():
    """Verify checkpoint_args returns correct SaveArgs."""
    sampler = MagicMock(spec=Sampler)
    sampler.grain_iterator = MagicMock()
    sampler.state = {"weights": 1}

    args = checkpoint_args(sampler, is_restore=False)

    assert isinstance(args, ocp.args.Composite)
    # Check if keys are present
    assert "sampler" in args
    assert "grain" in args
    assert isinstance(args["sampler"], ocp.args.StandardSave)
    assert isinstance(args["grain"], grain.PyGrainCheckpointSave)
    assert args["sampler"].item == sampler.state
    assert args["grain"].item == sampler.grain_iterator


def test_checkpoint_args_restore():
    """Verify checkpoint_args returns correct RestoreArgs."""
    sampler = MagicMock(spec=Sampler)
    sampler.grain_iterator = MagicMock()
    sampler.restore_state = {"weights": None}

    args = checkpoint_args(sampler, is_restore=True)

    assert isinstance(args, ocp.args.Composite)
    # Check if keys are present
    assert "sampler" in args
    assert "grain" in args
    assert isinstance(args["sampler"], ocp.args.StandardRestore)
    assert isinstance(args["grain"], grain.PyGrainCheckpointRestore)
    assert args["sampler"].item == sampler.restore_state
    assert args["grain"].item == sampler.grain_iterator


def test_checkpoint_args_no_iterator():
    """Verify checkpoint_args raises ValueError if grain_iterator is None."""
    sampler = MagicMock(spec=Sampler)
    sampler.grain_iterator = None

    with pytest.raises(
        ValueError, match="Sampler does not provide access to Grain iterator"
    ):
        checkpoint_args(sampler)


def test_checkpoint_args_restore_resolves_placeholders():
    """Placeholder leaves take the saved shape, or None if None was saved."""
    sampler = MagicMock(spec=Sampler)
    sampler.grain_iterator = MagicMock()
    sampler.restore_state = {
        "files_scheduler": jax.ShapeDtypeStruct((np.nan,), jnp.uint8),
        "unsaved": jax.ShapeDtypeStruct((np.nan,), jnp.uint8),
        "guessed": jax.ShapeDtypeStruct((2, 8, 8, 2), jnp.float32),
        "concrete": jnp.zeros(3),
    }
    saved_metadata = {
        "files_scheduler": jax.ShapeDtypeStruct((34,), jnp.uint8),
        "unsaved": None,
        "guessed": jax.ShapeDtypeStruct((1, 8, 8, 2), jnp.float32),
        "concrete": jax.ShapeDtypeStruct((3,), jnp.float32),
    }

    args = checkpoint_args(
        sampler, is_restore=True, saved_metadata=saved_metadata
    )

    template = args["sampler"].item
    assert template["files_scheduler"] == jax.ShapeDtypeStruct((34,), jnp.uint8)
    assert template["unsaved"] is None
    assert template["guessed"] == jax.ShapeDtypeStruct(
        (1, 8, 8, 2), jnp.float32
    )
    assert template["concrete"] is sampler.restore_state["concrete"]


def test_checkpoint_args_restore_without_metadata_keeps_template():
    """Without saved metadata the restore template is passed through as-is."""
    sampler = MagicMock(spec=Sampler)
    sampler.grain_iterator = MagicMock()
    placeholder = jax.ShapeDtypeStruct((np.nan,), jnp.uint8)
    sampler.restore_state = {"files_scheduler": placeholder}

    args = checkpoint_args(sampler, is_restore=True)

    assert args["sampler"].item["files_scheduler"] is placeholder
