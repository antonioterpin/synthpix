"""Tests for the checkpoint_args helper in synthpix.make."""

from unittest.mock import MagicMock

import grain.python as grain
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
