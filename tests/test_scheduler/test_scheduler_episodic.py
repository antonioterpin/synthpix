"""Tests for the episodic scheduling wrapper.

These tests verify that `EpisodicFlowFieldScheduler` correctly
groups consecutive frames into episodes, validates batch and episode
lengths, and reports the correct remaining steps during iteration
over high-level data sources.
"""

from __future__ import annotations

from typing import Any

import pytest

from synthpix.scheduler import EpisodicFlowFieldScheduler, MATFlowFieldScheduler


@pytest.mark.parametrize(
    "invalid_scheduler", [None, "not_a_scheduler", 123, [], {}]
)
@pytest.mark.parametrize("mock_mat_files", [64], indirect=True)
def test_invalid_scheduler(
    invalid_scheduler: Any, mock_mat_files: tuple[list[str], dict[str, int]]
) -> None:
    """Test that `EpisodicFlowFieldScheduler` rejects invalid base
    schedulers.

    Expects an instance of `BaseFlowFieldScheduler`.

    Args:
        invalid_scheduler: Invalid scheduler used to check validation.
        mock_mat_files: Temporary MATLAB file paths and dimensions.
    """
    _files, _dims = mock_mat_files

    with pytest.raises(TypeError):
        EpisodicFlowFieldScheduler(
            invalid_scheduler, batch_size=1, episode_length=2
        )


@pytest.mark.parametrize("invalid_batch_size", [-1, 0, 1.5, "two"])
@pytest.mark.parametrize("mock_mat_files", [64], indirect=True)
def test_invalid_batch_size(
    invalid_batch_size: Any, mock_mat_files: tuple[list[str], dict[str, int]]
) -> None:
    """Test that `EpisodicFlowFieldScheduler` rejects non-positive or
    non-integer batch sizes.

    Args:
        invalid_batch_size: Invalid batch size used to check validation.
        mock_mat_files: Temporary MATLAB file paths and dimensions.
    """
    files, dims = mock_mat_files
    H, W = dims["height"], dims["width"]

    base = MATFlowFieldScheduler(files, loop=False, output_shape=(H, W))
    with pytest.raises(ValueError):
        EpisodicFlowFieldScheduler(
            base, batch_size=invalid_batch_size, episode_length=2
        )


@pytest.mark.parametrize("invalid_episode_length", [-1, 0, 1.5, "two"])
@pytest.mark.parametrize("mock_mat_files", [64], indirect=True)
def test_invalid_episode_length(
    invalid_episode_length: Any,
    mock_mat_files: tuple[list[str], dict[str, int]],
) -> None:
    """Verify invalid episode length.

    Args:
        invalid_episode_length: Invalid episode length used by this
            test.
        mock_mat_files: Temporary MATLAB file paths and their
            dimensions.
    """
    files, dims = mock_mat_files
    H, W = dims["height"], dims["width"]

    base = MATFlowFieldScheduler(files, loop=False, output_shape=(H, W))
    with pytest.raises(ValueError):
        EpisodicFlowFieldScheduler(
            base, batch_size=2, episode_length=invalid_episode_length
        )


@pytest.mark.parametrize("invalid_batch_size", [-1, 0, 1.5, "two", 1, 3])
@pytest.mark.parametrize("mock_mat_files", [64], indirect=True)
def test_invalid_batch_size_in_get_batch(
    invalid_batch_size: Any, mock_mat_files: tuple[list[str], dict[str, int]]
) -> None:
    """Verify invalid batch size in get batch.

    Args:
        invalid_batch_size: Invalid batch size used by this test.
        mock_mat_files: Temporary MATLAB file paths and their
            dimensions.
    """
    files, dims = mock_mat_files
    H, W = dims["height"], dims["width"]
    base = MATFlowFieldScheduler(files, loop=False, output_shape=(H, W))

    epi = EpisodicFlowFieldScheduler(base, batch_size=2, episode_length=2)
    with pytest.raises(ValueError):
        epi.get_batch(invalid_batch_size)


@pytest.mark.parametrize("episode_length", [1, 2, 3])
@pytest.mark.parametrize("mock_mat_files", [64], indirect=True)
def test_steps_remaining(
    episode_length: int, mock_mat_files: tuple[list[str], dict[str, int]]
) -> None:
    """Test that `steps_remaining` correctly tracks the current position in
    the episode.

    Verifies that it starts at `episode_length` and decrements to zero
    as batches are retrieved.

    Args:
        episode_length: Episode length for this test case.
        mock_mat_files: Temporary MATLAB file paths and dimensions.
    """
    files, dims = mock_mat_files
    batch_size = 2
    H, W = dims["height"], dims["width"]
    base = MATFlowFieldScheduler(files, loop=False, output_shape=(H, W))
    epi = EpisodicFlowFieldScheduler(
        base, batch_size=batch_size, episode_length=episode_length
    )
    assert epi.steps_remaining() == episode_length, (
        f"Expected {episode_length} steps remaining, got "
        f"{epi.steps_remaining()}"
    )
    assert len(epi) == episode_length, (
        f"Expected length {episode_length}, got {len(epi)}"
    )

    for _ in range(episode_length):
        _ = epi.get_batch(batch_size=batch_size)
    assert epi.steps_remaining() == 0, (
        f"Expected 0 steps remaining after {episode_length} iterations, "
        f"got {epi.steps_remaining()}"
    )


@pytest.mark.parametrize("episode_length", [10])
@pytest.mark.parametrize("mock_mat_files", [8], indirect=True)
def test_not_enough_files_for_episode_length(
    episode_length: int, mock_mat_files: tuple[list[str], dict[str, int]]
) -> None:
    """Test that the scheduler raises an error if the data source contains
    fewer frames than requested.

    Ensures that an episode cannot be partially filled if the source is
    too short.

    Args:
        episode_length: Episode length for this test case.
        mock_mat_files: Temporary MATLAB file paths and dimensions.
    """
    files, dims = mock_mat_files
    H, W = dims["height"], dims["width"]
    base = MATFlowFieldScheduler(files, loop=False, output_shape=(H, W))

    with pytest.raises(ValueError):
        EpisodicFlowFieldScheduler(
            base, batch_size=1, episode_length=episode_length
        )
