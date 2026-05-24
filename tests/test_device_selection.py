"""Tests for GPU device selection and mesh configuration in `SyntheticImageSampler`.

These tests verify that the sampler correctly identifies available NVIDIA GPUs
and allows users to specify exactly which devices should be used for
JAX-accelerated image generation via the `device_ids` parameter.
"""

import jax
import pytest

from synthpix.sampler import SyntheticImageSampler
from synthpix.scheduler import BaseFlowFieldScheduler
from synthpix.types import ImageGenerationSpecification


class _DummyScheduler(BaseFlowFieldScheduler):
    def __init__(self, temp_file:str, h:int=64, w:int=64):
        """A minimal scheduler for testing device selection logic.

        Args:
            temp_file: Path to a temporary file to use for testing.
            h: Height of the flow field.
            w: Width of the flow field.
        """
        super().__init__(file_list=[temp_file])
        self._shape = (h, w, 2)

    def get_flow_fields_shape(self):
        return self._shape

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def reset(self):
        pass

    def get_batch(self, *_):
        raise StopIteration

    def load_file(self, file_path: str):
        pass

    def get_next_slice(self):
        pass

    @classmethod
    def from_config(cls, config: dict):
        return cls(config["temp_file"])


def _make_sampler(device_ids, temp_file):
    """Helper to create a `SyntheticImageSampler` with specific device IDs.

    Uses a dummy scheduler and minimal configuration to isolate the device
    selection logic.

    Args:
        device_ids:
            A list of device IDs to use, or None to use all available devices
        temp_file: A temporary file path to pass to the dummy scheduler.
    """
    return SyntheticImageSampler(
        scheduler=_DummyScheduler(temp_file),
        batches_per_flow_batch=1,
        flow_fields_per_batch=2,
        flow_field_size=(64, 64),
        resolution=1.0,
        velocities_per_pixel=1.0,
        seed=0,
        max_speed_x=0.0,
        max_speed_y=0.0,
        min_speed_x=0.0,
        min_speed_y=0.0,
        output_units="pixels",
        device_ids=device_ids,
        generation_specification=ImageGenerationSpecification(
            batch_size=4,
            image_shape=(32, 32),
            img_offset=(0, 0),
            seeding_density_range=(0.01, 0.01),
            p_hide_img1=0.0,
            p_hide_img2=0.0,
            diameter_ranges=[(1.0, 1.0)],
            diameter_var=0.0,
            intensity_ranges=[(1.0, 1.0)],
            intensity_var=0.0,
            rho_ranges=[(0.0, 0.0)],
            rho_var=0.0,
            dt=0.1,
            noise_uniform=0.0,
            noise_gaussian_mean=0.0,
            noise_gaussian_std=0.0,
        ),
    )


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------
@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="User not connected to the server.",
)
def test_sampler_uses_all_devices_when_none_passed(temp_file: str):
    """Test that the sampler defaults to using all available JAX devices.

    If `device_ids=None` is passed to the constructor, the internal
    sharding mesh should encompass all physical GPUs detected by JAX.

    Args:
        temp_file: A temporary file path provided by the test fixture.
    """
    sampler = _make_sampler(device_ids=None, temp_file=temp_file)

    # jax.devices() returns a list; sampler.mesh.devices is a tuple
    # Compare device IDs rather than device objects to avoid JAX array
    # comparison issues
    expected_device_ids = [d.id for d in jax.devices()]
    actual_device_ids = [d.id for d in sampler.mesh.devices]
    assert expected_device_ids == actual_device_ids, (
        f"Default device IDs mismatch. Expected {expected_device_ids}, got {actual_device_ids}"
    )
    assert len(sampler.mesh.devices) >= 1, (
        "Sampler mesh should contain at least one device"
    )


@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="User not connected to the server.",
)
@pytest.mark.parametrize("ids", [[0], [0, 1]])
def test_sampler_uses_requested_subset(ids: list[int], temp_file: str):
    """Test that the sampler respects a specific subset of device IDs.

    Verifies that only the requested indices are included in the
    sampler's sharding mesh, ignoring other available devices.

    Args:
        ids: A list of device IDs to test (e.g., [0], [0, 1]).
        temp_file: A temporary file path provided by the test fixture.
    """
    if not all(id in [d.id for d in jax.devices()] for id in ids):
        pytest.skip("Devices not available.")

    sampler = _make_sampler(device_ids=ids, temp_file=temp_file)

    picked = sorted(d.id for d in sampler.mesh.devices)
    assert picked == sorted(ids), f"Expected devices {ids}, got {picked}"


@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="User not connected to the server.",
)
def test_sampler_rejects_invalid_device_ids(temp_file: str):
    """Test that specifying only non-existent device IDs raises a ValueError.

    Ensures that the sampler fails early if it cannot map any of the
    provided `device_ids` to actual physical hardware.

    Args:
        temp_file: A temporary file path provided by the test fixture.
    """
    # one past the last valid index
    invalid_id = max(d.id for d in jax.devices()) + 1
    with pytest.raises(ValueError, match="No valid device IDs provided."):
        _make_sampler(device_ids=[invalid_id], temp_file=temp_file)
