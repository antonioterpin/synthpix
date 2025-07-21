import jax
import jax.numpy as jnp
import pytest

from synthpix.sampler import SyntheticImageSampler


class _DummyScheduler:
    def __init__(self, h=64, w=64):
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


def _dummy_img_gen_fn(*, key, flow_field, num_images, image_shape, **_):
    h, w = image_shape
    imgs1 = jnp.zeros((num_images, h, w), dtype=jnp.float32)
    imgs2 = jnp.zeros_like(imgs1)
    return {
        "images1": imgs1,
        "images2": imgs2,
        "params": {
            "seeding_densities": jnp.zeros((num_images,)),
            "diameter_ranges": jnp.zeros((num_images, 2)),
            "intensity_ranges": jnp.zeros((num_images, 2)),
            "rho_ranges": jnp.zeros((num_images, 2)),
        },
    }


def _make_sampler(device_ids):
    return SyntheticImageSampler(
        scheduler=_DummyScheduler(),
        img_gen_fn=_dummy_img_gen_fn,
        batches_per_flow_batch=1,
        batch_size=4,
        flow_fields_per_batch=2,
        flow_field_size=(64, 64),
        image_shape=(32, 32),
        resolution=1.0,
        velocities_per_pixel=1.0,
        img_offset=(0.1, 0.1),
        seeding_density_range=(0.01, 0.01),
        p_hide_img1=0.0,
        p_hide_img2=0.0,
        diameter_ranges=[[1.0, 1.0]],
        diameter_var=0.0,
        intensity_ranges=[[1.0, 1.0]],
        intensity_var=0.0,
        rho_ranges=[[0.0, 0.0]],
        rho_var=0.0,
        dt=0.1,
        seed=0,
        max_speed_x=0.0,
        max_speed_y=0.0,
        min_speed_x=0.0,
        min_speed_y=0.0,
        output_units="pixels",
        noise_level=0.0,
        device_ids=device_ids,
    )


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------
@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
def test_sampler_uses_all_devices_when_none_passed():
    """If `device_ids=None`, the sampler should use every available device."""
    sampler = _make_sampler(device_ids=None)

    # jax.devices() returns a list; sampler.mesh.devices is a tuple
    assert all(tuple(jax.devices()) == sampler.mesh.devices)
    assert len(sampler.mesh.devices) >= 1


@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize("ids", [[0], [0, 1]])
def test_sampler_uses_requested_subset(ids):
    """Sampler should pick exactly the devices specified by `device_ids`."""
    if max(ids) >= len(jax.devices()):
        pytest.skip("Not enough physical devices for this parametrisation.")

    sampler = _make_sampler(device_ids=ids)

    picked = sorted(d.id for d in sampler.mesh.devices)
    assert picked == sorted(ids), f"Expected devices {ids}, got {picked}"


@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
def test_sampler_rejects_invalid_device_ids():
    """Passing only out-of-range IDs must raise a ValueError."""
    invalid_id = len(jax.devices())  # one past the last valid index
    with pytest.raises(ValueError, match="No valid device IDs provided."):
        _make_sampler(device_ids=[invalid_id])
