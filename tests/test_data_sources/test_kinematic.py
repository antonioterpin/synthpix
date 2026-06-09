"""Tests for KinematicDataSource.

KinematicDataSource generates smooth random displacement fields in-memory
(the kinematic-training RFG of Manickathan et al. 2022) instead of reading
them from disk. Generation must be deterministic in the record index so the
Grain pipeline can checkpoint and replay it, the peak displacement must
honour the configured range, and a larger Gaussian filter width must yield a
smoother field.
"""

import numpy as np
import pytest

from synthpix.data_sources import KinematicDataSource


def test_len_reports_num_examples():
    """The dataset length equals the configured number of examples."""
    ds = KinematicDataSource(num_examples=37, image_shape=(32, 32))
    assert len(ds) == 37, f"Expected 37 examples, got {len(ds)}"


def test_getitem_returns_flow_field_record():
    """An item holds a float32 (H, W, 2) flow field and a file id."""
    ds = KinematicDataSource(num_examples=10, image_shape=(48, 64))

    item = ds[0]

    assert sorted(item.keys()) == ["file", "flow_fields"], (
        f"Unexpected record keys: {sorted(item.keys())}"
    )
    assert item["flow_fields"].shape == (48, 64, 2), (
        f"Expected (48, 64, 2), got {item['flow_fields'].shape}"
    )
    assert item["flow_fields"].dtype == np.float32, (
        f"Expected float32, got {item['flow_fields'].dtype}"
    )
    assert item["file"] == "kinematic://0", (
        f"Unexpected file id: {item['file']}"
    )


def test_generation_is_deterministic_per_index():
    """The same index always yields the same field (replay safety)."""
    ds = KinematicDataSource(num_examples=10, image_shape=(32, 32), seed=7)

    first = ds[5]["flow_fields"]
    second = ds[5]["flow_fields"]

    assert np.array_equal(first, second), (
        "Repeated access to the same index produced different fields"
    )


def test_distinct_indices_differ():
    """Different indices yield different fields."""
    ds = KinematicDataSource(num_examples=10, image_shape=(32, 32))

    assert not np.array_equal(
        ds[0]["flow_fields"], ds[1]["flow_fields"]
    ), "Distinct indices produced identical fields"


def test_seed_changes_output():
    """Changing the base seed changes the generated field."""
    a = KinematicDataSource(num_examples=4, image_shape=(32, 32), seed=0)
    b = KinematicDataSource(num_examples=4, image_shape=(32, 32), seed=1)

    assert not np.array_equal(
        a[0]["flow_fields"], b[0]["flow_fields"]
    ), "Different seeds produced identical fields"


@pytest.mark.parametrize("index", range(8))
def test_peak_displacement_within_range(index):
    """Each field's peak magnitude lies within the requested range."""
    low, high = 3.0, 9.0
    ds = KinematicDataSource(
        num_examples=8,
        image_shape=(64, 64),
        max_displacement_range=(low, high),
    )

    flow = ds[index]["flow_fields"]
    peak = float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).max())

    assert low - 1e-3 <= peak <= high + 1e-3, (
        f"Peak displacement {peak} outside [{low}, {high}]"
    )


def test_zero_max_displacement_yields_zero_field():
    """A zero displacement range produces a zero field."""
    ds = KinematicDataSource(
        num_examples=4,
        image_shape=(32, 32),
        max_displacement_range=(0.0, 0.0),
    )

    assert np.all(ds[0]["flow_fields"] == 0.0), (
        "Zero max-displacement range did not yield a zero field"
    )


def test_scale_to_max_displacement_uniform_field_is_zero():
    """A uniform (zero-peak) field scales to zero without dividing by zero."""
    out = KinematicDataSource._scale_to_max_displacement(
        np.zeros((4, 4, 2)), 5.0
    )
    assert np.all(out == 0.0), "Uniform field should scale to a zero field"


def test_larger_sigma_is_smoother():
    """A larger Gaussian width yields a lower-gradient (smoother) field."""

    def mean_abs_gradient(sigma: float) -> float:
        ds = KinematicDataSource(
            num_examples=1,
            image_shape=(96, 96),
            filter_sigma_range=(sigma, sigma),
            max_displacement_range=(8.0, 8.0),
            seed=0,
        )
        flow = ds[0]["flow_fields"]
        grads = [np.gradient(flow[..., c]) for c in (0, 1)]
        return float(np.mean([np.abs(g).mean() for gc in grads for g in gc]))

    sharp = mean_abs_gradient(3.0)
    smooth = mean_abs_gradient(40.0)

    assert smooth < sharp, (
        f"Larger sigma not smoother: grad(40)={smooth} >= grad(3)={sharp}"
    )


def test_include_images_is_false():
    """The kinematic source never advertises real images."""
    ds = KinematicDataSource(num_examples=1)
    assert ds.include_images is False


def test_repr_is_stable_and_descriptive():
    """The repr is deterministic and records the generation parameters."""
    ds = KinematicDataSource(
        num_examples=5,
        image_shape=(32, 32),
        filter_sigma_range=(5.0, 10.0),
        seed=3,
    )

    text = repr(ds)

    assert text == repr(ds), "repr is not stable"
    assert "num_examples=5" in text, f"repr missing num_examples: {text}"
    assert "seed=3" in text, f"repr missing seed: {text}"


def test_from_config_reads_keys():
    """`from_config` maps dataset-config keys onto the generator."""
    config = {
        "num_examples": 12,
        "image_shape": [40, 50],
        "filter_sigma_range": [2.0, 6.0],
        "max_displacement_range": [1.0, 4.0],
        "seed": 9,
    }

    ds = KinematicDataSource.from_config(config)

    assert len(ds) == 12, f"Expected 12 examples, got {len(ds)}"
    assert ds[0]["flow_fields"].shape == (40, 50, 2), (
        f"Unexpected shape {ds[0]['flow_fields'].shape}"
    )


def test_from_config_uses_defaults():
    """`from_config` falls back to the paper defaults when keys are absent."""
    ds = KinematicDataSource.from_config({})

    assert len(ds) == 18278, f"Expected default 18278, got {len(ds)}"
    assert ds[0]["flow_fields"].shape == (256, 256, 2), (
        f"Unexpected default shape {ds[0]['flow_fields'].shape}"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_examples": 0},
        {"num_examples": -3},
        {"image_shape": (32,)},
        {"image_shape": (0, 32)},
        {"filter_sigma_range": (10.0, 1.0)},
        {"filter_sigma_range": (-1.0, 10.0)},
        {"max_displacement_range": (5.0, 1.0)},
    ],
)
def test_invalid_arguments_raise(kwargs):
    """Malformed generator settings raise ValueError."""
    with pytest.raises(ValueError):
        KinematicDataSource(**kwargs)
