"""Tests for KinematicDataSource.

KinematicDataSource generates smooth random displacement fields in-memory
(the kinematic-training RFG of Manickathan et al. 2022) according to the
equation ds_ref = a * G_sigma * xi, where xi is independent U(-1, 1) noise,
G_sigma is a Gaussian filter, and a is the per-field scale. ``scale_mode``
selects how a is applied: ``"peak"`` (default) normalizes each field so its
peak displacement magnitude equals a px (paper-faithful, Manickathan et al.
2023 Table 1; sigma-independent), while ``"linear"`` multiplies the filtered
noise by a directly (sigma-dependent). Generation must be deterministic in the
record index so the Grain pipeline can checkpoint and replay it, and a larger
Gaussian filter width must yield a smoother field.
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


def test_zero_scale_factor_yields_zero_field():
    """A zero scale factor range produces a zero field."""
    ds = KinematicDataSource(
        num_examples=4,
        image_shape=(32, 32),
        scale_factor_range=(0.0, 0.0),
    )

    assert np.all(ds[0]["flow_fields"] == 0.0), (
        "Zero scale_factor_range did not yield a zero field"
    )


def test_scale_factor_produces_proportional_scaling():
    """Larger scale factors produce proportionally larger fields with same seed/sigma/index."""
    ds_small = KinematicDataSource(
        num_examples=1,
        image_shape=(32, 32),
        filter_sigma_range=(5.0, 5.0),
        scale_factor_range=(2.0, 2.0),
        seed=42,
    )
    ds_large = KinematicDataSource(
        num_examples=1,
        image_shape=(32, 32),
        filter_sigma_range=(5.0, 5.0),
        scale_factor_range=(4.0, 4.0),
        seed=42,
    )

    flow_small = ds_small[0]["flow_fields"]
    flow_large = ds_large[0]["flow_fields"]

    # flow_large should be approximately 2x flow_small (scale factor ratio is 4/2=2)
    assert np.allclose(flow_large, 2.0 * flow_small, rtol=1e-5, atol=1e-7)


def test_larger_sigma_is_smoother():
    """A larger Gaussian width yields a lower-gradient (smoother) field."""

    def mean_abs_gradient(sigma: float) -> float:
        ds = KinematicDataSource(
            num_examples=1,
            image_shape=(96, 96),
            filter_sigma_range=(sigma, sigma),
            scale_factor_range=(8.0, 8.0),
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


def _peak_magnitude(flow: np.ndarray) -> float:
    """Return the largest displacement magnitude in a flow field.

    Args:
        flow: Flow field of shape ``(H, W, 2)``.

    Returns:
        The maximum of ``sqrt(u**2 + v**2)`` over the field.
    """
    return float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).max())


def test_peak_mode_is_the_default():
    """Without an explicit mode, fields are peak-normalized in pixels."""
    a = 8.0
    ds = KinematicDataSource(
        num_examples=1,
        image_shape=(64, 64),
        filter_sigma_range=(40.0, 40.0),
        scale_factor_range=(a, a),
        seed=1,
    )

    assert _peak_magnitude(ds[0]["flow_fields"]) == pytest.approx(a, rel=1e-4)


@pytest.mark.parametrize("sigma", [5.0, 30.0, 100.0])
def test_peak_mode_hits_target_regardless_of_sigma(sigma: float):
    """Peak mode sets max|ds_ref| == a independent of the filter width.

    Args:
        sigma: Gaussian filter width (px) to test the normalization against.
    """
    a = 6.5
    ds = KinematicDataSource(
        num_examples=1,
        image_shape=(96, 96),
        filter_sigma_range=(sigma, sigma),
        scale_factor_range=(a, a),
        scale_mode="peak",
        seed=2,
    )

    assert _peak_magnitude(ds[0]["flow_fields"]) == pytest.approx(a, rel=1e-4)


def test_linear_mode_amplitude_collapses_with_sigma():
    """Linear mode applies a raw multiplier, so larger sigma shrinks motion.

    This is the sigma-dependent attenuation that motivates the peak-normalizing
    default: with the bare scale range a wide filter drives the displacement
    far below the nominal ``a``.
    """
    a = 8.0
    sharp = KinematicDataSource(
        num_examples=1,
        image_shape=(96, 96),
        filter_sigma_range=(5.0, 5.0),
        scale_factor_range=(a, a),
        scale_mode="linear",
        seed=2,
    )
    wide = KinematicDataSource(
        num_examples=1,
        image_shape=(96, 96),
        filter_sigma_range=(100.0, 100.0),
        scale_factor_range=(a, a),
        scale_mode="linear",
        seed=2,
    )

    sharp_peak = _peak_magnitude(sharp[0]["flow_fields"])
    wide_peak = _peak_magnitude(wide[0]["flow_fields"])

    assert wide_peak < sharp_peak < a, (
        f"Linear amplitude not sigma-attenuated: a={a}, "
        f"sharp_peak={sharp_peak}, wide_peak={wide_peak}"
    )


def test_peak_and_linear_modes_differ():
    """The two modes yield different fields for an identical random draw."""
    peak = KinematicDataSource(
        num_examples=1,
        image_shape=(48, 48),
        filter_sigma_range=(20.0, 20.0),
        scale_factor_range=(8.0, 8.0),
        scale_mode="peak",
        seed=3,
    )
    linear = KinematicDataSource(
        num_examples=1,
        image_shape=(48, 48),
        filter_sigma_range=(20.0, 20.0),
        scale_factor_range=(8.0, 8.0),
        scale_mode="linear",
        seed=3,
    )

    assert not np.allclose(
        peak[0]["flow_fields"], linear[0]["flow_fields"]
    ), "peak and linear modes produced identical fields"


def test_from_config_reads_scale_mode():
    """`from_config` honors an explicit scale_mode and defaults to peak."""
    linear = KinematicDataSource.from_config(
        {"image_shape": [48, 48], "scale_mode": "linear"}
    )
    assert "scale_mode='linear'" in repr(linear)

    default = KinematicDataSource.from_config({"image_shape": [48, 48]})
    assert "scale_mode='peak'" in repr(default)


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
        scale_factor_range=(2.0, 8.0),
        seed=3,
    )

    text = repr(ds)

    assert text == repr(ds), "repr is not stable"
    assert "num_examples=5" in text, f"repr missing num_examples: {text}"
    assert "seed=3" in text, f"repr missing seed: {text}"
    assert "scale_factor_range=(2.0, 8.0)" in text, (
        f"repr missing scale_factor_range: {text}"
    )
    assert "scale_mode='peak'" in text, f"repr missing scale_mode: {text}"


def test_from_config_reads_keys():
    """`from_config` maps dataset-config keys onto the generator."""
    config = {
        "num_examples": 12,
        "image_shape": [40, 50],
        "filter_sigma_range": [2.0, 6.0],
        "scale_factor_range": [1.0, 4.0],
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
        {"scale_factor_range": (5.0, 1.0)},
        {"scale_factor_range": (-1.0, 10.0)},
        {"scale_mode": "bogus"},
        {"scale_mode": "Peak"},
    ],
)
def test_invalid_arguments_raise(kwargs: dict) -> None:
    """Malformed generator settings raise ValueError.

    Args:
        kwargs: A dictionary of invalid arguments to pass to KinematicDataSource.
    """
    with pytest.raises(ValueError):
        KinematicDataSource(**kwargs)
