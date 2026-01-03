"""Tests for histogram equalization and matching.

These tests verify that `match_histogram` correctly transforms image 
intensities to follow a target distribution, and that the integration 
of histogram equalization in the image generation pipeline is robust.
"""
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit

from synthpix.data_generate import (generate_images_from_flow,
                                    input_check_gen_img_from_flow)
from synthpix.sampler import SyntheticImageSampler
from synthpix.types import ImageGenerationSpecification
from synthpix.utils import load_configuration, match_histogram

TARGET_SHAPE = (64, 64)
sampler_config = load_configuration("config/test_data.yaml")


@pytest.fixture
def mock_histogram_file(tmp_path, numpy_test_dims):
    """Create and save a numpy valid histogram to a temporary file."""
    shape = numpy_test_dims["height"], numpy_test_dims["width"]

    path = tmp_path / "histogram.npy"
    arr = np.zeros((256,))
    arr[0] = (
        shape[0] * shape[1]
    )  # Set the first bin to the total number of pixels
    np.save(path, arr)

    yield str(path), numpy_test_dims


@pytest.fixture
def mock_histogram_invalid_file(tmp_path, numpy_test_dims):
    """Create and save a numpy invalid histogram to a temporary file."""
    shape = numpy_test_dims["height"], numpy_test_dims["width"]

    path = tmp_path / "invalid_histogram.npy"
    arr = np.zeros((256,))
    arr[0] = shape[0] * shape[1] - 1
    np.save(path, arr)

    yield str(path), numpy_test_dims


@pytest.fixture(scope="module")
def random_image_uint8():
    """Generates a random TARGET_SHAPE uint8 grayscale image for testing."""
    key = jax.random.PRNGKey(42)
    img = jax.random.randint(
        key, TARGET_SHAPE, minval=0, maxval=256, dtype=jnp.uint8
    )
    return img


def test_identity_mapping(random_image_uint8):
    """Matching a histogram to its own should return the original image."""
    src = random_image_uint8.astype(jnp.float32)
    # Compute source histogram in 256 bins (0..255)
    template_hist, _ = jnp.histogram(
        src, bins=jnp.arange(257, dtype=jnp.float32)
    )
    assert template_hist.shape[0] == 256, f"Expected 256 histogram bins, got {template_hist.shape[0]}"
    assert jnp.isclose(jnp.sum(template_hist), src.size), f"Histogram sum mismatch. Expected {src.size}, got {jnp.sum(template_hist)}"

    matched = match_histogram(src, template_hist)

    # Output dtype equals input dtype
    assert matched.dtype == src.dtype, f"Dtype mismatch. Expected {src.dtype}, got {matched.dtype}"
    # Should be identical
    assert jnp.allclose(matched, src), "Matched image should be identical to source for identity histogram"


def test_uniform_histogram_ramp():
    """A linear ramp covering 0..255 in a 16x16 image remains unchanged"""
    src = jnp.arange(256, dtype=jnp.float32).reshape((16, 16))
    template_hist = jnp.full(256, src.size / 256, dtype=jnp.float32)
    assert jnp.isclose(jnp.sum(template_hist), src.size), f"Histogram sum mismatch. Expected {src.size}, got {jnp.sum(template_hist)}"

    matched = match_histogram(src, template_hist)
    assert jnp.allclose(matched, src), "Linear ramp with uniform histogram should remain unchanged"


def test_constant_source():
    """A constant source image maps all pixels to the highest intensity."""
    src = jnp.full(TARGET_SHAPE, 128, dtype=jnp.uint8).astype(jnp.float32)
    # Create an increasing histogram and scale to sum to number of pixels
    raw = jnp.arange(1, 257, dtype=jnp.float32)
    template_hist = raw / jnp.sum(raw) * src.size
    assert template_hist.shape[0] == 256, f"Expected 256 histogram bins, got {template_hist.shape[0]}"
    assert jnp.isclose(jnp.sum(template_hist), src.size), f"Histogram sum mismatch. Expected {src.size}, got {jnp.sum(template_hist)}"

    matched = match_histogram(src, template_hist)
    expected = 255.0
    assert jnp.allclose(matched.astype(jnp.float32), expected), f"Constant source should map to highest intensity {expected}, but got values like {matched.flatten()[0]}"


def test_jit_compatibility(random_image_uint8):
    """Ensure the function can be JIT-compiled and yields identical results."""
    src = random_image_uint8.astype(jnp.float32)
    template_hist, _ = jnp.histogram(
        src, bins=jnp.arange(257, dtype=jnp.float32)
    )
    assert template_hist.shape[0] == 256, f"Expected 256 histogram bins, got {template_hist.shape[0]}"
    assert jnp.isclose(jnp.sum(template_hist), src.size), f"Histogram sum mismatch. Expected {src.size}, got {jnp.sum(template_hist)}"

    jit_fn = jit(match_histogram)
    out1 = match_histogram(src, template_hist)
    out2 = jit_fn(src, template_hist)
    assert jnp.allclose(out1, out2), "JIT and non-JIT match_histogram results mismatch"


def test_input_check_gen_img_from_flow_logs_histogram(monkeypatch):
    """Test that the input_check_gen_img_from_flow function logs the histogram."""

    flow_field = jnp.zeros((1, 8, 8, 2))
    image_shape = (4, 4)
    histogram = jnp.zeros((256,))
    histogram = histogram.at[0].set(16)

    import synthpix.data_generate as generate_mod

    # Collect debug messages
    logged = []
    monkeypatch.setattr(
        generate_mod.logger, "debug", lambda msg: logged.append(msg)
    )

    # Call the function to test
    generate_mod.input_check_gen_img_from_flow(
        flow_field=flow_field,
        parameters=ImageGenerationSpecification(image_shape=image_shape),
        histogram=histogram,
    )

    # Check if the mask shape was logged
    expected_msg = "Histogram equalization will be applied to the images."
    assert any(expected_msg in m for m in logged), (
        f"Expected :{expected_msg}, got: {logged}"
    )


@pytest.mark.parametrize(
    "histogram,error",
    [
        ("not an array", "histogram must be a jnp.ndarray or None."),
        ([1, 2, 3], "histogram must be a jnp.ndarray or None."),
        (jnp.zeros((256, 1)), "histogram must be a 1D jnp.ndarray."),
        (jnp.zeros((255,)), "histogram must have 256 bins (shape (256,))."),
        (
            jnp.zeros((256,)),
            "Histogram must sum to the number of pixels in the image shape.",
        ),
        (
            jnp.ones((256,)) * 10,
            "Histogram must sum to the number of pixels in the image shape.",
        ),
    ],
)
def test_invalid_histogram_in_generate(histogram, error):
    """Test that `input_check_gen_img_from_flow` rejects invalid histograms.

    Verifies the validation of histogram shapes, types, and bin sums 
    relative to the total pixel count.
    """
    flow_field = jnp.zeros((1, 64, 64, 2))
    image_shape = (64, 64)
    with pytest.raises(
        ValueError,
        match=re.escape(error),
    ):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            parameters=ImageGenerationSpecification(image_shape=image_shape),
            histogram=histogram,
        )


def test_histogram_applies():
    """Verify that `generate_images_from_flow` correctly applies histogram equalization.

    Generates images and checks that their resulting pixel distributions 
    exactly match the requested histogram.
    """
    flow_field = jnp.zeros((1, 32, 32, 2))
    image_shape = (32, 32)
    image_offset = (0, 0)
    histogram = jnp.zeros((256,))
    histogram = histogram.at[0].set(image_shape[0] * image_shape[1])

    images1, images2, flow = generate_images_from_flow(
        key=jax.random.PRNGKey(0),
        flow_field=flow_field,
        parameters=ImageGenerationSpecification(
            image_shape=image_shape,
            img_offset=image_offset,
            batch_size=1,
        ),
        position_bounds=image_shape,
        histogram=histogram,
    )

    # Check if the histogram is applied correctly
    hist1, _ = jnp.histogram(images1, bins=jnp.arange(257, dtype=jnp.float32))
    hist2, _ = jnp.histogram(images2, bins=jnp.arange(257, dtype=jnp.float32))
    assert jnp.allclose(hist1, histogram), f"Histogram mismatch for images1. Expected sum {jnp.sum(histogram)}, got {jnp.sum(hist1)}"
    assert jnp.allclose(hist2, histogram), f"Histogram mismatch for images2. Expected sum {jnp.sum(histogram)}, got {jnp.sum(hist2)}"


@pytest.mark.parametrize(
    "histogram,error",
    [
        (1, "histogram must be a string representing the histogram path."),
        ([], "histogram must be a string representing the histogram path."),
        ({}, "histogram must be a string representing the histogram path."),
        (
            jnp.ones((256, 256)),
            "histogram must be a string representing the histogram path.",
        ),
        (
            jnp.ones((255,)),
            "histogram must be a string representing the histogram path.",
        ),
        (
            "non_existent_histogram.npy",
            "Histogram file non_existent_histogram.npy does not exist.",
        ),
    ],
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_histogram_sampler(histogram, error, scheduler):
    """Test that the sampler raises a ValueError when provided an invalid histogram path.

    Checks for non-string paths and missing files.
    """
    with pytest.raises(ValueError, match=error):
        config = sampler_config.copy()
        config["histogram"] = histogram
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            config=config,
        )


@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_histogram_values(scheduler, mock_histogram_invalid_file):
    """Test that histograms with incorrect shapes or sums raise a ValueError.

    Ensures that even if a file exists, its content must strictly follow 
    the expected histogram format.
    """
    # Create a dummy histogram with an invalid shape
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Histogram must be a (256,) array and "
            "sum to the number of pixels in the image."
        ),
    ):
        config = sampler_config.copy()
        config["histogram"] = mock_histogram_invalid_file[0]
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            config=config,
        )


@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_histogram_is_correct(scheduler, mock_histogram_file):
    """Test that a valid histogram is correctly loaded from a file.

    Verifies that the `SyntheticImageSampler` internal state matches 
    the loaded numpy array.
    """
    # Create a dummy histogram with a valid shape
    histogram = jnp.array(np.load(mock_histogram_file[0]))
    image_shape = (
        mock_histogram_file[1]["height"],
        mock_histogram_file[1]["width"],
    )

    config = sampler_config.copy()
    config["histogram"] = mock_histogram_file[0]
    config["image_shape"] = image_shape
    sampler = SyntheticImageSampler.from_config(
        scheduler=scheduler,
        config=config,
    )

    assert isinstance(sampler.histogram, jnp.ndarray), f"Expected sampler.histogram to be jnp.ndarray, got {type(sampler.histogram)}"
    assert sampler.histogram.shape == histogram.shape, f"Histogram shape mismatch. Expected {histogram.shape}, got {sampler.histogram.shape}"
    assert jnp.array_equal(sampler.histogram, histogram), (
        "Histogram loaded from file does not match the expected histogram."
    )
