import jax
import jax.numpy as jnp
import pytest
from jax import jit

from synthpix.utils import match_histogram


# Fixture to generate a random 64x64 grayscale uint8 image
@pytest.fixture(scope="module")
def random_image_uint8():
    key = jax.random.PRNGKey(42)
    # Random ints [0,256) in uint8
    img = jax.random.randint(key, (64, 64), minval=0, maxval=256, dtype=jnp.uint8)
    return img


def test_identity_mapping(random_image_uint8):
    """
    Matching a histogram to its own should return the original image.
    """
    src = random_image_uint8.astype(jnp.float32)
    # Compute source histogram for length-256 bins (0..255)
    template_hist, _ = jnp.histogram(src, bins=jnp.arange(257, dtype=jnp.float32))
    assert template_hist.shape[0] == 256

    matched = match_histogram(src, template_hist)

    assert matched.dtype == src.dtype
    assert jnp.allclose(matched, src)


def test_uniform_histogram():
    # Linear ramp 0..255 -> shape 16×16, uniform template -> identity mapping
    src = jnp.arange(256, dtype=jnp.float32).reshape(16, 16)
    template_hist = jnp.ones(256, dtype=jnp.float32)
    matched = match_histogram(src, template_hist)
    assert jnp.allclose(matched, src)


def test_constant_source():
    """A constant source image maps all pixels to the highest intensity (255)."""
    src = jnp.full((64, 64), 128, dtype=jnp.uint8).astype(jnp.float32)
    template_hist = jnp.arange(1, 257, dtype=jnp.float32)
    assert template_hist.shape[0] == 256

    matched = match_histogram(src, template_hist)
    expected = 255.0
    assert jnp.allclose(matched.astype(jnp.float32), expected)


def test_jit_compatibility(random_image_uint8):
    """Ensure the function can be JIT-compiled and matches for a random image."""
    src = random_image_uint8.astype(jnp.float32)
    template_hist, _ = jnp.histogram(src, bins=jnp.arange(257, dtype=jnp.float32))
    assert template_hist.shape[0] == 256

    jit_fn = jit(match_histogram)
    out1 = match_histogram(src, template_hist)
    out2 = jit_fn(src, template_hist)

    assert jnp.allclose(out1, out2)
