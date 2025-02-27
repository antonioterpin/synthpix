import jax
import pytest

from src.sym.generate import add_noise_to_image, generate_synthetic_particle_image


@pytest.mark.parametrize(
    "image_shape", [(-1, 128), (128, -1), (0, 128), (128, 0), (128.2, 128.2)]
)
def test_invalid_image_shape(image_shape):
    """Test that invalid image shapes raise a ValueError."""
    key = jax.random.PRNGKey(0)
    with pytest.raises(
        ValueError, match="image_shape must be a tuple of two positive integers."
    ):
        generate_synthetic_particle_image(key, image_shape=image_shape)


@pytest.mark.parametrize("seeding_density", [-0.1, 0, 1.1])
def test_invalid_seeding_density(seeding_density):
    """Test that invalid seeding densities raise a ValueError."""
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError, match="seeding_density must be positive."):
        generate_synthetic_particle_image(key, seeding_density=seeding_density)


@pytest.mark.parametrize("diameter_range", [(0, 1), (1, 0), (-1, 1), (1, -1)])
def test_invalid_diameter_range(diameter_range):
    """Test that invalid diameter ranges raise a ValueError."""
    key = jax.random.PRNGKey(0)
    with pytest.raises(
        ValueError, match="diameter_range must be a tuple of two positive floats."
    ):
        generate_synthetic_particle_image(key, diameter_range=diameter_range)


@pytest.mark.parametrize("intensity_range", [(-1, 200), (50, 300), (300, 50), (50, -1)])
def test_invalid_intensity_range(intensity_range):
    """Test that invalid intensity ranges raise a ValueError."""
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError):
        generate_synthetic_particle_image(key, intensity_range=intensity_range)


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("image_shape", [(128, 128)])
@pytest.mark.parametrize("density", [0.06, 0.99])
@pytest.mark.parametrize("background_level", [0.0, 5.0, 255.0])
def test_generate_image(seed, image_shape, density, background_level, visualize=False):
    """Test that we can generate a synthetic particle image."""
    key = jax.random.PRNGKey(seed)
    img = generate_synthetic_particle_image(
        key,
        image_shape=image_shape,
        seeding_density=density,
        diameter_range=(0.1, 1.0),
        intensity_range=(50, 200),
    )

    img_background = add_noise_to_image(key, img, background_level=background_level)

    if visualize:
        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(np.array(img), cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title("Image with Background Noise")
        plt.imshow(np.array(img_background), cmap="gray")
        plt.show()

    assert img.shape == image_shape, "Image shape is incorrect"
    assert img.min() >= 0, "Image contains negative values"
    assert img.max() <= 255, "Image contains values above 255"

    assert img_background.shape == image_shape, "Image shape is incorrect"
    assert img_background.min() >= 0, "Image contains negative values"
    assert img_background.max() <= 255, "Image contains values above 255"


if __name__ == "__main__":
    test_generate_image(
        seed=0, image_shape=(16, 16), density=0.1, background_level=5.0, visualize=True
    )
