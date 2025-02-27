import jax
import pytest

from src.sym.apply import apply_flow_to_image
from src.sym.generate import generate_synthetic_particle_image


@pytest.mark.parametrize(
    "image_shape", [(16, 16), (64, 32), (32, 64), (256, 128), (128, 256), (256, 256)]
)
def test_flow_apply(image_shape, visualize=False):
    """Test that we can apply a flow field to a synthetic image."""
    # 1. Generate a synthetic particle image
    key = jax.random.PRNGKey(0)
    img = generate_synthetic_particle_image(
        key,
        image_shape=image_shape,
        seeding_density=0.1,
        diameter_range=(0.1, 1.0),
        intensity_range=(50, 200),
    )

    # 2. Apply a simple horizontal flow
    def flow_f(t, x, y):
        return 1.0, 0.0

    img_warped = apply_flow_to_image(img, flow_f, t=0.0)

    if visualize:
        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(np.array(img), cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title("Warped Image")
        plt.imshow(np.array(img_warped), cmap="gray")
        plt.show()

    # 3. Check image shapes
    assert img.shape == img_warped.shape, "Image shapes do not match"


if __name__ == "__main__":
    test_flow_apply(image_shape=(16, 16), visualize=True)
