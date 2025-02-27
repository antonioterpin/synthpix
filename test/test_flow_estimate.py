"""Test the flow field computation using synthetic images."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.sym.apply import apply_flow_to_image
from src.sym.estimate import CrossCorrelationEstimator
from src.sym.example_flows import horizontal_flow, pipe_horizontal_flow, vortex_flow
from src.sym.generate import add_noise_to_image, generate_synthetic_particle_image
from src.utils import compute_image_scaled_height

FLOW_FIELD_TOLERANCE = 1e-2
WINDOW_SIZE = 16
VISUALIZE = False

CROSS_CORRELATION_CONFIG = {
    "WINDOW_SIZE": WINDOW_SIZE,
    "OVERLAP": WINDOW_SIZE - 1,
    "MAX_SPEED": 2.0,
    "RESOLUTION_LEVELS": 8,
    "BACKGROUND_SUPPRESSION": 0.0,
}


@pytest.fixture(scope="session")
def estimator():
    """Create a FlowFieldEstimator instance for testing."""
    return CrossCorrelationEstimator.from_config(CROSS_CORRELATION_CONFIG)


# Tests for CrossCorrelationEstimator
def test_crosscorrelationestimator_negative_window_size():
    """Test that negative window_size raises ValueError."""
    conf_window_size = CROSS_CORRELATION_CONFIG.copy()
    conf_window_size["WINDOW_SIZE"] = -1
    with pytest.raises(ValueError, match="Window size should be greater than 0."):
        CrossCorrelationEstimator.from_config(conf_window_size)


def test_crosscorrelationestimator_excessive_overlap():
    """Test that overlap >= window_size raises ValueError."""
    conf_overlap = CROSS_CORRELATION_CONFIG.copy()
    conf_overlap["OVERLAP"] = 32
    conf_overlap["WINDOW_SIZE"] = 32
    with pytest.raises(
        ValueError, match="Overlap should be less than the window size."
    ):
        CrossCorrelationEstimator.from_config(conf_overlap)


def test_crosscorrelationestimator_parent_negative_max_speed():
    """Test inherited max_speed validation."""
    conf_negative_speed = CROSS_CORRELATION_CONFIG.copy()
    conf_negative_speed["MAX_SPEED"] = -1.0
    with pytest.raises(ValueError, match="Maximum speed should be greater than 0."):
        CrossCorrelationEstimator.from_config(conf_negative_speed)


def test_crosscorrelationestimator_parent_zero_resolution_levels():
    """Test inherited resolution_levels validation."""
    conf_zero_resolution = CROSS_CORRELATION_CONFIG.copy()
    conf_zero_resolution["RESOLUTION_LEVELS"] = 0
    with pytest.raises(ValueError, match="Resolution levels should be greater than 0."):
        CrossCorrelationEstimator.from_config(conf_zero_resolution)


def test_crosscorrelationestimator_parent_excessive_resolution_levels():
    """Test inherited resolution_levels > 127 validation."""
    conf_excessive_resolution = CROSS_CORRELATION_CONFIG.copy()
    conf_excessive_resolution["RESOLUTION_LEVELS"] = 128
    with pytest.raises(ValueError, match="Resolution levels should be less than 16."):
        CrossCorrelationEstimator.from_config(conf_excessive_resolution)


def test_crosscorrelationestimator_parent_negative_background_suppression():
    """Test inherited background_suppression validation."""
    conf_negative_background_sup = CROSS_CORRELATION_CONFIG.copy()
    conf_negative_background_sup["BACKGROUND_SUPPRESSION"] = -1.0
    with pytest.raises(
        ValueError, match="Background suppression should be non-negative."
    ):
        CrossCorrelationEstimator.from_config(conf_negative_background_sup)


def test_quantize_basic_range(estimator):
    """Test quantization of flow field within expected range."""
    flow_field = jnp.array([[-1.0, 1.0], [-1.0, -1.0]])
    quantized = estimator.quantize(flow_field)

    # Expected step size: 2 * max_speed / resolution_levels = 4.0 / 8 = 0.5
    expected = jnp.array([[63, 191], [63, 63]], dtype=jnp.uint8)

    assert jnp.allclose(
        quantized, expected, atol=1
    ), "Quantization values are incorrect"
    assert quantized.dtype == jnp.uint8, "Output should be uint8"


def test_quantize_below_min(estimator):
    """Test quantization clips values below -max_speed."""
    flow_field = jnp.array([[-15.0, -10.0], [-20.0, -12.0]])
    quantized = estimator.quantize(flow_field)

    # All values below -2 should be clipped to 0
    expected = jnp.array([[0, 0], [0, 0]], dtype=jnp.uint8)

    assert jnp.all(quantized == expected), "Values below -max_speed should clip to 0"
    assert quantized.dtype == jnp.uint8, "Output should be uint8"


def test_quantize_above_max(estimator):
    """Test quantization clips values above max_speed."""
    flow_field = jnp.array([[1.0, 20.0], [1.0, 25.0]])
    quantized = estimator.quantize(flow_field)

    # All values above 10 should be clipped to 255
    expected = jnp.array([[191, 255], [191, 255]], dtype=jnp.uint8)

    assert jnp.all(quantized == expected), "Values above max_speed should clip to max"
    assert quantized.dtype == jnp.uint8, "Output should be uint8"


def test_quantize_zero_flow(estimator):
    """Test quantization of zero flow field."""
    flow_field = jnp.array([[0.0, 0.0], [0.0, 0.0]])
    quantized = estimator.quantize(flow_field)

    expected = 128 * jnp.ones(flow_field.shape, dtype=jnp.uint8)

    assert jnp.all(quantized == expected), "Zero flow should map to middle range"
    assert quantized.dtype == jnp.uint8, "Output should be uint8"


def test_quantize_different_shapes(estimator):
    """Test quantization preserves input shape."""
    # Test with 1D, 2D, and 3D arrays
    flow_field_1d = jnp.array([-5.0, 0.0, 5.0])
    flow_field_2d = jnp.array([[-5.0, 0.0], [5.0, 10.0]])
    flow_field_3d = jnp.array([[[0.0, 5.0], [-5.0, 10.0]]])

    quantized_1d = estimator.quantize(flow_field_1d)
    quantized_2d = estimator.quantize(flow_field_2d)
    quantized_3d = estimator.quantize(flow_field_3d)

    assert quantized_1d.shape == flow_field_1d.shape, "1D shape should be preserved"
    assert quantized_2d.shape == flow_field_2d.shape, "2D shape should be preserved"
    assert quantized_3d.shape == flow_field_3d.shape, "3D shape should be preserved"
    assert quantized_1d.dtype == jnp.uint8, "1D output should be uint8"
    assert quantized_2d.dtype == jnp.uint8, "2D output should be uint8"
    assert quantized_3d.dtype == jnp.uint8, "3D output should be uint8"


def test_quantize_resolution_levels_effect():
    """Test different resolution levels affect quantization steps."""
    conf_quantize_res_effect1 = CROSS_CORRELATION_CONFIG.copy()
    conf_quantize_res_effect1["MAX_SPEED"] = 10.0
    conf_quantize_res_effect1["RESOLUTION_LEVELS"] = 4
    conf_quantize_res_effect2 = CROSS_CORRELATION_CONFIG.copy()
    conf_quantize_res_effect2["MAX_SPEED"] = 10.0
    conf_quantize_res_effect2["RESOLUTION_LEVELS"] = 16
    estimator_low = CrossCorrelationEstimator.from_config(conf_quantize_res_effect1)
    estimator_high = CrossCorrelationEstimator.from_config(conf_quantize_res_effect2)

    flow_field = jnp.array([5.0])
    quantized_low = estimator_low.quantize(flow_field)
    quantized_high = estimator_high.quantize(flow_field)

    # Step size for low: 20 / 4 = 5 -> (5 + 10) / 5 = 3 -> 3 * 63.75 ≈ 191
    # Step size for high: 20 / 16 = 1.25 -> (5 + 10) / 1.25 = 12 -> 12 * 15.9375 ≈ 191
    assert quantized_low[0] == 191, "Low resolution quantization incorrect"
    assert quantized_high[0] == 191, "High resolution quantization incorrect"
    assert quantized_low.dtype == jnp.uint8, "Low resolution output should be uint8"
    assert quantized_high.dtype == jnp.uint8, "High resolution output should be uint8"


def test_quantize_max_speed_effect():
    """Test different max_speed values affect quantization range."""
    conf_quantize_max_speed_effect1 = CROSS_CORRELATION_CONFIG.copy()
    conf_quantize_max_speed_effect1["MAX_SPEED"] = 5.0
    conf_quantize_max_speed_effect1["RESOLUTION_LEVELS"] = 8
    conf_quantize_max_speed_effect2 = CROSS_CORRELATION_CONFIG.copy()
    conf_quantize_max_speed_effect2["MAX_SPEED"] = 20.0
    conf_quantize_max_speed_effect2["RESOLUTION_LEVELS"] = 8
    estimator_low = CrossCorrelationEstimator.from_config(
        conf_quantize_max_speed_effect1
    )
    estimator_high = CrossCorrelationEstimator.from_config(
        conf_quantize_max_speed_effect2
    )

    flow_field = jnp.array([5.0])
    quantized_low = estimator_low.quantize(flow_field)
    quantized_high = estimator_high.quantize(flow_field)

    assert quantized_low[0] == 255, "Low max_speed quantization incorrect"
    assert quantized_high[0] == 159, "High max_speed quantization incorrect"
    assert quantized_low.dtype == jnp.uint8, "Output should be uint8"
    assert quantized_high.dtype == jnp.uint8, "Output should be uint8"


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("image_shape", [(128, 64), (64, 64)])
@pytest.mark.parametrize("flow_width_ratio", [0.5, 1.0])
@pytest.mark.parametrize("density", [0.1, 0.2])
@pytest.mark.parametrize("background_level", [0.0, 5.0])
@pytest.mark.parametrize(
    "flow_func", [horizontal_flow, pipe_horizontal_flow, vortex_flow]
)
@pytest.mark.parametrize("flow_speed", [0.75, 1.0, 1.25])
def test_flow_estimation(
    estimator,
    seed,
    image_shape,
    flow_width_ratio,
    density,
    background_level,
    flow_func,
    flow_speed,
):
    """Test that we can recover a known flow field from a synthetic particle image.

    Args:
        estimator (CrossCorrelationEstimator): Flow field estimator instance.
        seed (int): Random seed for reproducibility.
        image_shape (Tuple[int, int]): Shape of the synthetic image.
        flow_width_ratio (float): Ratio of the flow field width to the image width.
        density (float): Density of particles in the image.
        background_level (float): Level of background noise.
        flow_func
            (Callable[[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray]]):
            Function that computes the flow field at a given time.
        flow_speed (float): Speed of the flow.
    """
    # 1. Generate a synthetic particle image
    flow_func = horizontal_flow
    key = jax.random.PRNGKey(seed)
    img1 = generate_synthetic_particle_image(
        key,
        image_shape=image_shape,
        seeding_density=density,
        diameter_range=(0.1, 1.0),
        intensity_range=(50, 200),
    )
    # resize the image to match the flow field
    H, W = image_shape
    if flow_width_ratio < 1.0 - 1e-6:
        W = int(W * flow_width_ratio)
        H = compute_image_scaled_height(W, image_shape[1], image_shape[0])
        img1 = jax.image.resize(img1, (H, W), method=jax.image.ResizeMethod.CUBIC)
        assert img1.shape == (H, W), f"Image shape is {img1.shape}, expected {(H, W)}"

    # 2. Apply the flow field to get a second image
    def flow_f(t, x, y):
        u, v = flow_func(t, x, y)
        return u * flow_speed, v * flow_speed

    img2 = apply_flow_to_image(img1, flow_f, t=0.0)

    # 3. Apply independent background noise to both images
    img1 = add_noise_to_image(key, img1, background_level=background_level)
    img2 = add_noise_to_image(key, img2, background_level=background_level)

    # 3. Compute the flow field
    flow_field = estimator.compute_flow(jnp.stack([img1, img2], axis=-1))

    # 4. Compute expected flow field
    xs = jnp.linspace(0, 1, W)
    ys = jnp.linspace(0, 1, H)
    X, Y = jnp.meshgrid(xs, ys)

    u_expected, v_expected = jax.vmap(flow_func, in_axes=(0, 0, None))(
        X.flatten(), Y.flatten(), 0.0
    )

    # Quantize
    uv_expected = estimator.quantize(jnp.stack([u_expected, v_expected], axis=-1))

    u_expected = uv_expected[..., 0].reshape(H, W)
    v_expected = uv_expected[..., 1].reshape(H, W)
    x_np, y_np = np.array(X), np.array(Y)

    if VISUALIZE:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        plt.imshow(img1, cmap="gray")
        plt.title("Image 1")
        plt.subplot(2, 2, 2)
        plt.imshow(img2, cmap="gray")
        plt.title("Image 2")
        plt.subplot(2, 2, 3)
        plt.quiver(
            x_np,
            y_np,
            np.array(u_expected),
            np.array(v_expected),
            pivot="mid",
            color="r",
            scale=1,
            scale_units="xy",
        )
        plt.xlim(xs.min(), xs.max())
        plt.ylim(ys.min(), ys.max())
        plt.title("Expected Flow Field")
        plt.subplot(2, 2, 4)
        plt.quiver(
            x_np,
            y_np,
            np.array(flow_field[:, :, 0]),
            np.array(flow_field[:, :, 1]),
            pivot="mid",
            color="b",
            scale=1,
            scale_units="xy",
        )
        plt.xlim(xs.min(), xs.max())
        plt.ylim(ys.min(), ys.max())
        plt.title("Estimated Flow Field")
        plt.show()

    # 5. Compute average errors
    # Remove boundaries from assessment
    IGNORE_BOUNDARY = WINDOW_SIZE // 2
    u_gt = u_expected[
        IGNORE_BOUNDARY:-IGNORE_BOUNDARY, IGNORE_BOUNDARY:-IGNORE_BOUNDARY
    ]
    v_gt = v_expected[
        IGNORE_BOUNDARY:-IGNORE_BOUNDARY, IGNORE_BOUNDARY:-IGNORE_BOUNDARY
    ]

    u_est = flow_field[..., 0]
    v_est = flow_field[..., 1]

    # Compute average errors
    assert (
        u_est.shape == u_gt.shape
    ), f"U-component of the flow field shape is {u_est.shape}, expected {u_gt.shape}"
    assert (
        v_est.shape == v_gt.shape
    ), f"V-component of the flow field shape is {v_est.shape}, expected {v_gt.shape}"
    u_error = jnp.mean(jnp.abs(u_gt - u_est))
    u_std = jnp.std(jnp.abs(u_gt - u_est))
    v_error = jnp.mean(jnp.abs(v_gt - v_est))
    v_std = jnp.std(jnp.abs(v_gt - v_est))

    assert u_error < FLOW_FIELD_TOLERANCE, f"U-component error is {u_error} +/- {u_std}"
    assert v_error < FLOW_FIELD_TOLERANCE, f"V-component error is {v_error} +/- {v_std}"


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    from src.analyses.image_data.dataloader import ImageDataset
    from src.utils import load_configuration, logger

    # Load PIV configuration from config
    config = load_configuration("config/flow_estimator/cross-correlation.yaml")
    logger.info("Using configuration:", config)

    # Set the directory containing the images and the sequence length.
    image_directory = "src/analyses/image_data/images"
    seq_length = 2

    # Create the dataset and dataloader
    dataset = ImageDataset(image_directory, seq_length)
    # Use a batch_size of 1 since each batch is a sequence of images.
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Instantiate cross correlation estimator
    flow_computer = CrossCorrelationEstimator.from_config(config)

    # Iterate over the dataset sequences.
    try:
        # Display images and flow field
        plt.ion()
        fig, ax = plt.subplots(1, seq_length + 1, figsize=(18, 6))
        for a in ax:
            a.set_aspect("equal")

        q = None  # Initialize variable to store quiver artist
        image_initialized = False
        for seq_idx, batch in enumerate(dataloader):
            sequence = batch[0].numpy()
            flow_field = flow_computer.compute_flow(sequence.transpose(1, 2, 0))

            u_np = np.array(flow_field[..., 0])
            v_np = np.array(flow_field[..., 1])

            x_np, y_np = np.meshgrid(np.arange(u_np.shape[1]), np.arange(u_np.shape[0]))

            for i in range(seq_length):
                if not image_initialized:
                    ax[i] = ax[i].imshow(sequence[i], cmap="gray")
                else:
                    ax[i].set_data(sequence[i])
            image_initialized = True

            # Remove the previous quiver if it exists.
            if q is not None:
                q.remove()

            # Plot the new quiver and store its reference.
            q = ax[-1].quiver(x_np, y_np, u_np, v_np, pivot="mid", color="r")

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

    except KeyboardInterrupt:
        pass
