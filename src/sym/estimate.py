#!/usr/bin/env python
"""Module to estimate the flow field from a sequence of images."""

import abc

import jax
import jax.numpy as jnp

from src.utils import logger

class FlowFieldEstimator(abc.ABC):
    """Base class for flow field estimators."""

    def __init__(
        self,
        max_speed: float = 10.0,
        resolution_levels: int = 8,
        background_suppression: float = 0.0,
    ):
        """Initialize the flow field estimator with a maximum speed.

        Args:
            max_speed (float): Maximum speed of the flow field.
            resolution_levels (int): Number of resolution levels for the speeds.
            background_suppression (float): Background suppression level.
        """
        if max_speed < 0:
            raise ValueError("Maximum speed should be greater than 0.")
        self.max_speed = max_speed

        if resolution_levels < 1:
            raise ValueError("Resolution levels should be greater than 0.")
        if resolution_levels > 127:
            logger.warning("This flow estimator returns 8 bits flows.")
            logger.warning("Resolution cropped to 127.")
            resolution_levels = 127
            raise ValueError("Resolution levels should be less than 16.")
        self.resolution_levels = resolution_levels

        if background_suppression < 0:
            raise ValueError("Background suppression should be non-negative.")
        self.background_suppression = background_suppression

    def compute_flow(self, images: jnp.ndarray) -> jnp.ndarray:
        """Compute the flow field from a sequence of images.

        Args:
            images (np.ndarray): Input images of shape (H, W, N).

        Returns:
            jnp.ndarray: The flow field.
        """
        # Check if the input images are valid
        if images.ndim != 3:
            raise ValueError("Input images should have shape (H, W, N).")

        # Suppress background
        images = images * (images > self.background_suppression)

        # Compute the flow field
        flow_field = self._compute_flow(images)

        # Quantize the flow field
        return self.quantize(flow_field)

    @abc.abstractmethod
    def _compute_flow(self, images: jnp.ndarray) -> jnp.ndarray:
        """Compute the flow field from a sequence of images.

        Args:
            images (np.ndarray): Input images of shape (H, W, N).

        Returns:
            jnp.ndarray: The flow field.
        """
        pass

    def quantize(self, flow_field: jnp.ndarray) -> jnp.ndarray:
        """Quantize the flow field to discrete levels and map to 0-255.

        Args:
            flow_field (jnp.ndarray): The flow field to quantize.

        Returns:
            jnp.ndarray: The quantized flow field.
        """
        # Quantize the flow field to discrete levels and map to 0-255
        step = 2 * self.max_speed / self.resolution_levels
        # Compute the level index k for each displacement
        k = jnp.round((flow_field + self.max_speed) / step)
        # Clip k to valid range [0, resolution_levels]
        k = jnp.clip(k, 0, self.resolution_levels)
        # Map k to 0-255 and convert to uint8
        quantized_flow = jnp.round(k * 255 / self.resolution_levels).astype(jnp.uint8)

        return quantized_flow


class CrossCorrelationEstimator(FlowFieldEstimator):
    """Cross-correlation flow estimator using a block-based approach."""

    def __init__(
        self,
        max_speed: float,
        resolution_levels: int,
        background_suppression: float,
        window_size: int,
        overlap: int,
    ):
        """Initialize the estimator with the window size and overlap.

        Args:
            max_speed (float):
                maximum speed of the flow field.
            resolution_levels (int):
                Number of resolution levels for the speeds.
            background_suppression (int):
                Background suppression level.
            window_size (int):
                The size of each interrogation window (square).
            overlap (int): Overlap between consecutive windows.
                The stride is (window_size - overlap).
        """
        super().__init__(max_speed, resolution_levels, background_suppression)

        if window_size < 1:
            raise ValueError("Window size should be greater than 0.")
        self.window_size = window_size

        if overlap >= self.window_size:
            raise ValueError("Overlap should be less than the window size.")
        self.overlap = overlap

    def from_config(config: dict):
        """Create an instance of the estimator from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary with the following keys:
                max_speed (float): Maximum speed of the flow field.
                resolution_levels (int): Number of resolution levels for the speeds.
                background_suppression (int): Background suppression level.
                window_size (int): The size of each interrogation window (square).
                overlap (int): Overlap between consecutive windows.
                    The stride is (window_size - overlap).
        """
        return CrossCorrelationEstimator(
            config.get("MAX_SPEED", 2.0),
            config.get("RESOLUTION_LEVELS", 4),
            config.get("BACKGROUND_SUPPRESSION", 0),
            config.get("WINDOW_SIZE", 32),
            config.get("OVERLAP", 16),
        )

    def _compute_flow(self, images: jnp.ndarray) -> jnp.ndarray:
        """Compute the flow field using the last two images in `images`.

        Args:
            images (jnp.ndarray): Shape (H, W, N), where N >= 2, representing a sequence
                of images with height H, width W, and N frames.

        Returns:
            jnp.ndarray: Shape (num_blocks_y, num_blocks_x, 2), containing the estimated
                displacements (dx, dy) for each interrogation block.

        Raises:
            ValueError: If the number of images (N) is less than 2 or if the window size
                is larger than the image dimensions.
        """
        if images.shape[2] < 2:
            raise ValueError(
                "At least two images are required to compute optical flow."
            )

        # Extract the last two images
        img1, img2 = images[..., -2], images[..., -1]
        H, W = img1.shape

        # Validate window size against image dimensions
        if self.window_size > H or self.window_size > W:
            raise ValueError(
                f"Window size {self.window_size} is larger than image dimensions {H}x{W}."
            )

        # Retrieve parameters
        window_size = self.window_size
        overlap = self.overlap
        stride = window_size - overlap  # Step size between blocks
        search_radius = int(self.max_speed)  # TODO sub-pixel accuracy

        # Pad img2 to handle search windows near edges
        padded_img2 = jnp.pad(
            img2,
            ((search_radius, search_radius), (search_radius, search_radius)),
            mode="constant",
            constant_values=0,
        )

        # Calculate the number of blocks
        num_blocks_y = (H - window_size) // stride
        num_blocks_x = (W - window_size) // stride

        # Define the block computation function
        @jax.jit
        def compute_flow_for_block(i, j):
            """Compute displacement for a block at position (i, j)."""
            y = i * stride
            x = j * stride

            # Extract block from img1
            block1 = jax.lax.dynamic_slice(
                img1,
                (y, x),  # Start indices (dynamic)
                (window_size, window_size),  # Slice sizes (static)
            )

            # Extract search window from padded_img2
            search_win = jax.lax.dynamic_slice(
                padded_img2,
                (y, x),
                (window_size + 2 * search_radius, window_size + 2 * search_radius),
            )

            # Compute cross-correlation
            corr = jax.scipy.signal.correlate2d(search_win, block1, mode="valid")

            # Find maximum correlation position
            k, h = jnp.unravel_index(jnp.argmax(corr), corr.shape)

            # Compute displacement
            dy = k - search_radius
            dx = h - search_radius

            return jnp.array([dx, dy])

        # Create block index arrays
        i_values = jnp.arange(num_blocks_y)
        j_values = jnp.arange(num_blocks_x)

        # Compute flow for all blocks
        flow_field = jax.vmap(
            lambda i: jax.vmap(lambda j: compute_flow_for_block(i, j))(j_values)
        )(i_values)

        return flow_field
