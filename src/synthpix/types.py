"""Type aliases for SynthPix library."""

from typing import TypeAlias
from collections.abc import Sequence
import numpy as np
from typing_extensions import Self
import jax.numpy as jnp

from dataclasses import dataclass, field
from jax import tree_util

PRNGKey: TypeAlias = jnp.ndarray


@tree_util.register_pytree_node_class
@dataclass(frozen=False)
class ImageGenerationParameters:
    """Dataclass representing generated images along with their parameters."""

    seeding_densities: jnp.ndarray
    diameter_ranges: jnp.ndarray
    intensity_ranges: jnp.ndarray
    rho_ranges: jnp.ndarray

    def tree_flatten(self) -> tuple[
        tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ],
        None,
    ]:
        """Flattens the ImageGenerationParameters into its constituent parts.

        Returns:
            A tuple containing the flattened children and auxiliary data.
        """
        children = (
            self.seeding_densities,
            self.diameter_ranges,
            self.intensity_ranges,
            self.rho_ranges,
        )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        """Reconstructs an ImageGenerationParameters from its flattened children.

        Args:
            aux_data: Auxiliary data (not used here).
            children: Tuple containing the flattened fields of the
                ImageGenerationParameters.

        Returns:
            An instance of ImageGenerationParameters.
        """
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=False)
class SynthpixBatch:
    """Dataclass representing a batch of SynthPix data."""

    images1: jnp.ndarray
    images2: jnp.ndarray
    flow_fields: jnp.ndarray
    params: ImageGenerationParameters | None = None
    done: jnp.ndarray | None = None
    mask: jnp.ndarray | None = None

    def update(self, **kwargs) -> Self:
        """Return a new SynthpixBatch with updated fields.

        Args:
            **kwargs: Fields to update in the batch.

        Returns:
            A new SynthpixBatch instance with updated fields.
        """
        return self.__class__(
            images1=kwargs.get("images1", self.images1),
            images2=kwargs.get("images2", self.images2),
            flow_fields=kwargs.get("flow_fields", self.flow_fields),
            params=kwargs.get("params", self.params),
            done=kwargs.get("done", self.done),
            mask=kwargs.get("mask", self.mask),
        )

    def tree_flatten(
        self,
    ) -> tuple[tuple[jnp.ndarray | None | ImageGenerationParameters, ...], None]:
        """Flattens the SynthpixBatch into its constituent parts.

        Returns:
            A tuple containing the flattened children and auxiliary data.
        """
        children = (
            self.images1,
            self.images2,
            self.flow_fields,
            self.params,
            self.done,
            self.mask,
        )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstructs a SynthpixBatch from its flattened children.

        Args:
            aux_data: Auxiliary data (not used here).

            children: Tuple containing the flattened fields of the SynthpixBatch.
        """
        return cls(*children)


@dataclass(frozen=True)
class SchedulerData:
    """Dataclass representing a batch returned by a scheduler."""

    flow_fields: np.ndarray
    images1: np.ndarray | None = None
    images2: np.ndarray | None = None
    mask: np.ndarray | None = None

    def update(self, **kwargs) -> Self:
        """Return a new SchedulerBatch with updated fields.

        Args:
            **kwargs: Fields to update in the batch.

        Returns:
            A new SchedulerBatch instance with updated fields.
        """
        return self.__class__(
            flow_fields=kwargs.get("flow_fields", self.flow_fields),
            images1=kwargs.get("images1", self.images1),
            images2=kwargs.get("images2", self.images2),
            mask=kwargs.get("mask", self.mask),
        )


@dataclass(frozen=True)
class ImageGenerationSpecification:
    """Dataclass representing parameters for image generation.

    Details:
        batch_size: Number of image pairs to generate.
        image_shape: (height, width) of the output image in pixels.
        img_offset: (y, x) offset to apply to the generated images in pixels.
        seeding_density_range: (min, max) range of density of particles
            in the images.
        p_hide_img1: Probability of hiding particles in the first image.
        p_hide_img2: Probability of hiding particles in the second image.
        diameter_ranges: Array of shape (N, 2) containing the minimum
            and maximum particle diameter in pixels.
        diameter_var: Variance of the particle diameter.
        intensity_ranges: Array of shape (N, 2) containing the minimum
            and maximum peak intensity (I0).
        intensity_var: Variance of the particle intensity.
        rho_ranges: Array of shape (N, 2) containing the minimum and maximum
            correlation coefficient (rho).
        rho_var: Variance of the correlation coefficient.
        dt: Time step for the simulation, used to scale the velocity
            to compute the displacement.
        noise_uniform: Maximum amplitude of the uniform noise to add.
        noise_gaussian_mean: Mean of the Gaussian noise to add.
        noise_gaussian_std: Standard deviation of the Gaussian noise to add.
    """

    batch_size: int = 300
    image_shape: tuple[int, int] = (256, 256)
    img_offset: tuple[int | float, int | float] = (128, 128)
    seeding_density_range: tuple[int | float, int | float] = (0.01, 0.02)
    p_hide_img1: float = 0.01
    p_hide_img2: float = 0.01
    diameter_ranges: Sequence[tuple[int | float, int | float]] = field(
        default_factory=lambda: [(0.1, 1.0)]
    )
    diameter_var: float = 1.0
    intensity_ranges: Sequence[tuple[int | float, int | float]] = field(
        default_factory=lambda: [(50, 200)]
    )
    intensity_var: float = 1.0
    rho_ranges: Sequence[tuple[int | float, int | float]] = field(
        default_factory=lambda: [(-0.99, 0.99)]
    )
    rho_var: float = 1.0
    dt: float = 1.0
    noise_uniform: float = 0.0
    noise_gaussian_mean: float = 0.0
    noise_gaussian_std: float = 0.0

    def __post_init__(self):
        """Validate the fields of the dataclass."""
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if (
            not isinstance(self.image_shape, tuple)
            or len(self.image_shape) != 2
            or not all(isinstance(s, int) and s > 0 for s in self.image_shape)
        ):
            raise ValueError("image_shape must be a tuple of two positive integers.")
        if not (0.0 <= self.p_hide_img1 <= 1.0):
            raise ValueError("p_hide_img1 must be between 0 and 1.")
        if not (0.0 <= self.p_hide_img2 <= 1.0):
            raise ValueError("p_hide_img2 must be between 0 and 1.")

        if (
            not isinstance(self.image_shape, tuple)
            or len(self.image_shape) != 2
            or not all(isinstance(s, int) and s > 0 for s in self.image_shape)
        ):
            raise ValueError("image_shape must be a tuple of two positive integers.")

        if not (
            isinstance(self.img_offset, tuple)
            and len(self.img_offset) == 2
            and all(isinstance(s, (int, float)) and s >= 0 for s in self.img_offset)
        ):
            raise ValueError("img_offset must be a tuple of two non-negative numbers.")

        if (
            not isinstance(self.seeding_density_range, tuple)
            or len(self.seeding_density_range) != 2
            or not all(
                isinstance(s, (int, float)) and s >= 0
                for s in self.seeding_density_range
            )
        ):
            raise ValueError(
                "seeding_density_range must be a tuple of two non-negative numbers."
            )

        if self.seeding_density_range[0] > self.seeding_density_range[1]:
            raise ValueError("seeding_density_range must be in the form (min, max).")

        # Check diameter_ranges
        if not (
            isinstance(self.diameter_ranges, list)
            and len(self.diameter_ranges) > 0
            and all(isinstance(r, tuple) and len(r) == 2 for r in self.diameter_ranges)
        ):
            raise ValueError("diameter_ranges must be a list of (min, max) tuples.")
        if not all(0 < d1 <= d2 for d1, d2 in self.diameter_ranges):
            raise ValueError("Each diameter_range must satisfy 0 < min <= max.")

        if not (
            isinstance(self.intensity_ranges, list)
            and len(self.intensity_ranges) > 0
            and all(isinstance(r, tuple) and len(r) == 2 for r in self.intensity_ranges)
        ):
            raise ValueError("intensity_ranges must be a list of (min, max) tuples.")
        if not all(0 < d1 <= d2 for d1, d2 in self.intensity_ranges):
            raise ValueError("Each intensity_range must satisfy 0 < min <= max.")

        if not (
            isinstance(self.rho_ranges, list)
            and len(self.rho_ranges) > 0
            and all(isinstance(r, tuple) and len(r) == 2 for r in self.rho_ranges)
        ):
            raise ValueError("rho_ranges must be a list of (min, max) tuples.")

        if not all(-1 < r1 <= r2 < 1 for r1, r2 in self.rho_ranges):
            raise ValueError("Each rho_range must satisfy -1 < min <= max < 1.")

        if not (isinstance(self.diameter_var, (int, float)) and 0 <= self.diameter_var):
            raise ValueError("diameter_var must be a non-negative number.")
        if not (
            isinstance(self.intensity_var, (int, float)) and 0 <= self.intensity_var
        ):
            raise ValueError("intensity_var must be a non-negative number.")
        if not (isinstance(self.rho_var, (int, float)) and 0 <= self.rho_var):
            raise ValueError("rho_var must be a non-negative number.")

        if not (
            isinstance(self.noise_uniform, (int, float)) and (0 <= self.noise_uniform)
        ):
            raise ValueError("noise_uniform must be a non-negative number.")

        if not (
            isinstance(self.noise_gaussian_mean, (int, float))
            and self.noise_gaussian_mean >= 0
        ):
            raise ValueError("noise_gaussian_mean must be a non-negative number.")
        if (
            not isinstance(self.noise_gaussian_std, (int, float))
            or self.noise_gaussian_std < 0
        ):
            raise ValueError("noise_gaussian_std must be a non-negative number.")
        if not isinstance(self.dt, (int, float)) or self.dt <= 0:
            raise ValueError("dt must be a positive number.")

    def update(self, **kwargs) -> Self:
        """Return a new ImageGenerationSpecification with updated fields.

        Args:
            **kwargs: Fields to update in the specification.

        Returns:
            A new ImageGenerationSpecification instance with updated fields.
        """
        return self.__class__(
            batch_size=kwargs.get("batch_size", self.batch_size),
            image_shape=kwargs.get("image_shape", self.image_shape),
            img_offset=kwargs.get("img_offset", self.img_offset),
            seeding_density_range=kwargs.get(
                "seeding_density_range", self.seeding_density_range
            ),
            p_hide_img1=kwargs.get("p_hide_img1", self.p_hide_img1),
            p_hide_img2=kwargs.get("p_hide_img2", self.p_hide_img2),
            diameter_ranges=kwargs.get("diameter_ranges", self.diameter_ranges),
            diameter_var=kwargs.get("diameter_var", self.diameter_var),
            intensity_ranges=kwargs.get("intensity_ranges", self.intensity_ranges),
            intensity_var=kwargs.get("intensity_var", self.intensity_var),
            rho_ranges=kwargs.get("rho_ranges", self.rho_ranges),
            rho_var=kwargs.get("rho_var", self.rho_var),
            dt=kwargs.get("dt", self.dt),
            noise_uniform=kwargs.get("noise_uniform", self.noise_uniform),
            noise_gaussian_mean=kwargs.get(
                "noise_gaussian_mean", self.noise_gaussian_mean
            ),
            noise_gaussian_std=kwargs.get(
                "noise_gaussian_std", self.noise_gaussian_std
            ),
        )
