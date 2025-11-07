"""Type aliases for SynthPix library."""

from typing import TypeAlias
import numpy as np
from typing_extensions import Self
import jax.numpy as jnp

from dataclasses import dataclass
from jax import tree_util

PRNGKey: TypeAlias = jnp.ndarray

@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ImageGenerationParameters:
    """Dataclass representing generated images along with their parameters."""

    seeding_densities: jnp.ndarray
    diameter_ranges: jnp.ndarray
    intensity_ranges: jnp.ndarray
    rho_ranges: jnp.ndarray

    def tree_flatten(self):
        children = (self.seeding_densities, self.diameter_ranges, self.intensity_ranges, self.rho_ranges)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SynthpixBatch:
    """Dataclass representing a batch of SynthPix data."""

    images1: jnp.ndarray
    images2: jnp.ndarray
    flow_fields: jnp.ndarray
    params: ImageGenerationParameters | None = None
    done: jnp.ndarray | None = None

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
        )

    def tree_flatten(self):
        children = (self.images1, self.images2, self.flow_fields, self.params, self.done)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    
@dataclass(frozen=True)
class SchedulerData:
    """Dataclass representing a batch returned by a scheduler."""

    flow_fields: np.ndarray
    images1: np.ndarray | None = None
    images2: np.ndarray | None = None

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
        )
    
@dataclass(frozen=True)
class ImageGenerationSpecification:
    """Dataclass representing parameters for image generation.
    
    Details:
        batch_size: Number of image pairs to generate.
        image_shape: (height, width) of the output image in pixels.
        img_offset: (x, y) offset to apply to the generated images.
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
    img_offset: tuple[int, int] = (128, 128)
    seeding_density_range: tuple[float, float] = (0.01, 0.02)
    p_hide_img1: float = 0.01
    p_hide_img2: float = 0.01
    diameter_ranges: list[tuple[float, float]] = [(0.1, 1.0)]
    diameter_var: float = 1.0
    intensity_ranges: list[tuple[float, float]] = [(50, 200)]
    intensity_var: float = 1.0
    rho_ranges: list[tuple[float, float]] = [(-0.99, 0.99)]
    rho_var: float = 1.0
    dt: float = 1.0
    noise_uniform: float = 0.0
    noise_gaussian_mean: float = 0.0
    noise_gaussian_std: float = 0.0
