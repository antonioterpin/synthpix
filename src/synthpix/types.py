"""Type aliases for SynthPix library."""

from typing import TypeAlias
import numpy as np
from typing_extensions import Self
import jax.numpy as jnp

from dataclasses import dataclass

PRNGKey: TypeAlias = jnp.ndarray

@dataclass(frozen=True)
class SynthpixBatch:
    """Dataclass representing a batch of SynthPix data."""

    images1: jnp.ndarray
    images2: jnp.ndarray
    flow_fields: jnp.ndarray
    params: jnp.ndarray | None = None
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