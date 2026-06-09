"""KinematicDataSource: procedurally generated random displacement fields.

This data source implements the random flow-field generator (RFG) of the
*kinematic training* strategy (Manickathan, Mucignat & Lunati, "Kinematic
training of convolutional neural networks for particle image velocimetry",
Meas. Sci. Technol. 33, 124006, 2022). Instead of reading displacement
fields from disk, it synthesises a fresh smooth random field for every
record, which the synthetic image pipeline then renders into a particle
image pair. It is selected with ``scheduler_class: "kinematic"`` and only
makes sense with ``include_images: false`` (the images are generated, not
read).

The per-field algorithm (Manickathan et al. 2022, section 2.2):

1. Draw each velocity component independently from ``U(-1, 1)`` over the
   image grid.
2. Low-pass filter each component with a Gaussian of width ``sigma`` to
   give the field a finite correlation length.
3. Scale the field so that its peak displacement magnitude equals ``a``.

``sigma`` and ``a`` are drawn uniformly per field from the configured
ranges (defaults follow the paper's Table 3: ``sigma`` in 5-100 px and
``a`` in 0-16 px). Generation is deterministic in the record index so the
Grain pipeline can checkpoint and replay it exactly.
"""

from typing import Any

import grain.python as grain
import numpy as np
from scipy.ndimage import gaussian_filter

DEFAULT_NUM_EXAMPLES = 18278
DEFAULT_IMAGE_SHAPE = (256, 256)
DEFAULT_FILTER_SIGMA_RANGE = (5.0, 100.0)
DEFAULT_MAX_DISPLACEMENT_RANGE = (0.0, 16.0)


class KinematicDataSource(grain.RandomAccessDataSource):
    """Random-flow data source for kinematic training.

    Generates smooth random displacement fields on the fly rather than
    loading them from files. See the module docstring for the generation
    algorithm and references.
    """

    def __init__(
        self,
        num_examples: int = DEFAULT_NUM_EXAMPLES,
        image_shape: tuple[int, int] = DEFAULT_IMAGE_SHAPE,
        filter_sigma_range: tuple[float, float] = DEFAULT_FILTER_SIGMA_RANGE,
        max_displacement_range: tuple[
            float, float
        ] = DEFAULT_MAX_DISPLACEMENT_RANGE,
        seed: int = 0,
    ) -> None:
        """Initialize the KinematicDataSource.

        Args:
            num_examples: Number of distinct flow fields in the (virtual)
                dataset. Acts as the epoch length; with ``loop: true`` the
                Grain sampler reshuffles and re-renders these fields
                indefinitely. Larger values give more flow-field variety.
            image_shape: ``(height, width)`` of the generated flow field in
                pixels.
            filter_sigma_range: ``(min, max)`` of the Gaussian filter width
                ``sigma`` (px), sampled uniformly per field.
            max_displacement_range: ``(min, max)`` of the peak displacement
                magnitude ``a`` (px), sampled uniformly per field.
            seed: Base seed; field ``i`` is generated from ``(seed, i)`` so
                generation is deterministic and checkpoint-replayable.

        Raises:
            ValueError: If any argument is out of range or malformed.
        """
        if not isinstance(num_examples, int) or num_examples <= 0:
            raise ValueError("num_examples must be a positive integer.")
        if len(image_shape) != 2 or not all(
            isinstance(s, int) and s > 0 for s in image_shape
        ):
            raise ValueError(
                "image_shape must be two positive integers (height, width)."
            )
        for name, rng in (
            ("filter_sigma_range", filter_sigma_range),
            ("max_displacement_range", max_displacement_range),
        ):
            if len(rng) != 2 or rng[0] > rng[1] or rng[0] < 0:
                raise ValueError(
                    f"{name} must be a (min, max) pair with 0 <= min <= max."
                )

        self._num_examples = num_examples
        self._image_shape = (int(image_shape[0]), int(image_shape[1]))
        self._filter_sigma_range = (
            float(filter_sigma_range[0]),
            float(filter_sigma_range[1]),
        )
        self._max_displacement_range = (
            float(max_displacement_range[0]),
            float(max_displacement_range[1]),
        )
        self._seed = int(seed)
        super().__init__()

    def __len__(self) -> int:
        """Return the number of flow fields in the dataset.

        Returns:
            The configured number of generated flow fields.
        """
        return self._num_examples

    def __repr__(self) -> str:
        """Return a stable string representation for Grain checkpointing.

        Returns:
            A representation capturing every generation parameter.
        """
        return (
            f"KinematicDataSource(num_examples={self._num_examples}, "
            f"image_shape={self._image_shape}, "
            f"filter_sigma_range={self._filter_sigma_range}, "
            f"max_displacement_range={self._max_displacement_range}, "
            f"seed={self._seed})"
        )

    @property
    def include_images(self) -> bool:
        """Return whether the source provides real images.

        Returns:
            Always ``False``; kinematic images are generated downstream.
        """
        return False

    def __getitem__(self, record_key: int) -> dict[str, Any]:
        """Generate the flow field for a record.

        Args:
            record_key: Index of the flow field to generate.

        Returns:
            Dictionary with the generated ``flow_fields`` array of shape
            ``(H, W, 2)`` and a synthetic ``file`` identifier.
        """
        flow = self._generate_flow_field(record_key)
        return {
            "flow_fields": flow,
            "file": f"kinematic://{record_key}",
        }

    def _generate_flow_field(self, record_key: int) -> np.ndarray:
        """Generate one smooth random displacement field.

        Args:
            record_key: Index seeding the deterministic generation.

        Returns:
            Flow field of shape ``(H, W, 2)`` and dtype ``float32`` whose
            channels are the ``(x, y)`` displacement components in pixels.
        """
        rng = np.random.default_rng((self._seed, record_key))
        sigma = rng.uniform(*self._filter_sigma_range)
        max_displacement = rng.uniform(*self._max_displacement_range)

        noise = rng.uniform(-1.0, 1.0, size=(*self._image_shape, 2))
        # Filter the two spatial axes only; leave the component axis intact.
        smoothed = gaussian_filter(noise, sigma=(sigma, sigma, 0.0))
        flow = self._scale_to_max_displacement(smoothed, max_displacement)
        return flow.astype(np.float32)

    @staticmethod
    def _scale_to_max_displacement(
        field: np.ndarray, max_displacement: float
    ) -> np.ndarray:
        """Scale a vector field to a target peak displacement magnitude.

        Args:
            field: Vector field of shape ``(H, W, 2)``.
            max_displacement: Desired peak displacement magnitude (px).

        Returns:
            The field scaled so that its largest displacement magnitude
            equals ``max_displacement`` (zero when the field is uniform).
        """
        magnitude = np.sqrt(field[..., 0] ** 2 + field[..., 1] ** 2)
        peak = float(magnitude.max())
        scale = max_displacement / peak if peak > 0.0 else 0.0
        return field * scale

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "KinematicDataSource":
        """Create a KinematicDataSource from a dataset configuration.

        Args:
            config: Dataset configuration dictionary. Recognised keys are
                ``num_examples``, ``image_shape``, ``filter_sigma_range``,
                ``max_displacement_range`` and ``seed``; all are optional.

        Returns:
            A configured KinematicDataSource instance.
        """
        image_shape = tuple(config.get("image_shape", DEFAULT_IMAGE_SHAPE))
        sigma_range = tuple(
            config.get("filter_sigma_range", DEFAULT_FILTER_SIGMA_RANGE)
        )
        displacement_range = tuple(
            config.get("max_displacement_range", DEFAULT_MAX_DISPLACEMENT_RANGE)
        )
        return cls(
            num_examples=config.get("num_examples", DEFAULT_NUM_EXAMPLES),
            image_shape=image_shape,
            filter_sigma_range=sigma_range,
            max_displacement_range=displacement_range,
            seed=config.get("seed", 0),
        )
