"""KinematicDataSource: procedurally generated random displacement fields.

This data source implements the random flow-field generator (RFG) of the
*kinematic training* strategy (Manickathan, Mucignat & Lunati, "Kinematic
training of convolutional neural networks for particle image velocimetry",
Meas. Sci. Technol. 33, 124006, 2022). It generates smooth random
displacement fields according to the paper's equation:

    ds_ref = a * G_sigma * xi

where ``xi`` is independent U(-1, 1) noise, ``G_sigma`` is a Gaussian filter
of width ``sigma``, and ``a`` is the per-field scale sampled uniformly from a
configured range. The synthetic image pipeline then renders particle images
from the generated fields on the fly. It is selected with
``scheduler_class: "kinematic"`` and only makes sense with
``include_images: false`` (the images are generated, not read).

The per-field algorithm (Manickathan et al. 2022, section 2.2):

1. Draw each velocity component independently from ``U(-1, 1)`` over the
   image grid.
2. Low-pass filter each component with a Gaussian of width ``sigma`` to
   give the field a finite correlation length.
3. Scale the field by the sampled factor ``a`` according to ``scale_mode``.

``scale_mode`` selects how ``a`` is interpreted:

* ``"peak"`` (default, paper-faithful): normalize each field so its peak
  displacement magnitude equals ``a`` pixels. The follow-up paper
  parameterizes this same dataset by exactly this quantity -- Manickathan,
  Mucignat & Lunati, Exp. Fluids 64, 161 (2023), Table 1 lists the field
  scale as "Maximum displacement, max(ds_ref) (px): 0-16". Gaussian filtering
  attenuates the noise amplitude by a strongly ``sigma``-dependent factor
  (~1/sigma), so normalizing to a target peak is what keeps ``a`` physically
  meaningful in pixels across the whole ``sigma`` range.
* ``"linear"``: apply ``a`` as a raw multiplier, ds_ref = a * G_sigma * xi,
  the literal form of the equation above. The realized displacement then
  depends strongly on ``sigma`` (large ``sigma`` collapses the motion toward
  zero), so ``a`` is a dimensionless gain, not a pixel target.

``sigma`` and ``a`` are drawn uniformly per field from the configured ranges
(defaults follow the papers' tables: ``sigma`` in 5-100 px and ``a`` in
0-16 px). Generation is deterministic in the record index so the Grain
pipeline can checkpoint and replay it exactly. The dataset is finite and
deterministic: the same index always produces the same field.
"""

from typing import Any

import grain.python as grain
import numpy as np
from scipy.ndimage import gaussian_filter

DEFAULT_NUM_EXAMPLES = 18278
DEFAULT_IMAGE_SHAPE = (256, 256)
DEFAULT_FILTER_SIGMA_RANGE = (5.0, 100.0)
DEFAULT_SCALE_FACTOR_RANGE = (0.0, 16.0)
DEFAULT_SCALE_MODE = "peak"
SCALE_MODES = ("peak", "linear")


class KinematicDataSource(grain.RandomAccessDataSource):
    """Deterministic finite random-access dataset of kinematic flow fields.

    Generates smooth random displacement fields on the fly according to the
    Manickathan et al. 2022 random-flow-field algorithm, without reading from
    files. The dataset is finite (size ``num_examples``) and deterministic by
    record index, making it checkpoint-replayable with Grain. Each epoch
    cycles through the same ``num_examples`` fields in deterministic or
    shuffled order, depending on Grain's sampler configuration. See the
    module docstring for the generation algorithm and references.
    """

    def __init__(
        self,
        num_examples: int = DEFAULT_NUM_EXAMPLES,
        image_shape: tuple[int, int] = DEFAULT_IMAGE_SHAPE,
        filter_sigma_range: tuple[float, float] = DEFAULT_FILTER_SIGMA_RANGE,
        scale_factor_range: tuple[float, float] = DEFAULT_SCALE_FACTOR_RANGE,
        scale_mode: str = DEFAULT_SCALE_MODE,
        seed: int = 0,
    ) -> None:
        """Initialize the KinematicDataSource.

        A finite, deterministic random-access dataset that generates smooth
        random displacement fields according to Manickathan et al. 2022,
        section 2.2: ds_ref = a * G_sigma * xi.

        Args:
            num_examples: Number of distinct flow fields in the finite
                (virtual) dataset. Defines the epoch length; with ``loop: true``
                the Grain sampler cycles through these same ``num_examples``
                fields indefinitely in reshuffled order. Larger values give
                more flow-field variety per epoch.
            image_shape: ``(height, width)`` of the generated flow field in
                pixels.
            filter_sigma_range: ``(min, max)`` of the Gaussian filter width
                ``sigma`` (px), sampled uniformly per field.
            scale_factor_range: ``(min, max)`` of the per-field scale ``a``,
                sampled uniformly per field. Its meaning depends on
                ``scale_mode``: in ``"peak"`` mode it is the target peak
                displacement magnitude (px); in ``"linear"`` mode it is the
                dimensionless multiplier applied directly to the filtered noise.
            scale_mode: How ``a`` is applied. ``"peak"`` (default) normalizes
                each field so its peak displacement magnitude equals ``a`` px
                (paper-faithful, Manickathan et al. 2023 Table 1). ``"linear"``
                multiplies the filtered noise by ``a`` directly
                (ds_ref = a * G_sigma * xi), so the realized displacement is
                ``sigma``-dependent.
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
            ("scale_factor_range", scale_factor_range),
        ):
            if len(rng) != 2 or rng[0] > rng[1] or rng[0] < 0:
                raise ValueError(
                    f"{name} must be a (min, max) pair with 0 <= min <= max."
                )
        if scale_mode not in SCALE_MODES:
            raise ValueError(
                f"scale_mode must be one of {SCALE_MODES}, got {scale_mode!r}."
            )

        self._num_examples = num_examples
        self._image_shape = (int(image_shape[0]), int(image_shape[1]))
        self._filter_sigma_range = (
            float(filter_sigma_range[0]),
            float(filter_sigma_range[1]),
        )
        self._scale_factor_range = (
            float(scale_factor_range[0]),
            float(scale_factor_range[1]),
        )
        self._scale_mode = scale_mode
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
            f"scale_factor_range={self._scale_factor_range}, "
            f"scale_mode={self._scale_mode!r}, "
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

        Implements ds_ref = a * G_sigma * xi, where xi is U(-1,1) noise,
        G_sigma is a Gaussian filter, and a is the per-field scale. In
        ``"peak"`` mode the field is normalized so its peak displacement
        magnitude equals ``a`` (px); in ``"linear"`` mode ``a`` multiplies the
        filtered noise directly.

        Args:
            record_key: Index seeding the deterministic generation.

        Returns:
            Flow field of shape ``(H, W, 2)`` and dtype ``float32`` whose
            channels are the ``(x, y)`` displacement components in pixels.
        """
        rng = np.random.default_rng((self._seed, record_key))
        sigma = rng.uniform(*self._filter_sigma_range)
        scale_factor = rng.uniform(*self._scale_factor_range)

        noise = rng.uniform(-1.0, 1.0, size=(*self._image_shape, 2))
        # Filter the two spatial axes only; leave the component axis intact.
        smoothed = gaussian_filter(noise, sigma=(sigma, sigma, 0.0))
        if self._scale_mode == "peak":
            # Normalize so the peak displacement magnitude equals scale_factor
            # (px); sigma-independent (Manickathan et al. 2023, Table 1).
            flow = self._scale_to_max_displacement(smoothed, scale_factor)
        else:
            # Apply the scale factor as a raw multiplier (sigma-dependent).
            flow = smoothed * scale_factor
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
                ``scale_factor_range``, ``scale_mode`` and ``seed``; all are
                optional.

        Returns:
            A configured KinematicDataSource instance.
        """
        image_shape = tuple(config.get("image_shape", DEFAULT_IMAGE_SHAPE))
        sigma_range = tuple(
            config.get("filter_sigma_range", DEFAULT_FILTER_SIGMA_RANGE)
        )
        scale_factor_range = tuple(
            config.get("scale_factor_range", DEFAULT_SCALE_FACTOR_RANGE)
        )

        return cls(
            num_examples=config.get("num_examples", DEFAULT_NUM_EXAMPLES),
            image_shape=image_shape,
            filter_sigma_range=sigma_range,
            scale_factor_range=scale_factor_range,
            scale_mode=config.get("scale_mode", DEFAULT_SCALE_MODE),
            seed=config.get("seed", 0),
        )
