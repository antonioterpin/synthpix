"""Class for generating synthetic images from flow fields."""

import os
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from goggles import get_logger
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from typing_extensions import Self

from synthpix.data_generate import (
    generate_images_from_flow,
    input_check_gen_img_from_flow,
)
from synthpix.scheduler.episodic import EpisodicFlowFieldScheduler
from synthpix.scheduler.protocol import SchedulerProtocol
from synthpix.types import ImageGenerationSpecification, PRNGKey, SynthpixBatch
from synthpix.utils import (
    DEBUG_JIT,
    SYNTHPIX_SCOPE,
    flow_field_adapter,
    input_check_flow_field_adapter,
)

from .base import Sampler

logger = get_logger(__name__, scope=SYNTHPIX_SCOPE)


class SyntheticImageSampler(Sampler):
    """Iterator class that generates synthetic images from flow fields.

    This class repeatedly samples flow fields from a FlowFieldScheduler,
    and for each, generates a specified number of synthetic images using
    JAX random keys.
    The generation is performed by a JAX-compatible synthesis function
    passed by the user.
    The sampler yields batches of synthetic images, and automatically switches
    to a new flow field after generating a defined number of images from the
    current one.

    Typical usage involves feeding the resulting batches into a model training
    loop or downstream processing pipeline.

    If the underlying scheduler is an episodic scheduler, it automatically
    outputs also a done flag to indicate the end of an episode.
    """

    def __init__(  # noqa: PLR0912, PLR0915
        self,
        scheduler: SchedulerProtocol,
        batches_per_flow_batch: int,
        flow_fields_per_batch: int,
        flow_field_size: tuple[float, float],
        resolution: float,
        velocities_per_pixel: float,
        seed: int,
        max_speed_x: float,
        max_speed_y: float,
        min_speed_x: float,
        min_speed_y: float,
        output_units: str,
        device_ids: Sequence[int] | None = None,
        generation_specification: ImageGenerationSpecification | None = None,
        mask_images: jnp.ndarray | None = None,
        histogram: jnp.ndarray | None = None,
    ):
        """Initializes the SyntheticImageSampler.

        Args:
            scheduler:
                An instance of FlowFieldScheduler that provides flow fields.
            batches_per_flow_batch:
                Number of batches of (imgs1, imgs2, flows) tuples
                per flow field batch.
            flow_fields_per_batch: Number of flow fields to use per batch.
            flow_field_size: Area in which the flow field has been calculated
                in a length measure unit. (e.g in meters, cm, etc.)
            resolution: Resolution of the images in pixels per unit length.
            velocities_per_pixel:
                Number of velocities per pixel in the output flow field.
            seed: Random seed for JAX PRNG.
            max_speed_x: Maximum speed in the x-direction for the flow field
                in length measure unit per seconds.
            max_speed_y: Maximum speed in the y-direction for the flow field
                in length measure unit per seconds.
            min_speed_x: Minimum speed in the x-direction for the flow field
                in length measure unit per seconds.
            min_speed_y: Minimum speed in the y-direction for the flow field
                in length measure unit per seconds.
            output_units: Units of the output flow field.
                Can be 'pixels' or 'measure units'.
            device_ids: List of device IDs to use for sharding the
                flow fields and images.
            generation_specification: Specification for image generation.
            mask_images: Optional binary mask to apply during image generation.
                Should be a jnp.ndarray of shape (height, width).
            histogram: Optional histogram to match during image generation.
                Should be a jnp.ndarray of shape (256,).
                NOTE: Histogram equalization is very slow!

        Raises:
            ValueError: If parameters are invalid or incompatible.
        """
        if generation_specification is None:
            generation_specification = ImageGenerationSpecification()

        # unpack for convenience
        self.batch_size = generation_specification.batch_size
        image_shape = generation_specification.image_shape
        img_offset = generation_specification.img_offset
        dt = generation_specification.dt

        super().__init__(scheduler, self.batch_size)

        # Check provided mask
        if mask_images is not None and mask_images.shape != image_shape:
            raise ValueError(
                f"Mask shape {mask_images.shape} does not match image shape "
                f"{image_shape}."
            )
        self.mask_images = mask_images
        # Check provided histogram
        if histogram is not None and (
            histogram.shape != (256,)
            or not jnp.isclose(histogram.sum(), image_shape[0] * image_shape[1])
        ):
            raise ValueError(
                "Histogram must be a (256,) array and "
                "sum to the number of pixels in the image."
            )
        self.histogram = histogram

        # Name of the axis for the device mesh
        self.shard_fields = "fields"

        # Select the devices based on the provided device IDs
        all_devices = jax.devices()
        if device_ids is None:
            devices = all_devices
            logger.info(
                "No device IDs provided. Using all available devices "
                "for sharding."
            )
        else:
            devices = [
                all_devices[i] for i in device_ids if i < len(all_devices)
            ]
            if len(devices) == 0:
                raise ValueError("No valid device IDs provided.")
            logger.info(f"Using devices {devices} for sharding.")

        self.ndevices = len(devices)

        # We want to shard a key to each device
        # and duplicate the flow field.
        # The idea is that each device will generate a num_images images
        # and then stack it with the images generated by the other GPUs.
        self.mesh = Mesh(devices, axis_names=(self.shard_fields,))

        self.sharding = NamedSharding(
            self.mesh,
            PartitionSpec(
                self.shard_fields,
            ),
        )

        if (
            not isinstance(batches_per_flow_batch, int)
            or batches_per_flow_batch <= 0
        ):
            raise ValueError(
                "batches_per_flow_batch must be a positive integer."
            )
        self.batches_per_flow_batch = batches_per_flow_batch
        if (
            isinstance(self.scheduler, EpisodicFlowFieldScheduler)
            and batches_per_flow_batch != 1
        ):
            self.batches_per_flow_batch = 1
            logger.warning(
                "Using batches_per_flow_batch = 1 for episodic setting."
            )

        if (
            not isinstance(flow_fields_per_batch, int)
            or flow_fields_per_batch <= 0
        ):
            raise ValueError(
                "flow_fields_per_batch must be a positive integer."
            )
        if flow_fields_per_batch > self.batch_size:
            raise ValueError("flow_fields_per_batch must be <= batch_size.")
        self.flow_fields_per_batch = flow_fields_per_batch

        # Make sure the batch size is divisible by the number of devices
        if self.batch_size % self.ndevices != 0:
            self.batch_size = (
                self.batch_size // self.ndevices + 1
            ) * self.ndevices
            logger.warning(
                f"Batch size was not divisible by the number of devices. "
                f"Setting batch_size to {self.batch_size}."
            )

        if (
            # not isinstance(flow_field_size, tuple) or
            # len(flow_field_size) != 2 or
            not all(
                isinstance(s, int | float) and s > 0 for s in flow_field_size
            )
        ):
            raise ValueError(
                "flow_field_size must be a tuple of two positive numbers."
            )
        self.flow_field_size = flow_field_size

        # Use the scheduler to get the flow field shape
        flow_field_shape = scheduler.get_flow_fields_shape()
        if (
            not isinstance(flow_field_shape, tuple)
            or len(flow_field_shape) != 3
            or (flow_field_shape[2] != 2 and flow_field_shape[2] != 3)
            or not all(isinstance(s, int) and s > 0 for s in flow_field_shape)
        ):
            raise ValueError(
                "scheduler.get_flow_fields_shape must return a tuple "
                "of three positive integers with the last being 2 or 3; "
                f"got {flow_field_shape}."
            )
        flow_field_shape = (flow_field_shape[0], flow_field_shape[1])

        if not isinstance(resolution, int | float) or resolution <= 0:
            raise ValueError("resolution must be a positive number.")
        self.resolution = resolution

        if not isinstance(velocities_per_pixel, int | float):
            raise ValueError("velocities_per_pixel must be a number.")
        if velocities_per_pixel <= 0:
            raise ValueError("velocities_per_pixel must be a positive number.")
        self.velocities_per_pixel = velocities_per_pixel
        self.output_flow_field_shape = (
            int(image_shape[0] * velocities_per_pixel),
            int(image_shape[1] * velocities_per_pixel),
        )

        self.max_diameter = max(
            r[1] for r in generation_specification.diameter_ranges
        )

        if output_units not in ["pixels", "measure units per second"]:
            raise ValueError(
                "output_units must be 'pixels' or 'measure units per second'."
            )
        self.output_units = output_units

        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a positive integer.")
        self.seed = seed

        if self.batch_size % flow_fields_per_batch != 0:
            extra_batch_size = self.batch_size % flow_fields_per_batch
            logger.warning(
                "batch_size was not divisible by number of flows per batch. "
                "There will be one more sample for the first "
                f"{extra_batch_size} flow fields of each batch."
            )

        # Check min and max speeds
        if not isinstance(max_speed_x, int | float):
            raise ValueError("max_speed_x must be a number.")
        if not isinstance(max_speed_y, int | float):
            raise ValueError("max_speed_y must be a number.")
        if not isinstance(min_speed_x, int | float):
            raise ValueError("min_speed_x must be a number.")
        if not isinstance(min_speed_y, int | float):
            raise ValueError("min_speed_y must be a number.")
        if max_speed_x < min_speed_x:
            raise ValueError("max_speed_x must be greater than min_speed_x.")
        if max_speed_y < min_speed_y:
            raise ValueError("max_speed_y must be greater than min_speed_y.")

        # Clip the max and min speeds.
        # Positive values of min speeds and negative values of max speeds
        # are not useful to create the position bounds
        max_speed_x = max(0.0, max_speed_x)
        max_speed_y = max(0.0, max_speed_y)
        min_speed_x = min(0.0, min_speed_x)
        min_speed_y = min(0.0, min_speed_y)

        logger.info(
            f"max_speed_x: {max_speed_x},\n max_speed_y: {max_speed_y},\n "
            f"min_speed_x: {min_speed_x},\n min_speed_y: {min_speed_y}"
        )

        # Calculate the resolution of the flow field
        # in grid steps per length measure unit
        self.flow_field_res_y = flow_field_shape[0] / flow_field_size[0]
        self.flow_field_res_x = flow_field_shape[1] / flow_field_size[1]

        # Calculate the position bounds offset in length measure unit
        position_bounds_offset = (
            img_offset[0] - max_speed_y * dt,
            img_offset[1] - max_speed_x * dt,
        )

        # Position bounds in length measure unit
        position_bounds = (
            image_shape[0] / resolution + max_speed_y * dt - min_speed_y * dt,
            image_shape[1] / resolution + max_speed_x * dt - min_speed_x * dt,
        )

        # Check if the position bounds offset is negative or if the
        # position bounds exceed the flow field size
        if position_bounds_offset[0] < 0 or position_bounds_offset[1] < 0:
            raise ValueError(
                f"The image is too close the flow field left or top edge. "
                "The minimum image offset is "
                f"{(max_speed_y * dt, max_speed_x * dt)}."
            )
        if (
            position_bounds[0] + position_bounds_offset[0] > flow_field_size[0]
            or position_bounds[1] + position_bounds_offset[1]
            > flow_field_size[1]
        ):
            raise ValueError(
                f"The size {flow_field_size} of the flow field is too small. "
                f"It must be at least "
                f"({position_bounds[0] + position_bounds_offset[0]},"
                f"{position_bounds[1] + position_bounds_offset[1]})."
            )

        # Compute the particle size in length measure unit
        particle_pixel_radius = int(3 * self.max_diameter / 2)
        particle_size = (2 * particle_pixel_radius + 1) / resolution

        # Check if a bigger position bounds is needed
        if (
            generation_specification.p_hide_img1 > 0
            or generation_specification.p_hide_img2 > 0
            or generation_specification.seeding_density_range[0]
            != generation_specification.seeding_density_range[1]
        ) and (
            particle_size > max_speed_x * dt or particle_size > max_speed_y * dt
        ):
            # Compute the extra length of the position bounds
            extra_length_x = max(0.0, particle_size - max_speed_x * dt)
            extra_length_y = max(0.0, particle_size - max_speed_y * dt)

            # Calculate the position bounds offset in length measure unit
            position_bounds_offset = (
                position_bounds_offset[0] - extra_length_y,
                position_bounds_offset[1] - extra_length_x,
            )

            # Position bounds in length measure unit
            position_bounds = (
                position_bounds[0] + extra_length_y,
                position_bounds[1] + extra_length_x,
            )

        # Compute zero padding in length measure unit
        zero_padding = tuple(max(0, -x) for x in position_bounds_offset)

        # Compute the zero padding in pixels
        pad_y = int(jnp.ceil(zero_padding[0] * self.flow_field_res_y))
        pad_x = int(jnp.ceil(zero_padding[1] * self.flow_field_res_x))
        self.zero_padding = (pad_y, pad_x)

        # Set the position bounds offset
        self.position_bounds_offset = (
            max(0, position_bounds_offset[0]),
            max(0, position_bounds_offset[1]),
        )

        # Calculate the image offset in pixels
        self.img_offset = (
            int((img_offset[0] - position_bounds_offset[0]) * resolution),
            int((img_offset[1] - position_bounds_offset[1]) * resolution),
        )

        # Calculate the position bounds in pixels
        self.position_bounds = (
            int(position_bounds[0] * resolution),
            int(position_bounds[1] * resolution),
        )

        # Update generation specification with adjusted parameters
        generation_specification = generation_specification.update(
            batch_size=self.batch_size // self.ndevices,
            img_offset=self.img_offset,
        )

        if DEBUG_JIT:  # pragma: no cover
            _current_flows = jnp.asarray(
                self.scheduler.get_batch(self.flow_fields_per_batch).flow_fields
            )

            input_check_gen_img_from_flow(
                flow_field=_current_flows,
                parameters=generation_specification,
                position_bounds=self.position_bounds,
                flow_field_res_x=self.flow_field_res_x,
                flow_field_res_y=self.flow_field_res_y,
                mask=self.mask_images,
                histogram=self.histogram,
            )

            input_check_flow_field_adapter(
                flow_field=_current_flows,
                new_flow_field_shape=self.output_flow_field_shape,
                image_shape=generation_specification.image_shape,
                img_offset=self.img_offset,
                resolution=self.resolution,
                res_x=self.flow_field_res_x,
                res_y=self.flow_field_res_y,
                position_bounds=self.position_bounds,
                position_bounds_offset=self.position_bounds_offset,
                batch_size=generation_specification.batch_size,
                output_units=self.output_units,
                dt=generation_specification.dt,
                zero_padding=self.zero_padding,
            )

        self.img_gen_fn_jit = lambda key, flow: generate_images_from_flow(
            key=key,
            flow_field=flow,
            parameters=generation_specification,
            position_bounds=self.position_bounds,
            flow_field_res_x=self.flow_field_res_x,
            flow_field_res_y=self.flow_field_res_y,
            mask=self.mask_images,
            histogram=self.histogram,
        )

        self.flow_field_adapter_jit = lambda flow: flow_field_adapter(
            flow,
            new_flow_field_shape=self.output_flow_field_shape,
            image_shape=generation_specification.image_shape,
            img_offset=self.img_offset,
            resolution=self.resolution,
            res_x=self.flow_field_res_x,
            res_y=self.flow_field_res_y,
            position_bounds=self.position_bounds,
            position_bounds_offset=self.position_bounds_offset,
            batch_size=generation_specification.batch_size,
            output_units=self.output_units,
            dt=generation_specification.dt,
            zero_padding=self.zero_padding,
        )
        if not DEBUG_JIT:
            self.img_gen_fn_jit = jax.jit(
                jax.shard_map(
                    self.img_gen_fn_jit,
                    mesh=self.mesh,
                    in_specs=(
                        PartitionSpec(self.shard_fields),
                        PartitionSpec(self.shard_fields),
                    ),
                    out_specs=(
                        PartitionSpec(self.shard_fields),
                        PartitionSpec(self.shard_fields),
                        PartitionSpec(self.shard_fields),
                    ),
                )
            )
            self.flow_field_adapter_jit = jax.jit(
                jax.shard_map(
                    self.flow_field_adapter_jit,
                    mesh=self.mesh,
                    in_specs=PartitionSpec(self.shard_fields),
                    out_specs=(
                        PartitionSpec(self.shard_fields),
                        PartitionSpec(self.shard_fields),
                    ),
                )
            )

        self._reset()

    def _reset(self) -> None:
        """Resets the state variables to their initial values."""
        self._rng: PRNGKey = jax.random.PRNGKey(self.seed)
        self._current_flows: np.ndarray | Array | None = None
        self.output_flow_fields: Array | None = None
        self._batches_generated: int = 0

    def _get_next(self) -> SynthpixBatch:
        """Generates the next batch of synthetic images.

        Raises:
            RuntimeError: If flow field generation or processing fails.

        Returns:
            The next batch of data as a SynthpixBatch instance.
        """
        # Check if we need to initialize or switch to a new batch of flow fields
        if (
            self._current_flows is None
            or self._batches_generated >= self.batches_per_flow_batch
        ):
            # Reset the batch counter
            self._batches_generated = 0

            scheduler_batch = self.scheduler.get_batch(
                self.flow_fields_per_batch
            )
            self._current_flows = scheduler_batch.flow_fields
            # Notice that self._mask refers to the current mask provided by the
            # scheduler denoting the valid flows of the current batch,
            # shape (batch_size,)
            self._mask_scheduler = scheduler_batch.mask
            # While instead self.mask_images refers to the static mask
            # provided at initialization
            self._files_scheduler = scheduler_batch.files

            # Shard the flow fields across devices
            current_flows_array = jnp.array(
                self._current_flows, device=self.sharding
            )
            # Creating the output flow field
            self.output_flow_fields, self._current_flows = (
                self.flow_field_adapter_jit(current_flows_array)
            )
            if isinstance(self.output_flow_fields, Array):
                self.output_flow_fields.block_until_ready()
            if isinstance(self._current_flows, Array):
                self._current_flows.block_until_ready()

        # Generate a new random key for image generation
        self._rng, subkey = jax.random.split(self._rng)
        keys = jax.random.split(subkey, self.ndevices)

        # Generate a new batch of images using the current flow fields
        imgs1, imgs2, params = self.img_gen_fn_jit(keys, self._current_flows)
        if self.output_flow_fields is None:
            raise RuntimeError("output_flow_fields is None.")

        self._batches_generated += 1

        return SynthpixBatch(
            images1=imgs1,
            images2=imgs2,
            flow_fields=self.output_flow_fields,
            params=params,
            done=None,
            mask=(
                jnp.array(self._mask_scheduler)
                if self._mask_scheduler is not None
                else None
            ),
            files=self._files_scheduler,
        )

    @classmethod
    def from_config(cls, scheduler: SchedulerProtocol, config: dict) -> Self:
        """Creates a SyntheticImageSampler from a configuration dictionary.

        Args:
            scheduler:
                An instance of FlowFieldScheduler that provides flow fields.
            config: Configuration dictionary containing the parameters
                for the sampler. The parameters include all the arguments
                required by the SyntheticImageSampler constructor, except for
                the scheduler. The mask and histogram parameters are optional
                and are provided as paths to .npy files instead.
                mask: Optional path to a .npy file containing a mask.
                    Mask must be a 2D array with 1 where unmasked,
                    0 where masked.
                histogram: Optional path to a .npy file containing a
                    histogram of the flow field.
                    Histogram must be a 1D array with 256 bins,
                    summing to number of pixels.

        Returns:
            An instance of SyntheticImageSampler.

        Raises:
            KeyError: If required configuration keys are missing.
            ValueError:
                If configuration values are invalid or files don't exist.
        """
        # Parse mask
        mask_path = config.get("mask")
        if mask_path is not None:
            if not isinstance(mask_path, str):
                raise ValueError(
                    "mask must be a string representing the mask path."
                )
            if not os.path.isfile(mask_path):
                raise ValueError(f"Mask file {mask_path} does not exist.")
            mask_array = np.load(mask_path)
            if not np.isin(mask_array, [0, 1]).all():
                raise ValueError("Mask must only contain 0 and 1 values.")
            mask_images = jnp.array(mask_array)
        else:
            mask_images = None

        # Parse histogram
        histogram_path = config.get("histogram")

        if histogram_path is not None:
            if not isinstance(histogram_path, str):
                raise ValueError(
                    "histogram must be a string representing the histogram "
                    "path."
                )
            if not os.path.isfile(histogram_path):
                raise ValueError(
                    f"Histogram file {histogram_path} does not exist."
                )
            hist_array = np.load(histogram_path)
            histogram = jnp.array(hist_array)
        else:
            histogram = None

        try:
            gs = ImageGenerationSpecification(
                batch_size=config["batch_size"],
                image_shape=tuple(config["image_shape"]),
                img_offset=tuple(config["img_offset"]),
                seeding_density_range=tuple(config["seeding_density_range"]),
                p_hide_img1=config["p_hide_img1"],
                p_hide_img2=config["p_hide_img2"],
                diameter_ranges=[tuple(t) for t in config["diameter_ranges"]],
                diameter_var=config["diameter_var"],
                intensity_ranges=[tuple(t) for t in config["intensity_ranges"]],
                intensity_var=config["intensity_var"],
                rho_ranges=[tuple(t) for t in config["rho_ranges"]],
                rho_var=config["rho_var"],
                dt=config["dt"],
                noise_uniform=config["noise_uniform"],
                noise_gaussian_mean=config["noise_gaussian_mean"],
                noise_gaussian_std=config["noise_gaussian_std"],
            )
            return cls(
                scheduler=scheduler,
                batches_per_flow_batch=config["batches_per_flow_batch"],
                flow_fields_per_batch=config["flow_fields_per_batch"],
                flow_field_size=tuple(config["flow_field_size"]),
                resolution=config["resolution"],
                velocities_per_pixel=config["velocities_per_pixel"],
                seed=config["seed"],
                max_speed_x=config["max_speed_x"],
                max_speed_y=config["max_speed_y"],
                min_speed_x=config["min_speed_x"],
                min_speed_y=config["min_speed_y"],
                output_units=str(config["output_units"]),
                device_ids=config.get("device_ids"),
                generation_specification=gs,
                mask_images=mask_images,
                histogram=histogram,
            )
        except KeyError as e:
            raise KeyError(
                f"Missing key in configuration: {e}. "
                "Please check the configuration file using the "
                "synthpix.sanity script."
            ) from e
