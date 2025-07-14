"""SyntheticImageSampler class for generating synthetic images from flow fields."""
from typing import Callable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from ..data_generate import input_check_gen_img_from_flow
from ..utils import (
    DEBUG_JIT,
    flow_field_adapter,
    input_check_flow_field_adapter,
    logger,
)


class SyntheticImageSampler:
    """Iterator class that generates synthetic images from flow fields.

    This class repeatedly samples flow fields from a FlowFieldScheduler, and for each,
    generates a specified number of synthetic images using JAX random keys.
    The generation is performed by a JAX-compatible synthesis function passed by the user.
    The sampler yields batches of synthetic images, and automatically switches to a new
    flow field after generating a defined number of images from the current one.

    Typical usage involves feeding the resulting batches into a model training loop or
    downstream processing pipeline.

    If the underlying scheduler is an episodic scheduler, it automatically outputs also a
    done flag to indicate the end of an episode.

    Predefined Configurations:
        - JHTDB: Parameters for a specific case using JHTDB data.
    """

    def __init__(
        self,
        scheduler,
        img_gen_fn: Callable[..., jnp.ndarray],
        batches_per_flow_batch: int,
        batch_size: int,
        flow_fields_per_batch: int,
        flow_field_size: Tuple[float, float],
        image_shape: Tuple[int, int],
        resolution: float,
        velocities_per_pixel: float,
        img_offset: Tuple[float, float],
        seeding_density_range: Tuple[float, float],
        p_hide_img1: float,
        p_hide_img2: float,
        diameter_ranges: List[List[float]],
        diameter_var: float,
        intensity_ranges: List[List[float]],
        intensity_var: float,
        rho_ranges: List[List[float]],
        rho_var: float,
        dt: float,
        seed: int,
        max_speed_x: float,
        max_speed_y: float,
        min_speed_x: float,
        min_speed_y: float,
        output_units: str,
        noise_level: float,
        device_ids: Optional[Sequence[int]] = None,
    ):
        """Initializes the SyntheticImageSampler.

        Args:
            scheduler: An instance of FlowFieldScheduler that provides flow fields.
            img_gen_fn: Callable[..., jnp.ndarray]
                JAX-compatible function (flow_field, key, ...) -> batch of images.
            batches_per_flow_batch: int
                Number of batches of (imgs1, imgs2, flows) tuples per flow field batch.
            batch_size: int
                Number of synthetic image couples per batch.
            flow_fields_per_batch: int
                Number of flow fields to use per batch.
            flow_field_size: Tuple[float, float]
                Area in which the flow field has been calculated
                in a length measure unit. (e.g in meters, cm, etc.)
            image_shape: Tuple[int, int]
                Shape of the synthetic images.
            resolution: float
                Resolution of the images in pixels per unit length.
            velocities_per_pixel: float
                Number of velocities per pixel in the output flow field.
            img_offset: Tuple[float, float]
                Distance in the two axes from the top left corner of the flow field
                and the top left corner of the image a length measure unit.
            seeding_density_range: Tuple[float, float]
                Range of density of particles in the images.
            p_hide_img1: float
                Probability of hiding particles in the first image.
            p_hide_img2: float
                Probability of hiding particles in the second image.
            diameter_ranges: List[List[float, float], ...]
                List of ranges of diameters for particles.
            diameter_var: float
                Variance of the diameters for particles.
            intensity_ranges: List[List[float, float], ...]
                List of ranges of intensities for particles.
            intensity_var: float
                Variance of the intensities for particles.
            rho_ranges: List[List[float, float], ...]
                List of ranges of correlation coefficients for particles.
            rho_var: float
                Variance of the correlation coefficients for particles.
            dt: float
                Time step for the simulation.
            seed: int
                Random seed for JAX PRNG.
            max_speed_x: float
                Maximum speed in the x-direction for the flow field
                in length measure unit per seconds.
            max_speed_y: float
                Maximum speed in the y-direction for the flow field
                in length measure unit per seconds.
            min_speed_x: float
                Minimum speed in the x-direction for the flow field
                in length measure unit per seconds.
            min_speed_y: float
                Minimum speed in the y-direction for the flow field
                in length measure unit per seconds.
            output_units: str
                Units of the output flow field. Can be 'pixels' or 'measure units'.
            noise_level: float
                Maximum amplitude of the uniform noise to add.
            device_ids: Sequence[int]
                List of device IDs to use for sharding the flow fields and images.
        """
        # Name of the axis for the device mesh
        self.shard_fields = "fields"

        # Select the devices based on the provided device IDs
        all_devices = jax.devices()
        if device_ids is None:
            devices = all_devices
            logger.info(
                "No device IDs provided. Using all available devices for sharding."
            )
        else:
            devices = [all_devices[i] for i in device_ids if i < len(all_devices)]
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

        if not hasattr(scheduler, "__iter__"):
            raise ValueError("scheduler must be an iterable object.")
        if not hasattr(scheduler, "__next__"):
            raise ValueError(
                "scheduler must be an iterable object with __next__ method."
            )
        self.scheduler = scheduler

        if not callable(img_gen_fn):
            raise ValueError("img_gen_fn must be a callable function.")

        if not isinstance(batches_per_flow_batch, int) or batches_per_flow_batch <= 0:
            raise ValueError("batches_per_flow_batch must be a positive integer.")
        self.batches_per_flow_batch = batches_per_flow_batch

        if not isinstance(flow_fields_per_batch, int) or flow_fields_per_batch <= 0:
            raise ValueError("flow_fields_per_batch must be a positive integer.")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if flow_fields_per_batch > batch_size:
            raise ValueError(
                "flow_fields_per_batch must be less than or equal to batch_size."
            )
        self.flow_fields_per_batch = flow_fields_per_batch

        # Make sure the batch size is divisible by the number of devices
        if batch_size % self.ndevices != 0:
            batch_size = (batch_size // self.ndevices + 1) * self.ndevices
            logger.warning(
                f"Batch size was not divisible by the number of devices. "
                f"Setting batch_size to {batch_size}."
            )
        self.batch_size = batch_size

        if (
            not isinstance(flow_field_size, tuple)
            or len(flow_field_size) != 2
            or not all(isinstance(s, (int, float)) and s > 0 for s in flow_field_size)
        ):
            raise ValueError("flow_field_size must be a tuple of two positive numbers.")
        self.flow_field_size = flow_field_size

        # Use the scheduler to get the flow field shape
        flow_field_shape = scheduler.get_flow_fields_shape()
        flow_field_shape = (flow_field_shape[0], flow_field_shape[1])
        if (
            not isinstance(flow_field_shape, tuple)
            or len(flow_field_shape) != 2
            or not all(isinstance(s, int) and s > 0 for s in flow_field_shape)
        ):
            raise ValueError(
                "flow_field_shape must be a tuple of two positive integers."
            )

        if (
            not isinstance(image_shape, tuple)
            or len(image_shape) != 2
            or not all(isinstance(s, int) and s > 0 for s in image_shape)
        ):
            raise ValueError("image_shape must be a tuple of two positive integers.")
        self.image_shape = image_shape

        if not isinstance(resolution, (int, float)) or resolution <= 0:
            raise ValueError("resolution must be a positive number.")
        self.resolution = resolution

        if (
            not isinstance(velocities_per_pixel, (int, float))
            or velocities_per_pixel <= 0
        ):
            raise ValueError("velocities_per_pixel must be a positive number.")
        self.velocities_per_pixel = velocities_per_pixel
        self.output_flow_field_shape = (
            int(image_shape[0] * velocities_per_pixel),
            int(image_shape[1] * velocities_per_pixel),
        )

        if (
            not isinstance(img_offset, tuple)
            or len(img_offset) != 2
            or not all(isinstance(s, (int, float)) and s >= 0 for s in img_offset)
        ):
            raise ValueError("img_offset must be a tuple of two non-negative numbers.")

        if (
            not isinstance(seeding_density_range, tuple)
            or len(seeding_density_range) != 2
            or not all(
                isinstance(s, (int, float)) and s >= 0 for s in seeding_density_range
            )
        ):
            raise ValueError(
                "seeding_density_range must be a tuple of two non-negative numbers."
            )
        if seeding_density_range[0] > seeding_density_range[1]:
            raise ValueError("seeding_density_range must be in the form (min, max).")
        self.seeding_density_range = seeding_density_range

        if not (0 <= p_hide_img1 <= 1):
            raise ValueError("p_hide_img1 must be between 0 and 1.")
        self.p_hide_img1 = p_hide_img1

        if not (0 <= p_hide_img2 <= 1):
            raise ValueError("p_hide_img2 must be between 0 and 1.")
        self.p_hide_img2 = p_hide_img2

        if (
            not isinstance(diameter_ranges, list)
            or len(diameter_ranges) == 0
            or not all(
                isinstance(r, (list, tuple))
                and len(r) == 2
                and all(isinstance(d, (int, float)) and d > 0 for d in r)
                and r[0] <= r[1]
                for r in diameter_ranges
            )
        ):
            raise ValueError(
                "diameter_ranges must be a non-empty list "
                "of [min, max] pairs, with 0 < min <= max."
            )
        self.diameter_ranges = jnp.array(diameter_ranges)

        self.max_diameter = max(r[1] for r in diameter_ranges)

        if not isinstance(diameter_var, (int, float)) or diameter_var < 0:
            raise ValueError("diameter_var must be a non-negative number.")
        self.diameter_var = diameter_var

        if (
            not isinstance(intensity_ranges, list)
            or len(intensity_ranges) == 0
            or not all(
                isinstance(r, (list, tuple))
                and len(r) == 2
                and all(isinstance(i, (int, float)) and i >= 0 for i in r)
                and r[0] <= r[1]
                for r in intensity_ranges
            )
        ):
            raise ValueError(
                "intensity_ranges must be a non-empty list "
                "of [min, max] pairs, with 0 <= min <= max."
            )
        self.intensity_ranges = jnp.array(intensity_ranges)

        if not isinstance(intensity_var, (int, float)) or intensity_var < 0:
            raise ValueError("intensity_var must be a non-negative number.")
        self.intensity_var = intensity_var

        if (
            not isinstance(rho_ranges, list)
            or len(rho_ranges) == 0
            or not all(
                isinstance(r, (list, tuple))
                and len(r) == 2
                and all(isinstance(x, (int, float)) and -1.0 < x < 1.0 for x in r)
                and r[0] <= r[1]
                for r in rho_ranges
            )
        ):
            raise ValueError(
                "rho_ranges must be a non-empty list of [min, max] "
                "pairs, each in (-1, 1), with min <= max."
            )
        self.rho_ranges = jnp.array(rho_ranges)

        if not isinstance(rho_var, (int, float)) or rho_var < 0:
            raise ValueError("rho_var must be a non-negative number.")
        self.rho_var = rho_var

        if not isinstance(dt, (int, float)):
            raise ValueError("dt must be a scalar (int or float)")
        self.dt = dt

        if not isinstance(output_units, str) or output_units not in [
            "pixels",
            "measure units per second",
        ]:
            raise ValueError(
                "output_units must be 'pixels' or 'measure units per second'."
            )
        self.output_units = output_units

        if not isinstance(noise_level, (int, float)) or noise_level < 0:
            raise ValueError("noise_level must be a non-negative number.")
        self.noise_level = noise_level

        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a positive integer.")
        self.seed = seed

        if batch_size % flow_fields_per_batch != 0:
            extra_batch_size = batch_size % flow_fields_per_batch
            logger.warning(
                f"batch_size was not divisible by number of flows per batch. "
                f"There will be one more sample for the first {extra_batch_size}"
                f" flow fields of each batch."
            )

        # Check min and max speeds
        if not isinstance(max_speed_x, (int, float)):
            raise ValueError("max_speed_x must be a number.")
        if not isinstance(max_speed_y, (int, float)):
            raise ValueError("max_speed_y must be a number.")
        if not isinstance(min_speed_x, (int, float)):
            raise ValueError("min_speed_x must be a number.")
        if not isinstance(min_speed_y, (int, float)):
            raise ValueError("min_speed_y must be a number.")
        if max_speed_x < min_speed_x:
            raise ValueError("max_speed_x must be greater than min_speed_x.")
        if max_speed_y < min_speed_y:
            raise ValueError("max_speed_y must be greater than min_speed_y.")

        # Clip the max and min speeds.
        # Positive values of min speeds and negative values of max speeds
        # are not useful to create the position bounds
        if max_speed_x < 0 or max_speed_y < 0:
            max_speed_x = 0.0
            max_speed_y = 0.0
        if min_speed_x > 0 or min_speed_y > 0:
            min_speed_x = 0.0
            min_speed_y = 0.0

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

        # Check if the position bounds offset is negative or if the position bounds
        # exceed the flow field size
        if position_bounds_offset[0] < 0 or position_bounds_offset[1] < 0:
            raise ValueError(
                f"The image is too close the flow field left or top edge. "
                f"The minimum image offset is {(max_speed_y * dt, max_speed_x * dt)}."
            )
        if (
            position_bounds[0] + position_bounds_offset[0] > flow_field_size[0]
            or position_bounds[1] + position_bounds_offset[1] > flow_field_size[1]
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
            p_hide_img1 > 0
            or p_hide_img2 > 0
            or seeding_density_range[0] != seeding_density_range[1]
        ) and (particle_size > max_speed_x * dt or particle_size > max_speed_y * dt):
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
        self.position_bounds_offset = tuple(max(0, x) for x in position_bounds_offset)

        # Calculate the image offset in pixels
        self.img_offset = (
            int((img_offset[0] - position_bounds_offset[0]) * resolution),
            int((img_offset[1] - position_bounds_offset[1]) * resolution),
        )

        # Calculate the position bounds in pixels
        self.position_bounds = tuple(int(x * resolution) for x in position_bounds)

        if DEBUG_JIT:
            _current_flows = jnp.asarray(
                self.scheduler.get_batch(self.flow_fields_per_batch)
            )

            input_check_gen_img_from_flow(
                key=jax.random.PRNGKey(self.seed),
                flow_field=_current_flows,
                position_bounds=self.position_bounds,
                image_shape=self.image_shape,
                img_offset=self.img_offset,
                num_images=self.batch_size,
                seeding_density_range=self.seeding_density_range,
                p_hide_img1=self.p_hide_img1,
                p_hide_img2=self.p_hide_img2,
                diameter_ranges=self.diameter_ranges,
                diameter_var=self.diameter_var,
                max_diameter=self.max_diameter,
                intensity_ranges=self.intensity_ranges,
                intensity_var=self.intensity_var,
                rho_ranges=self.rho_ranges,
                rho_var=self.rho_var,
                dt=self.dt,
                flow_field_res_x=self.flow_field_res_x,
                flow_field_res_y=self.flow_field_res_y,
                noise_level=self.noise_level,
            )

            input_check_flow_field_adapter(
                flow_field=_current_flows,
                new_flow_field_shape=self.output_flow_field_shape,
                image_shape=self.image_shape,
                img_offset=self.img_offset,
                resolution=self.resolution,
                res_x=self.flow_field_res_x,
                res_y=self.flow_field_res_y,
                position_bounds=self.position_bounds,
                position_bounds_offset=self.position_bounds_offset,
                batch_size=self.batch_size // self.ndevices,
                output_units=self.output_units,
                dt=self.dt,
                zero_padding=self.zero_padding,
            )

        self.img_gen_fn_jit = lambda key, flow: img_gen_fn(
            key=key,
            flow_field=flow,
            position_bounds=self.position_bounds,
            image_shape=self.image_shape,
            img_offset=self.img_offset,
            num_images=self.batch_size // self.ndevices,
            seeding_density_range=seeding_density_range,
            p_hide_img1=self.p_hide_img1,
            p_hide_img2=self.p_hide_img2,
            diameter_ranges=self.diameter_ranges,
            diameter_var=self.diameter_var,
            max_diameter=self.max_diameter,
            intensity_ranges=self.intensity_ranges,
            intensity_var=self.intensity_var,
            rho_ranges=self.rho_ranges,
            rho_var=self.rho_var,
            dt=self.dt,
            flow_field_res_x=self.flow_field_res_x,
            flow_field_res_y=self.flow_field_res_y,
            noise_level=self.noise_level,
        )

        self.flow_field_adapter_jit = lambda flow: flow_field_adapter(
            flow,
            new_flow_field_shape=self.output_flow_field_shape,
            image_shape=self.image_shape,
            img_offset=self.img_offset,
            resolution=self.resolution,
            res_x=self.flow_field_res_x,
            res_y=self.flow_field_res_y,
            position_bounds=self.position_bounds,
            position_bounds_offset=self.position_bounds_offset,
            batch_size=self.batch_size // self.ndevices,
            output_units=self.output_units,
            dt=self.dt,
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
                    out_specs={
                        "images1": PartitionSpec(self.shard_fields),
                        "images2": PartitionSpec(self.shard_fields),
                        "params": {
                            "seeding_densities": PartitionSpec(self.shard_fields),
                            "diameter_ranges": PartitionSpec(self.shard_fields),
                            "intensity_ranges": PartitionSpec(self.shard_fields),
                            "rho_ranges": PartitionSpec(self.shard_fields),
                        },
                    },
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

        if hasattr(scheduler, "episode_length"):
            self._episodic = True
            logger.info("The underlying scheduler is episodic.")
        else:
            self._episodic = False
            logger.info("The underlying scheduler is not episodic.")

        logger.debug("Input arguments of SyntheticImageSampler are valid.")
        logger.debug(f"Flow field scheduler: {self.scheduler}")
        logger.debug(f"Image generation function: {img_gen_fn}")
        logger.debug(f"Batches per flow batch: {self.batches_per_flow_batch}")
        logger.debug(f"Batch size: {self.batch_size}")
        logger.debug(f"Flow fields per batch: {flow_fields_per_batch}")
        logger.debug(f"Flow field shape: {flow_field_shape}")
        logger.debug(f"Flow field size: {self.flow_field_size}")
        logger.debug(f"Image shape: {self.image_shape}")
        logger.debug(f"Resolution: {self.resolution}")
        logger.debug(f"Velocities per pixel: {velocities_per_pixel}")
        logger.debug(f"Image offset: {self.img_offset}")
        logger.debug(f"Seeding density Range: {self.seeding_density_range}")
        logger.debug(f"p_hide_img1: {self.p_hide_img1}")
        logger.debug(f"p_hide_img2: {self.p_hide_img2}")
        logger.debug(f"Diameter ranges: {self.diameter_ranges}")
        logger.debug(f"Diameter var: {self.diameter_var}")
        logger.debug(f"Max diameter: {self.max_diameter}")
        logger.debug(f"Intensity ranges: {self.intensity_ranges}")
        logger.debug(f"Intensity var: {self.intensity_var}")
        logger.debug(f"Rho ranges: {self.rho_ranges}")
        logger.debug(f"Rho var: {self.rho_var}")
        logger.debug(f"dt: {self.dt}")
        logger.debug(f"Seed: {self.seed}")
        logger.debug(f"Max speed x: {max_speed_x}")
        logger.debug(f"Max speed y: {max_speed_y}")
        logger.debug(f"Min speed x: {min_speed_x}")
        logger.debug(f"Min speed y: {min_speed_y}")
        logger.debug(f"Output units: {self.output_units}")
        logger.debug(f"Background level: {self.noise_level}")
        self._rng = jax.random.PRNGKey(seed)
        self._current_flows = None
        self._batches_generated = 0
        self._total_generated_image_couples = 0

    def __iter__(self):
        """Returns the iterator instance itself."""
        return self

    def reset(self, scheduler_reset: bool = True):
        """Resets the state variables to their initial values."""
        self._rng = jax.random.PRNGKey(self.seed)
        self._current_flows = None
        self._batches_generated = 0
        if scheduler_reset:
            self.scheduler.reset()
        logger.debug("Sampler state has been reset.")

    def __next__(self):
        """Generates the next batch of synthetic images.

        Raises:
            StopIteration: can only be thrown by the underlying scheduler.

        Returns:
            batch: dict
                A dictionary containing:
                - "images1": First images of the batch.
                - "images2": Second images of the batch.
                - "flow_fields": Flow fields used to generate the images.
                - "done": (optional) A boolean array indicating if the episode is done.
                - "params": A dictionary with parameters used for image generation.
        """
        # Check if we need to initialize or switch to a new batch of flow fields
        if (
            self._current_flows is None
            or self._batches_generated >= self.batches_per_flow_batch
        ):
            # Reset the batch counter
            self._batches_generated = 0

            # Get the next batch of flow fields from the scheduler
            if self._episodic and self.scheduler.steps_remaining() == 0:
                raise IndexError(
                    "Episode ended. No more flow fields available. "
                    "Use next_episode() to continue."
                )

            _current_flows = self.scheduler.get_batch(self.flow_fields_per_batch)

            # Shard the flow fields across devices
            _current_flows = jnp.array(_current_flows, device=self.sharding)

            # logger.info("Flow fields have been successfully loaded and sharded.")
            logger.debug(f"Current flow fields sharding: {_current_flows.sharding}")

            # Creating the output flow field
            self.output_flow_fields, self._current_flows = self.flow_field_adapter_jit(
                _current_flows
            )

        # Generate a new random key for image generation
        self._rng, subkey = jax.random.split(self._rng)
        keys = jax.random.split(subkey, self.ndevices)

        logger.debug(f"Number of flow fields: {self._current_flows.shape[0]}")
        logger.debug(f"Current flow fields shape: {self._current_flows.shape[1:]}")
        logger.debug(f"Current flow sharding: {self._current_flows.sharding}")
        logger.debug(f"Current random keys: {keys}")
        logger.debug(f"Keys sharding: {keys.sharding}")

        # Generate a new batch of images using the current flow fields
        batch = self.img_gen_fn_jit(keys, self._current_flows)
        imgs1 = batch["images1"]
        imgs2 = batch["images2"]
        batch["flow_fields"] = self.output_flow_fields
        logger.debug(f"imgs1 location: {imgs1.sharding}")
        logger.debug(f"imgs2 location: {imgs2.sharding}")
        logger.debug(f"Current flow fields location: {self._current_flows.sharding}")
        logger.debug(f"Output flow fields location: {self.output_flow_fields.sharding}")
        logger.debug(f"Generated images shape: {imgs1.shape}, {imgs2.shape}")
        logger.debug(f"Output flow fields shape: {self.output_flow_fields.shape}")

        assert (
            imgs1.shape[0] == self.batch_size
        ), f"Expected {self.batch_size} images but got {imgs1.shape[0]}"
        assert (
            imgs2.shape[0] == self.batch_size
        ), f"Expected {self.batch_size} images but got {imgs2.shape[0]}"

        self._batches_generated += 1
        logger.debug(
            f"Generated {self._batches_generated * self.batch_size} " "image couples"
        )
        self._total_generated_image_couples += self.batch_size
        logger.debug(
            f"Generated {self._total_generated_image_couples} image couples so far."
        )

        if self._episodic:
            done = self._make_done()
            batch["done"] = done
        return batch

    def next_episode(self):
        """Flush the current episode and return the first batch of the next one.

        The underlying scheduler is expected to be the prefetching scheduler.

        Returns:
            next(self): dict
                The first batch of the next episode.
        """
        if not hasattr(self.scheduler, "next_episode"):
            raise AttributeError("Underlying scheduler lacks next_episode(), ")

        self.scheduler.next_episode()

        return next(self)

    def _make_done(self):
        """Return a `(batch_size,)` bool array if episodic, else None."""
        if not self._episodic:
            raise NotImplementedError("The underlying scheduler is not episodic.")

        is_last_step = self.scheduler.steps_remaining() == 0
        logger.debug(f"Is last step: {is_last_step}")
        logger.debug(f"Steps remaining: {self.scheduler.steps_remaining()}")
        # broadcast identical flag to every episode (synchronous horizons)
        # implemented like this to make it easier in the future to implement
        # asynchronous horizons
        return jnp.full((self.batch_size,), is_last_step, dtype=bool)

    def shutdown(self):
        """Shutdown the sampler and release resources."""
        logger.info("Shutting down the SyntheticImageSampler.")
        if hasattr(self.scheduler, "shutdown"):
            self.scheduler.shutdown()
        else:
            logger.warning(
                "The underlying scheduler does not have a shutdown method. "
                "Skipping shutdown."
            )
        logger.info("SyntheticImageSampler shutdown complete.")

    @classmethod
    def from_config(cls, scheduler, img_gen_fn, config) -> "SyntheticImageSampler":
        """Creates a SyntheticImageSampler instance from a configuration dictionary.

        Args:
            scheduler: FlowFieldScheduler
                An instance of FlowFieldScheduler that provides flow fields.
            img_gen_fn: Callable[..., jnp.ndarray]
                JAX-compatible function (flow_field, key, ...) -> batch of images.
            config: dict
                Configuration dictionary containing the parameters for the sampler.

        Returns:
            SyntheticImageSampler: An instance of SyntheticImageSampler.
        """
        try:
            return SyntheticImageSampler(
                scheduler=scheduler,
                img_gen_fn=img_gen_fn,
                batches_per_flow_batch=config["batches_per_flow_batch"],
                batch_size=config["batch_size"],
                flow_fields_per_batch=config["flow_fields_per_batch"],
                flow_field_size=tuple(config["flow_field_size"]),
                image_shape=tuple(config["image_shape"]),
                resolution=config["resolution"],
                velocities_per_pixel=config["velocities_per_pixel"],
                img_offset=tuple(config["img_offset"]),
                seeding_density_range=tuple(config["seeding_density_range"]),
                p_hide_img1=config["p_hide_img1"],
                p_hide_img2=config["p_hide_img2"],
                diameter_ranges=config["diameter_ranges"],
                diameter_var=config["diameter_var"],
                intensity_ranges=config["intensity_ranges"],
                intensity_var=config["intensity_var"],
                rho_ranges=config["rho_ranges"],
                rho_var=config["rho_var"],
                dt=config["dt"],
                seed=config["seed"],
                max_speed_x=config["max_speed_x"],
                max_speed_y=config["max_speed_y"],
                min_speed_x=config["min_speed_x"],
                min_speed_y=config["min_speed_y"],
                output_units=config["output_units"],
                noise_level=config["noise_level"],
                device_ids=config.get("device_ids", None),
            )
        except KeyError as e:
            raise KeyError(
                f"Missing key in configuration: {e}. "
                f"Please check the configuration file using the synthpix.sanity script."
            ) from e
