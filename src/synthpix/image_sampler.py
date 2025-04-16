"""SyntheticImageSampler class for generating synthetic images from flow fields."""
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from synthpix.data_generate import input_check_gen_img_from_flow
from synthpix.utils import (
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
        seeding_density: int,
        p_hide_img1: float,
        p_hide_img2: float,
        diameter_range: Tuple[float, float],
        intensity_range: Tuple[float, float],
        rho_range: Tuple[float, float],
        dt: float,
        seed: int,
        max_speed_x: float,
        max_speed_y: float,
        min_speed_x: float,
        min_speed_y: float,
        output_units: str,
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
            flow_field_shape: Tuple[int, int]
                Shape of the flow field in grid steps.
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
            seeding_density: float
                Density of particles in the images.
            p_hide_img1: float
                Probability of hiding particles in the first image.
            p_hide_img2: float
                Probability of hiding particles in the second image.
            diameter_range: Tuple[float, float]
                Range of diameters for particles.
            intensity_range: Tuple[float, float]
                Range of intensities for particles.
            rho_range: Tuple[float, float]
                Range of correlation coefficients for particles.
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
        """
        # Name of the axis for the device mesh
        self.shard_fields = "fields"

        # Check how many GPUs are available
        num_devices = len(jax.devices())

        # Setup device mesh
        # We want to shard a key to each device
        # and duplicate the flow field.
        # The idea is that each device will generate a num_images images
        # and then stack it with the images generated by the other GPUs.
        devices = mesh_utils.create_device_mesh((num_devices,))
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
        self.flow_fields_per_batch = flow_fields_per_batch

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        # Make sure the batch size is divisible by the number of devices
        if batch_size % num_devices != 0:
            batch_size = (batch_size // num_devices + 1) * num_devices
            logger.warning(
                f"Batch size was not divisible by the number of devices. "
                f"Setting batch_size to {batch_size}."
            )
        self.batch_size = batch_size

        if len(flow_field_size) != 2 or not all(
            isinstance(s, (int, float)) and s > 0 for s in flow_field_size
        ):
            raise ValueError("flow_field_size must be a tuple of two positive numbers.")
        self.flow_field_size = flow_field_size

        # Use the scheduler to get the flow field shape
        flow_field_shape = scheduler.get_flow_fields_shape()
        flow_field_shape = (flow_field_shape[0], flow_field_shape[1])
        if len(flow_field_shape) != 2 or not all(
            isinstance(s, int) and s > 0 for s in flow_field_shape
        ):
            raise ValueError(
                "flow_field_shape must be a tuple of two positive integers."
            )

        if len(image_shape) != 2 or not all(
            isinstance(s, int) and s > 0 for s in image_shape
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

        if len(img_offset) != 2 or not all(
            isinstance(s, (int, float)) and s >= 0 for s in img_offset
        ):
            raise ValueError("img_offset must be a tuple of two non-negative numbers.")

        if (
            not isinstance(seeding_density, float)
            or seeding_density <= 0
            or seeding_density >= 1
        ):
            raise ValueError("seeding_density must be a float between 0 and 1.")
        self.seeding_density = seeding_density

        if not (0 <= p_hide_img1 <= 1):
            raise ValueError("p_hide_img1 must be between 0 and 1.")
        self.p_hide_img1 = p_hide_img1

        if not (0 <= p_hide_img2 <= 1):
            raise ValueError("p_hide_img2 must be between 0 and 1.")
        self.p_hide_img2 = p_hide_img2

        if len(diameter_range) != 2 or not all(
            isinstance(d, (int, float)) and d > 0 for d in diameter_range
        ):
            raise ValueError("diameter_range must be a tuple of two positive floats.")
        self.diameter_range = diameter_range

        if len(intensity_range) != 2 or not all(
            isinstance(i, (int, float)) and i >= 0 for i in intensity_range
        ):
            raise ValueError(
                "intensity_range must be a tuple of two non-negative floats."
            )
        self.intensity_range = intensity_range

        if len(rho_range) != 2 or not all(
            isinstance(r, (int, float)) and -1 <= r <= 1 for r in rho_range
        ):
            raise ValueError(
                "rho_range must be a tuple of two floats between -1 and 1."
            )
        self.rho_range = rho_range

        if not isinstance(dt, (int, float)):
            raise ValueError("dt must be a scalar (int or float)")
        self.dt = dt

        if not isinstance(output_units, str) or output_units not in [
            "pixels",
            "measure units",
        ]:
            raise ValueError("output_units must be 'pixels' or 'measure units'.")
        self.output_units = output_units

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
                f"The size of the flow field is too small."
                f"it must be at least "
                f"({position_bounds[0] + position_bounds_offset[0]},"
                f"{position_bounds[1] + position_bounds_offset[1]})."
            )

        # Compute the particle size in length measure unit
        particle_pixel_radius = int(3 * diameter_range[1] / 2)
        particle_size = (2 * particle_pixel_radius + 1) / resolution

        # Check if a bigger position bounds is needed
        if (p_hide_img1 > 0 or p_hide_img2 > 0) and (
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
        self.position_bounds_offset = tuple(max(0, x) for x in position_bounds_offset)

        # Calculate the image offset in pixels
        self.img_offset = (
            int(img_offset[0] * resolution - position_bounds_offset[0] * resolution),
            int(img_offset[1] * resolution - position_bounds_offset[1] * resolution),
        )

        # Calculate the position bounds in pixels
        self.position_bounds = tuple(int(x * resolution) for x in position_bounds)

        if not DEBUG_JIT:
            self.img_gen_fn_jit = jax.jit(
                shard_map(
                    lambda key, flow: img_gen_fn(
                        key=key,
                        flow_field=flow,
                        position_bounds=self.position_bounds,
                        image_shape=self.image_shape,
                        img_offset=self.img_offset,
                        num_images=self.batch_size // num_devices,
                        seeding_density=self.seeding_density,
                        p_hide_img1=self.p_hide_img1,
                        p_hide_img2=self.p_hide_img2,
                        diameter_range=self.diameter_range,
                        intensity_range=self.intensity_range,
                        rho_range=self.rho_range,
                        dt=self.dt,
                        flow_field_res_x=self.flow_field_res_x,
                        flow_field_res_y=self.flow_field_res_y,
                    ),
                    mesh=self.mesh,
                    in_specs=(
                        PartitionSpec(self.shard_fields),
                        PartitionSpec(self.shard_fields),
                    ),
                    out_specs=(
                        PartitionSpec(self.shard_fields),
                        PartitionSpec(self.shard_fields),
                    ),
                )
            )
        else:
            input_check_gen_img_from_flow(
                self._current_flows,
                self.position_bounds,
                self.image_shape,
                self.img_offset,
                num_images=self.batch_size,
                seeding_density=self.seeding_density,
                p_hide_img1=self.p_hide_img1,
                p_hide_img2=self.p_hide_img2,
                diameter_range=self.diameter_range,
                intensity_range=self.intensity_range,
                rho_range=self.rho_range,
                dt=self.dt,
                flow_field_res_x=self.flow_field_res_x,
                flow_field_res_y=self.flow_field_res_y,
            )
            self.img_gen_fn_jit = lambda key, flow: img_gen_fn(
                key=key,
                flow_field=flow,
                position_bounds=self.position_bounds,
                image_shape=self.image_shape,
                img_offset=self.img_offset,
                num_images=self.batch_size // num_devices,
                seeding_density=seeding_density,
                p_hide_img1=self.p_hide_img1,
                p_hide_img2=self.p_hide_img2,
                diameter_range=self.diameter_range,
                intensity_range=self.intensity_range,
                rho_range=self.rho_range,
                dt=self.dt,
                flow_field_res_x=self.flow_field_res_x,
                flow_field_res_y=self.flow_field_res_y,
            )

        if not DEBUG_JIT:
            self.flow_field_adapter_jit = jax.jit(
                shard_map(
                    lambda flow: flow_field_adapter(
                        flow,
                        new_flow_field_shape=self.output_flow_field_shape,
                        image_shape=self.image_shape,
                        img_offset=self.img_offset,
                        resolution=self.resolution,
                        res_x=self.flow_field_res_x,
                        res_y=self.flow_field_res_y,
                        batch_size=self.batch_size // num_devices,
                        position_bounds=self.position_bounds,
                        position_bounds_offset=self.position_bounds_offset,
                        output_units=self.output_units,
                        dt=self.dt,
                    ),
                    mesh=self.mesh,
                    in_specs=PartitionSpec(self.shard_fields),
                    out_specs=PartitionSpec(self.shard_fields),
                )
            )
        else:
            input_check_flow_field_adapter(
                self._current_flows,
                new_flow_field_shape=self.output_flow_field_shape,
                image_shape=self.image_shape,
                image_offset=self.img_offset,
                resolution=self.resolution,
                res_x=self.flow_field_res_x,
                res_y=self.flow_field_res_y,
                batch_size=self.batch_size,
                output_units=self.output_units,
                dt=self.dt,
            )
            self.flow_field_adapter_jit = flow_field_adapter

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
        logger.debug(f"Seeding density: {self.seeding_density}")
        logger.debug(f"p_hide_img1: {self.p_hide_img1}")
        logger.debug(f"p_hide_img2: {self.p_hide_img2}")
        logger.debug(f"Diameter range: {self.diameter_range}")
        logger.debug(f"Intensity range: {self.intensity_range}")
        logger.debug(f"Rho range: {self.rho_range}")
        logger.debug(f"dt: {self.dt}")
        logger.debug(f"Seed: {self.seed}")
        logger.debug(f"Max speed x: {max_speed_x}")
        logger.debug(f"Max speed y: {max_speed_y}")
        logger.debug(f"Min speed x: {min_speed_x}")
        logger.debug(f"Min speed y: {min_speed_y}")
        logger.debug(f"Output units: {self.output_units}")
        self._rng = jax.random.PRNGKey(seed)
        self._current_flows = None
        self._batches_generated = 0
        self._total_generated_image_couples = 0

    def __iter__(self):
        """Returns the iterator instance itself."""
        return self

    def reset(self):
        """Resets the state variables to their initial values."""
        self._rng = jax.random.PRNGKey(self.seed)
        self._current_flows = None
        self._batches_generated = 0
        self.scheduler.reset()
        logger.debug("Sampler state has been reset.")

    def __next__(self):
        """Generates the next batch of synthetic images.

        Raises:
            StopIteration: Never raised by default, it is thrown by scheduler.

        Returns:
            jnp.ndarray: A batch of synthetic images generated on GPU.
        """
        # Check if we need to initialize or switch to a new batch of flow fields
        if (
            self._current_flows is None
            or self._batches_generated >= self.batches_per_flow_batch
        ):
            # Reset the batch counter
            self._batches_generated = 0

            # Get the next batch of flow fields from the scheduler
            _current_flows = self.scheduler.get_batch(self.flow_fields_per_batch)

            # Shard the flow fields across devices
            _current_flows = jnp.array(_current_flows, device=self.sharding)

            logger.debug(f"Current flow fields sharding: {_current_flows.sharding}")

            # Creating the output flow field
            self.output_flow_fields, self._current_flows = self.flow_field_adapter_jit(
                _current_flows
            )

        # Generate a new random key for image generation
        self._rng, subkey = jax.random.split(self._rng)
        keys = jax.random.split(subkey, jax.device_count())

        logger.debug(f"Number of flow fields: {self._current_flows.shape[0]}")
        logger.debug(f"Current flow fields shape: {self._current_flows.shape[1:]}")
        logger.debug(f"Current random keys: {keys}")

        # Generate a new batch of images using the current flow fields
        imgs1, imgs2 = self.img_gen_fn_jit(keys, self._current_flows)

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

        logger.debug(f"Generated {self._batches_generated} couples of images")
        logger.debug(
            f"Generated {self._batches_generated * self.batch_size} " "image couples"
        )
        self._batches_generated += 1
        self._total_generated_image_couples += self.batch_size
        return imgs1, imgs2, self.output_flow_fields

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
                flow_field_size=config["flow_field_size"],
                image_shape=config["image_shape"],
                resolution=config["resolution"],
                velocities_per_pixel=config["velocities_per_pixel"],
                img_offset=config["img_offset"],
                seeding_density=config["seeding_density"],
                p_hide_img1=config["p_hide_img1"],
                p_hide_img2=config["p_hide_img2"],
                diameter_range=config["diameter_range"],
                intensity_range=config["intensity_range"],
                rho_range=config["rho_range"],
                dt=config["dt"],
                seed=config["seed"],
                max_speed_x=config["max_speed_x"],
                max_speed_y=config["max_speed_y"],
                min_speed_x=config["min_speed_x"],
                min_speed_y=config["min_speed_y"],
                output_units=config["output_units"],
            )
        except KeyError as e:
            raise KeyError(
                f"Missing key in configuration: {e}. "
                f"Please check the configuration file using the synthpix.sanity script."
            ) from e
