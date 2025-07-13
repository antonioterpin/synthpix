import csv
import itertools
import timeit

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from synthpix.data_generate import generate_images_from_flow

# Import existing modules
from synthpix.example_flows import get_flow_function
from synthpix.utils import generate_array_flow_field, load_configuration

config = load_configuration("config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_DATA_GEN"]


def write_speed_stats_to_csv(filename, all_rows):
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "GPU_number",
                "batch_size",
                "image_size",
                "flow_fields_per_batch",
                "particles_dim",
                "seeding_density",
                "Q1",
                "Q3",
                "Mean",
                "Min",
                "Max",
                "StdDev",
            ],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


@pytest.mark.run_explicitly
@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connected to the server.",
)
def test_speed_generate_images_sweep_all():
    output_csv = "new_seeding_densities.csv"
    # ---- PARAMETERS TO SWEEP ----
    batch_sizes = [64]
    image_sizes = [512]
    flow_fields_per_batch = [1]
    particles_dims = [[[0.8, 1.2]]]
    seeding_densities = [
        0.001,
        0.005,
        0.010,
        0.015,
        0.020,
        0.025,
        0.030,
        0.035,
        0.040,
        0.045,
        0.050,
        0.055,
        0.060,
    ]

    selected_flow = "horizontal"
    img_offset = (10, 10)
    NUMBER_OF_EXECUTIONS = 1000
    REPETITIONS = 10

    # ---- DEVICE AND MESH SETUP ----
    shard_fields = "fields"
    devices = jax.devices()
    num_devices = len(devices)

    mesh = Mesh(devices, axis_names=(shard_fields,))
    key = jax.random.PRNGKey(0)

    all_rows = []

    # Sweep all parameter combinations
    for (
        batch_size,
        image_size,
        flows_per_batch,
        particles_dim,
        seeding_density,
    ) in itertools.product(
        batch_sizes,
        image_sizes,
        flow_fields_per_batch,
        particles_dims,
        seeding_densities,
    ):
        image_shape = (image_size, image_size)
        position_bounds = (image_shape[0] + 20, image_shape[1] + 20)

        # 1. Create flow field
        flow_field = generate_array_flow_field(
            get_flow_function(selected_flow, position_bounds), position_bounds
        )
        flow_field = jnp.expand_dims(flow_field, axis=0)
        flow_field = jnp.repeat(flow_field, flows_per_batch, axis=0)
        flow_field_sharded = jax.device_put(
            flow_field, NamedSharding(mesh, PartitionSpec(shard_fields))
        )

        # 2. Setup random keys
        keys = jax.random.split(key, num_devices)
        keys = jnp.stack(keys)
        keys_sharded = jax.device_put(
            keys, NamedSharding(mesh, PartitionSpec(shard_fields))
        )
        jax.block_until_ready(keys_sharded)
        jax.block_until_ready(flow_field_sharded)

        seeding_density_range = (seeding_density, seeding_density)

        # 3. Prepare the jit function
        out_specs = {
            "images1": PartitionSpec(shard_fields),
            "images2": PartitionSpec(shard_fields),
            "params": {
                "seeding_densities": PartitionSpec(shard_fields),
                "diameter_ranges": PartitionSpec(shard_fields),
                "intensity_ranges": PartitionSpec(shard_fields),
                "rho_ranges": PartitionSpec(shard_fields),
            },
        }

        jit_generate_images = jax.jit(
            jax.shard_map(
                lambda key, flow: generate_images_from_flow(
                    key=key,
                    flow_field=flow,
                    position_bounds=position_bounds,
                    image_shape=image_shape,
                    img_offset=img_offset,
                    seeding_density_range=seeding_density_range,
                    num_images=batch_size,
                    p_hide_img1=0.00,
                    p_hide_img2=0.00,
                    diameter_ranges=jnp.array(particles_dim),
                    diameter_var=0,
                    intensity_ranges=jnp.array([[80, 100]]),
                    intensity_var=0,
                    noise_level=0,
                    rho_ranges=jnp.array([[-0.01, 0.01]]),
                    rho_var=0,
                ),
                mesh=mesh,
                in_specs=(PartitionSpec(shard_fields), PartitionSpec(shard_fields)),
                out_specs=out_specs,
            )
        )

        def run_generate_jit():
            data = jit_generate_images(keys_sharded, flow_field_sharded)
            imgs1 = data["images1"]
            imgs2 = data["images2"]
            params = data["params"]
            seeding_densities = params["seeding_densities"]
            diameter_ranges = params["diameter_ranges"]
            intensity_ranges = params["intensity_ranges"]
            rho_ranges = params["rho_ranges"]
            imgs1.block_until_ready()
            imgs2.block_until_ready()
            seeding_densities.block_until_ready()
            diameter_ranges.block_until_ready()
            intensity_ranges.block_until_ready()
            rho_ranges.block_until_ready()

        # Warm up
        run_generate_jit()

        # Timing
        total_time_jit = timeit.repeat(
            stmt=run_generate_jit,
            number=NUMBER_OF_EXECUTIONS // num_devices,
            repeat=REPETITIONS,
        )

        timings = np.asarray(total_time_jit)
        timings_per_img = timings / batch_size / NUMBER_OF_EXECUTIONS
        hz_per_img = 1.0 / timings_per_img

        q1 = np.percentile(hz_per_img, 25)
        q3 = np.percentile(hz_per_img, 75)
        mean = np.mean(hz_per_img)
        min_ = np.min(hz_per_img)
        max_ = np.max(hz_per_img)
        std = np.std(hz_per_img)

        all_rows.append(
            dict(
                GPU_number=num_devices,
                batch_size=batch_size,
                image_size=image_size,
                flow_fields_per_batch=flows_per_batch,
                particles_dim=str(particles_dim),
                seeding_density=seeding_density,
                Q1=q1,
                Q3=q3,
                Mean=mean,
                Min=min_,
                Max=max_,
                StdDev=std,
            )
        )

    # --- WRITE TO CSV ---
    write_speed_stats_to_csv(output_csv, all_rows)
