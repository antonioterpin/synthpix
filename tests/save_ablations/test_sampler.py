import csv
import timeit

import jax
import numpy as np
import pytest

from synthpix.sampler import SyntheticImageSampler
from synthpix.scheduler import PrefetchingFlowFieldScheduler
from synthpix.utils import load_configuration

sampler_config = load_configuration("config/test_data.yaml")

config = load_configuration("config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_SAMPLER"]


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
                "batches_per_flow_batch",
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
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": True}], indirect=True
)
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("batches_per_flow_batch", [1, 10, 100, 1000])
def test_speed_sampler_sweep_all(
    scheduler,
    batch_size,
    batches_per_flow_batch,
):
    config = sampler_config.copy()
    particles_dim = [[0.8, 1.2]]  # diameter_range
    seeding_density = 0.06  # seeding_density
    config["batch_size"] = batch_size
    config["batches_per_flow_batch"] = batches_per_flow_batch
    config["seeding_density_range"] = (0.06, 0.06)
    config["seed"] = 0
    config["image_shape"] = (512, 512)
    config["img_offset"] = (10, 10)
    config["flow_field_size"] = (532, 532)
    config["resolution"] = 1
    config["max_speed_x"] = 10
    config["max_speed_y"] = 10
    config["min_speed_x"] = 10
    config["min_speed_y"] = 10
    config["dt"] = 1
    config["noise_uniform"] = 0.0
    config["noise_gaussian_mean"] = 0.0
    config["noise_gaussian_std"] = 0.0
    config["flow_fields_per_batch"] = 1
    config["p_hide_img1"] = 0
    config["p_hide_img2"] = 0
    config["rho_ranges"] = [[-0.01, 0.01]]  # rho_range
    config["rho_var"] = 0.0
    config["intensity_ranges"] = [[80.0, 100.0]]  # intensity_range
    config["intensity_var"] = 0.0
    config["diameter_ranges"] = [[0.8, 1.2]]  # diameter_range
    config["diameter_var"] = 0.0

    output_csv = f"batches_per_flow_batch{batches_per_flow_batch}.csv"

    NUMBER_OF_EXECUTIONS = 1000
    REPETITIONS = 10

    num_devices = len(jax.devices())

    # Create the sampler
    prefetching_scheduler = PrefetchingFlowFieldScheduler(
        scheduler=scheduler,
        batch_size=config["flow_fields_per_batch"],
        buffer_size=3 * config["flow_fields_per_batch"],
    )
    sampler = SyntheticImageSampler.from_config(
        scheduler=prefetching_scheduler,
        config=config,
    )

    all_rows = []

    def run_sampler():
        # Time the data sampling (batch_size batches)
        for i, batch in enumerate(sampler):
            batch.images1.block_until_ready()
            batch.images2.block_until_ready()
            batch.flow_fields.block_until_ready()
            batch.params.seeding_densities.block_until_ready()  # type: ignore
            batch.params.diameter_ranges.block_until_ready()  # type: ignore
            batch.params.intensity_ranges.block_until_ready()  # type: ignore
            batch.params.rho_ranges.block_until_ready()  # type: ignore
            if i + 1 >= NUMBER_OF_EXECUTIONS:
                sampler.reset(scheduler_reset=False)
                break

    # Warm up
    run_sampler()

    # Timing
    total_time = timeit.repeat(
        stmt=run_sampler,
        number=1,
        repeat=REPETITIONS,
    )

    timings = np.asarray(total_time)
    timings_per_img = timings / (batch_size * NUMBER_OF_EXECUTIONS)
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
            image_size=512,
            flow_fields_per_batch=1,
            particles_dim=str(particles_dim),
            seeding_density=seeding_density,
            batches_per_flow_batch=batches_per_flow_batch,
            Q1=q1,
            Q3=q3,
            Mean=mean,
            Min=min_,
            Max=max_,
            StdDev=std,
        )
    )

    write_speed_stats_to_csv(output_csv, all_rows)

    prefetching_scheduler.shutdown()
