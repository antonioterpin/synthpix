"""Make module to instantiate SynthPix."""
import os
from typing import Union

from .data_generate import generate_images_from_flow
from .sampler import RealImageSampler, SyntheticImageSampler
from .scheduler import (
    EpisodicFlowFieldScheduler,
    FloFlowFieldScheduler,
    HDF5FlowFieldScheduler,
    MATFlowFieldScheduler,
    NumpyFlowFieldScheduler,
    PrefetchingFlowFieldScheduler,
)
from .utils import load_configuration, logger

SCHEDULERS = {
    ".h5": HDF5FlowFieldScheduler,
    ".mat": MATFlowFieldScheduler,
    ".npy": NumpyFlowFieldScheduler,
    ".flo": FloFlowFieldScheduler,
}


def make(
    config: str | dict,
    images_from_file: bool = False,
    buffer_size: int = 0,
    episode_length: int = 0,
) -> Union[SyntheticImageSampler, RealImageSampler]:
    """Load the dataset configuration and initialize the sampler.

    The loading file must be a YAML file containing the dataset configuration.
    Extracting images from files is supported only for .mat files.

    Args:
        config (str | dict): The dataset configuration.
        images_from_file (bool): If true, images are loaded from files.
        buffer_size (int): Size of the buffer (in batches) for prefetching.
            If 0, no prefetching is used.
        episode_length (int): Length of the episode for episodic sampling.

    Returns:
        SyntheticImageSampler | RealImageSampler: The initialized sampler.
    """
    # Input validation
    if not isinstance(config, (str, dict)):
        raise TypeError("config_path must be a string or a dictionary.")
    if isinstance(config, str):
        if not config.endswith(".yaml"):
            raise ValueError("config must point to a .yaml file.")
        if not os.path.exists(config):
            raise FileNotFoundError(f"Configuration file {config} not found.")
        if not os.path.isfile(config):
            raise ValueError(f"Configuration path {config} is not a file.")
        # Load the dataset configuration
        dataset_config = load_configuration(config)

        logger.info(f"Loading dataset configuration from {config}")
    elif isinstance(config, dict):
        dataset_config = config
        logger.info("Using provided dataset configuration dictionary.")

    # Configuration validation
    if not isinstance(dataset_config, dict):
        raise TypeError("dataset_config must be a dictionary.")
    if "scheduler_class" not in dataset_config:
        raise ValueError("dataset_config must contain 'scheduler_class' key.")
    if not isinstance(images_from_file, bool):
        raise TypeError("images_from_file must be a boolean.")
    if not isinstance(buffer_size, int) or buffer_size <= 0:
        raise ValueError("buffer_size must be a positive integer.")
    if not isinstance(episode_length, int) or episode_length < 0:
        raise ValueError("episode_length must be a non-negative integer.")

    if images_from_file:
        if dataset_config["scheduler_class"] != ".mat":
            raise ValueError(
                f"Scheduler class {dataset_config['scheduler_class']} "
                "is not supported for file images."
            )
        if (
            "include_images" not in dataset_config
            or not dataset_config["include_images"]
        ):
            logger.warning(
                "The dataset configuration does not have 'include_images' set to True. "
                "It will be set to True by default."
            )
        dataset_config["include_images"] = True

        # Initialize the base scheduler
        base = MATFlowFieldScheduler.from_config(dataset_config)

        # If episode_length is specified, use EpisodicFlowFieldScheduler
        if episode_length > 0:
            sched = EpisodicFlowFieldScheduler(
                base,
                batch_size=dataset_config["batch_size"],
                episode_length=episode_length,
                seed=dataset_config.get("seed"),
            )
        else:
            sched = base

        # If buffer_size is specified, use PrefetchingFlowFieldScheduler
        if buffer_size > 0:
            scheduler = PrefetchingFlowFieldScheduler(
                sched,
                batch_size=dataset_config["batch_size"],
                buffer_size=buffer_size,
            )
        else:
            scheduler = sched

        # Initialize the sampler
        sampler = RealImageSampler(scheduler, batch_size=dataset_config["batch_size"])
    else:
        if dataset_config["scheduler_class"] not in SCHEDULERS:
            raise ValueError(
                f"Scheduler class {dataset_config['scheduler_class']} not found."
            )
        scheduler_class = SCHEDULERS.get(dataset_config["scheduler_class"])

        # Initialize the base scheduler
        base = scheduler_class.from_config(dataset_config)

        # If episode_length is specified, use EpisodicFlowFieldScheduler
        if episode_length > 0:
            sched = EpisodicFlowFieldScheduler(
                base,
                batch_size=dataset_config["flow_fields_per_batch"],
                episode_length=episode_length,
                seed=dataset_config.get("seed"),
            )
        else:
            sched = base

        # If buffer_size is specified, use PrefetchingFlowFieldScheduler
        if buffer_size > 0:
            scheduler = PrefetchingFlowFieldScheduler(
                scheduler=sched,
                batch_size=dataset_config["flow_fields_per_batch"],
                buffer_size=buffer_size,
            )
        else:
            scheduler = sched

        # Initialize the sampler
        sampler = SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=generate_images_from_flow,
            config=dataset_config,
        )

    return sampler
