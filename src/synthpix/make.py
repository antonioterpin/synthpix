"""Make module to instantiate SynthPix."""

import os

import jax
import goggles as gg
from rich.console import Console
from rich.text import Text

from synthpix.data_generate import generate_images_from_flow
from synthpix.sampler import RealImageSampler, Sampler, SyntheticImageSampler
from synthpix.scheduler import (
    BaseFlowFieldScheduler,
    EpisodicFlowFieldScheduler,
    HDF5FlowFieldScheduler,
    MATFlowFieldScheduler,
    NumpyFlowFieldScheduler,
    PrefetchingFlowFieldScheduler,
)
from .utils import load_configuration, SYNTHPIX_SCOPE

logger = gg.get_logger(__name__, scope=SYNTHPIX_SCOPE)

def get_base_scheduler(name: str) -> BaseFlowFieldScheduler:
    """Get the base scheduler class by name.

    Args:
        name: Name of the scheduler class.

    Returns:
        The scheduler class.

    Raises:
        ValueError: If the scheduler class is not found.
    """
    SCHEDULERS = {
        ".h5": HDF5FlowFieldScheduler,
        ".mat": MATFlowFieldScheduler,
        ".npy": NumpyFlowFieldScheduler,
    }

    if name not in SCHEDULERS:
        raise ValueError(f"Scheduler class {name} not found.")

    return SCHEDULERS[name]

def make(
    config: str | dict,
    images_from_file: bool = False,
    buffer_size: int = 0,
    episode_length: int = 0,
) -> Sampler:
    """Load the dataset configuration and initialize the sampler.

    The loading file must be a YAML file containing the dataset configuration.
    Extracting images from files is supported only for .mat files.

    Args:
        config: The dataset configuration.
        images_from_file: If true, images are loaded from files.
        buffer_size: Size of the buffer (in batches) for prefetching.
            If 0, no prefetching is used.
        episode_length: Length of the episode for episodic sampling.

    Returns: The initialized sampler.
    """
    # Initialize console for colored output
    console = Console()

    # SynthPix Banner
    banner = [
        r"   ____              _   _     ____  _         ",
        r"  / ___| _   _ _ __ | |_| |__ |  _ \(_)_  __   ",
        r"  \___ \| | | | '_ \| __| '_ \| |_) | \ \/ /   ",
        r"   ___) | |_| | | | | |_| | | |  __/| |>  <    ",
        r"  |____/ \__, |_| |_|\__|_| |_|_|   |_/_/\_\   ",
        r"         |___/                           ",
    ]

    # Define rainbow color cycle
    rainbow_colors = ["red", "yellow", "green", "cyan", "blue", "magenta"]

    # Print each line with cycling rainbow colors using Rich Text
    for line in banner:
        text = Text()
        for idx, char in enumerate(line):
            if char == " ":
                text.append(char)
            else:
                color = rainbow_colors[idx % len(rainbow_colors)]
                text.append(char, style=color)
        console.print(text)

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
    scheduler_class_name = dataset_config["scheduler_class"]
    scheduler_class = get_base_scheduler(scheduler_class_name)
    if "batch_size" not in dataset_config:
        raise ValueError("dataset_config must contain 'batch_size' key.")
    batch_size = dataset_config["batch_size"]
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(images_from_file, bool):
        raise TypeError("images_from_file must be a boolean.")
    if not isinstance(buffer_size, int) or buffer_size < 0:
        raise ValueError("buffer_size must be a non-negative integer.")
    if not isinstance(episode_length, int) or episode_length < 0:
        raise ValueError("episode_length must be a non-negative integer.")
    if "flow_fields_per_batch" not in dataset_config:
        raise ValueError(
            "dataset_config must contain 'flow_fields_per_batch' key."
        )

    # Initialize the random number generator
    cpu = jax.devices("cpu")[0]
    seed = dataset_config.get("seed", 0)
    key = jax.random.PRNGKey(seed)
    key = jax.device_put(key, cpu)

    key, sched_key = jax.random.split(key)

    kwargs = {
        "file_list": dataset_config.get("scheduler_files", []),
        "randomize": dataset_config.get("randomize", False),
        "loop": dataset_config.get("loop", True),
        "key": sched_key,
    }

    if images_from_file:
        if scheduler_class_name != ".mat":
            raise ValueError(
                f"Scheduler class {scheduler_class_name} "
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
        kwargs = {
            **kwargs,
            "include_images": True,
            "output_shape": tuple(
                dataset_config.get("image_shape", (256, 256))
            ),
        }

    scheduler = scheduler_class.from_config(kwargs)

    # If episode_length is specified, use EpisodicFlowFieldScheduler
    if episode_length > 0:
        key, epi_key = jax.random.split(key)
        scheduler = EpisodicFlowFieldScheduler(
            scheduler=scheduler,
            batch_size=batch_size,
            episode_length=episode_length,
            key=epi_key,
        )

    # If buffer_size is specified, use PrefetchingFlowFieldScheduler
    if buffer_size > 0:
        scheduler = PrefetchingFlowFieldScheduler(
            scheduler=scheduler,
            batch_size=batch_size,
            buffer_size=buffer_size,
        )

    if images_from_file:
        sampler = RealImageSampler(scheduler, batch_size=batch_size)
    else:
        # If episode_length is specified, use EpisodicFlowFieldScheduler
        if episode_length > 0 and dataset_config["batches_per_flow_batch"] > 1:
            # NOTE: batches_per_flow_batch is used below by the 
            # synthetic sampler
            logger.warning(
                "Using EpisodicFlowFieldScheduler with batches_per_flow_batch > 1 "
                "may lead to unexpected behavior. "
                "Consider using a single batch per flow field."
            )

        # Initialize the sampler
        sampler = SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=generate_images_from_flow,
            config=dataset_config,
        )

    logger.info(
        f"--- SynthPix sampler and scheduler initialized ---\n{dataset_config}"
    )

    return sampler
