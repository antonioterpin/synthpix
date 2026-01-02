"""Make module to instantiate SynthPix."""

import os

import jax
import goggles as gg
from rich.console import Console
from rich.text import Text

import grain.python as grain

from synthpix.sampler import RealImageSampler, Sampler, SyntheticImageSampler
from synthpix.scheduler import (
    BaseFlowFieldScheduler,
    EpisodicFlowFieldScheduler,
    HDF5FlowFieldScheduler,
    MATFlowFieldScheduler,
    NumpyFlowFieldScheduler,
    PrefetchingFlowFieldScheduler,
)
from synthpix.data_sources import (
    EpisodicDataSource,
    HDF5DataSource,
    MATDataSource,
    NumpyDataSource,
    FileDataSource,
)
from synthpix.data_sources.adapter import (
    GrainEpisodicAdapter,
    GrainSchedulerAdapter,
)
from .utils import load_configuration, SYNTHPIX_SCOPE

logger = gg.get_logger(__name__, scope=SYNTHPIX_SCOPE)

SCHEDULERS = {
    ".h5": HDF5FlowFieldScheduler,
    ".mat": MATFlowFieldScheduler,
    ".npy": NumpyFlowFieldScheduler,
}

DATA_SOURCES = {
    ".h5": HDF5DataSource,
    ".mat": MATDataSource,
    ".npy": NumpyDataSource,
}


def get_base_scheduler(name: str) -> BaseFlowFieldScheduler:
    """Get the base scheduler class by file extension.

    Args:
        name: File extension identifying the scheduler type (".h5", ".mat", or ".npy").

    Returns:
        The scheduler class corresponding to the file extension.

    Raises:
        ValueError: If the file extension is not supported.
    """
    if name not in SCHEDULERS:
        raise ValueError(
            f"Scheduler class {name} not found. Should be one of {list(SCHEDULERS.keys())}."
        )

    return SCHEDULERS[name]


def get_data_source_class(name: str) -> type[FileDataSource]:
    """Get the data source class by file extension.

    Args:
        name: File extension identifying the scheduler type (".h5", ".mat", or ".npy").

    Returns:
        The data source class corresponding to the file extension.

    Raises:
        ValueError: If the file extension is not supported.
    """
    if name not in DATA_SOURCES:
        raise ValueError(
            f"DataSource class {name} not found. Should be one of {list(DATA_SOURCES.keys())}."
        )

    return DATA_SOURCES[name]


def make(
    config: str | dict,
    use_grain_scheduler: bool = False,
) -> Sampler:
    """Load the dataset configuration and initialize the sampler.

    The loading file must be a YAML file containing the dataset configuration.
    Extracting images from files is supported only for .mat files.

    Required configuration keys:
    - scheduler_class: The file extension of the scheduler to use (".h5", ".mat", or ".npy").
    - batch_size: The batch size for training (positive integer).
    - flow_fields_per_batch: Number of flow fields to use per batch.
    - batches_per_flow_batch: Required when generating synthetic images (include_images=False).

    Optional configuration keys:
    - include_images: Whether to extract real images from files (bool, default False).
    - buffer_size: Size of prefetching buffer (non-negative int, default 0).
    - episode_length: Length of episodes for episodic scheduler (non-negative int, default 0).
    - seed: Random seed (int, default 0).
    - file_list: List of data files (list, default empty).
    - randomize: Whether to randomize file order (bool, default False).
    - loop: Whether to loop through files (bool, default True).
    - image_shape: Shape for image extraction when include_images=True (tuple, default (256, 256)).

    Args:
        config: Either a path to a YAML configuration file or a configuration dictionary.
        use_grain_scheduler: Whether to use the new Grain-based scheduler (default False).

    Returns:
        The initialized sampler (RealImageSampler or SyntheticImageSampler).

    Raises:
        TypeError: If config is not a string or dictionary, or if parameter types are invalid.
        ValueError: If required keys are missing or parameter values are invalid.
        FileNotFoundError: If the configuration file doesn't exist.
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
        raise TypeError("config must be a string or a dictionary.")
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
        raise TypeError("config must be a dictionary.")
    if "scheduler_class" not in dataset_config:
        raise ValueError("config must contain 'scheduler_class' key.")
    batch_size = dataset_config["batch_size"]
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    scheduler_class_name = dataset_config["scheduler_class"]
    if use_grain_scheduler:
        # Just validate existence here, retrieved later
        if scheduler_class_name not in DATA_SOURCES:
            raise ValueError(
                f"DataSource class {scheduler_class_name} not found. Should be one of {list(DATA_SOURCES.keys())}."
            )
    else:
        scheduler_class = get_base_scheduler(scheduler_class_name)

    # Optional parameters
    include_images = dataset_config.get("include_images", False)
    if not isinstance(include_images, bool):
        raise TypeError("include_images must be a boolean.")
    buffer_size = dataset_config.get("buffer_size", 0)
    if not isinstance(buffer_size, int) or buffer_size < 0:
        raise ValueError("buffer_size must be a non-negative integer.")
    episode_length = dataset_config.get("episode_length", 0)
    if not isinstance(episode_length, int) or episode_length < 0:
        raise ValueError("episode_length must be a non-negative integer.")
    if "flow_fields_per_batch" not in dataset_config:
        raise ValueError("config must contain 'flow_fields_per_batch' key.")

    # Initialize the random number generator
    cpu = jax.devices("cpu")[0]
    seed = dataset_config.get("seed", 0)
    key = jax.random.PRNGKey(seed)
    key = jax.device_put(key, cpu)

    key, sched_key = jax.random.split(key)

    kwargs = {
        "file_list": dataset_config.get("file_list", []),
        "randomize": dataset_config.get("randomize", False),
        "loop": dataset_config.get("loop", True),
        "key": sched_key,
        "include_images": include_images,
    }

    if include_images:
        if scheduler_class_name != ".mat":
            raise ValueError(
                f"Scheduler class {scheduler_class_name} "
                "is not supported for file images."
            )
        kwargs = {
            **kwargs,
            "output_shape": tuple(dataset_config.get("image_shape", (256, 256))),
        }

    # Initialize the scheduler (Legacy or Grain)
    if use_grain_scheduler:
        # 1. Instantiate Data Source
        data_source_cls = get_data_source_class(scheduler_class_name)

        # Extract args relevant for DataSource
        ds_kwargs = {}
        if dataset_config.get("file_list"):
            ds_kwargs["dataset_path"] = dataset_config["file_list"]
        else:
            ds_kwargs["dataset_path"] = []

        if include_images:
            ds_kwargs["include_images"] = True
            ds_kwargs["output_shape"] = tuple(
                dataset_config.get("image_shape", (256, 256))
            )

        data_source = data_source_cls(**ds_kwargs)

        # 2. Episodic Wrapping
        is_episodic = episode_length > 0
        if is_episodic:
            data_source = EpisodicDataSource(
                source=data_source,
                batch_size=batch_size,
                episode_length=episode_length,
                seed=dataset_config.get("seed", 0),
            )

        # 3. Grain Sampler & Loader
        # IndexSampler handles shuffling and infinite looping
        sampler_grain = grain.IndexSampler(
            num_records=len(data_source),
            shuffle=kwargs["randomize"],
            seed=dataset_config.get("seed", 0),
            shard_options=grain.NoSharding(),
            num_epochs=None if kwargs["loop"] else 1,
        )

        # Grain Options from Config
        worker_count = dataset_config.get("worker_count", 0)
        num_threads = dataset_config.get("num_threads", 16)
        buffer_size = dataset_config.get("buffer_size", 500)  # Default grain prefetch

        # Enforce worker_count constraints for Grain
        if is_episodic and worker_count > 0:
            raise ValueError(
                f"worker_count must be 0 when using episodic data (episode_length={episode_length}). "
                "Using multiple workers (multiprocessing) with EpisodicDataSource breaks data order "
                "because workers consume the interleaved episode chunks independently."
            )
        elif not is_episodic and worker_count > 0:
            logger.warning(
                f"Using worker_count={worker_count} with non-episodic data. "
                "This enables multiprocessing in Grain. Notice that the order of the data is not preserved,"
                "but still deterministic if you keep the same number of workers."
            )

        grain_loader = grain.DataLoader(
            data_source=data_source,
            sampler=sampler_grain,
            operations=[grain.Batch(batch_size=batch_size, drop_remainder=False)],
            worker_count=worker_count,
            read_options=grain.ReadOptions(
                num_threads=num_threads, prefetch_buffer_size=buffer_size
            ),
        )

        # 4. Adapter
        if is_episodic:
            scheduler = GrainEpisodicAdapter(grain_loader)
        else:
            scheduler = GrainSchedulerAdapter(grain_loader)

    else:
        # Legacy Path
        # Initialize the base scheduler
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

    if include_images:
        sampler = RealImageSampler(scheduler, batch_size=batch_size)
    else:
        batches_per_flow_batch = dataset_config.get("batches_per_flow_batch", None)
        if batches_per_flow_batch is None:
            raise ValueError(
                "config must contain the 'batches_per_flow_batch' key when"
                " generating synthetic images."
            )
        # If episode_length is specified, use EpisodicFlowFieldScheduler
        if episode_length > 0 and batches_per_flow_batch > 1:
            # NOTE: batches_per_flow_batch is used below by the synthetic sampler
            logger.warning(
                "Using EpisodicFlowFieldScheduler with batches_per_flow_batch > 1 "
                "may lead to unexpected behavior. "
                "Consider using a single batch per flow field."
            )

        # Initialize the sampler
        sampler = SyntheticImageSampler.from_config(
            scheduler=scheduler,
            config=dataset_config,
        )

    logger.info(f"--- SynthPix sampler and scheduler initialized ---\n{dataset_config}")

    return sampler
