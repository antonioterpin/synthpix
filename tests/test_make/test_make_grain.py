"""Tests for the Grain-based scheduler creation in the `make` module.

These tests verify that the `make` function correctly instantiates the Grain 
data loading stack, including adapters, samplers, and data sources, when 
`use_grain_scheduler=True` is provided. It uses extensive monkeypatching to 
isolate the factory logic from actual Grain or JAX execution.
"""

import importlib
from types import SimpleNamespace

import pytest

# Import the module to be tested
make_mod = importlib.import_module("synthpix.make")
make = make_mod.make


class DummyDataSource:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __len__(self):
        return 10


class DummyEpisodicDataSource:
    def __init__(self, source, batch_size, episode_length, seed):
        self.source = source
        self.batch_size = batch_size
        self.episode_length = episode_length
        self.seed = seed

    def __len__(self):
        return 100


class DummyGrainAdapter:
    def __init__(self, loader):
        self.loader = loader


class DummyLoader:
    def __init__(
        self,
        data_source,
        sampler,
        operations,
        worker_count=0,
        read_options=None,
    ):
        self.data_source = data_source
        self.sampler = sampler
        self.operations = operations
        self.worker_count = worker_count
        self.read_options = read_options


class DummySampler:
    def __init__(self, scheduler, batch_size, *args, **kwargs):
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.kwargs = kwargs

    @classmethod
    def from_config(cls, scheduler, config):
        return cls(scheduler, config["batch_size"], config=config)


@pytest.fixture
def patch_grain(monkeypatch):
    """Patch Grain and DataSource symbols."""
    # Mock Grain
    mock_grain = SimpleNamespace()
    mock_grain.IndexSampler = lambda **kwargs: "IndexSampler"
    mock_grain.DataLoader = DummyLoader
    mock_grain.NoSharding = lambda: "NoSharding"

    class MockReadOptions:
        def __init__(self, num_threads=None, **kwargs):
            self.num_threads = num_threads
            self.kwargs = kwargs

    mock_grain.ReadOptions = MockReadOptions

    from unittest.mock import MagicMock

    mock_grain.Batch = MagicMock(return_value="Batch")

    monkeypatch.setattr(make_mod, "grain", mock_grain)

    # Mock DataSources
    monkeypatch.setitem(make_mod.DATA_SOURCES, ".mat", DummyDataSource)
    monkeypatch.setitem(make_mod.DATA_SOURCES, ".npy", DummyDataSource)
    monkeypatch.setattr(make_mod, "EpisodicDataSource", DummyEpisodicDataSource)

    # Mock Adapters
    monkeypatch.setattr(make_mod, "GrainEpisodicAdapter", DummyGrainAdapter)
    monkeypatch.setattr(make_mod, "GrainSchedulerAdapter", DummyGrainAdapter)

    # Mock Samplers (reuse if possible, or simple mock)
    monkeypatch.setattr(make_mod, "RealImageSampler", DummySampler)
    monkeypatch.setattr(make_mod, "SyntheticImageSampler", DummySampler)

    # Mock Logger and Config loader
    monkeypatch.setattr(
        make_mod,
        "logger",
        SimpleNamespace(info=lambda *_: None, warning=lambda *_: None),
    )
    monkeypatch.setattr(
        make_mod, "load_configuration", lambda p: {}, raising=False
    )


def test_make_grain_basic(patch_grain, monkeypatch):
    """Test standard Grain path instantiation for non-episodic data.

    Verifies that the `make` function correctly sets up a `MATDataSource` 
    (mocked), wraps it in a Grain `DataLoader`, and connects it to a 
    `GrainSchedulerAdapter`.
    """
    cfg = {
        "scheduler_class": ".mat",
        "batch_size": 4,
        "flow_fields_per_batch": 1,
        "include_images": False,
        "batches_per_flow_batch": 1,
        "file_list": ["file1.mat"],
    }

    sampler = make(cfg, use_grain_scheduler=True)

    assert isinstance(sampler, DummySampler), f"Expected DummySampler, got {type(sampler)}"
    adapter = sampler.scheduler
    assert isinstance(adapter, DummyGrainAdapter), f"Expected DummyGrainAdapter, got {type(adapter)}"
    loader = adapter.loader
    assert isinstance(loader.data_source, DummyDataSource), f"Expected DummyDataSource, got {type(loader.data_source)}"
    # Check that file_list was passed correctly
    assert loader.data_source.kwargs["dataset_path"] == ["file1.mat"], f"Expected dataset_path ['file1.mat'], got {loader.data_source.kwargs['dataset_path']}"


def test_make_grain_episodic(patch_grain):
    """Test standard Grain path instantiation for episodic data.

    When `episode_length > 0` is specified, the factory should insert an 
    `EpisodicDataSource` into the stack and use the `GrainEpisodicAdapter`.
    """
    cfg = {
        "scheduler_class": ".npy",
        "batch_size": 4,
        "flow_fields_per_batch": 1,
        "batches_per_flow_batch": 1,
        "episode_length": 10,
    }

    sampler = make(cfg, use_grain_scheduler=True)

    adapter = sampler.scheduler
    # Should wrap in EpisodicDataSource then Adapter
    loader = adapter.loader
    assert isinstance(loader.data_source, DummyEpisodicDataSource), f"Expected DummyEpisodicDataSource, got {type(loader.data_source)}"
    assert loader.data_source.episode_length == 10, f"Expected episode_length 10, got {loader.data_source.episode_length}"
    assert isinstance(loader.data_source.source, DummyDataSource), f"Expected underlying source to be DummyDataSource, got {type(loader.data_source.source)}"


def test_make_grain_include_images(patch_grain):
    """Test standard Grain path instantiation when images are requested.

    Verifies that `include_images=True` is propagated to the data source 
    and that a `RealImageSampler` (mocked) is created.
    """
    cfg = {
        "scheduler_class": ".mat",
        "batch_size": 4,
        "flow_fields_per_batch": 1,
        "include_images": True,
        "image_shape": (128, 128),
    }

    sampler = make(cfg, use_grain_scheduler=True)

    # RealImageSampler logic
    assert isinstance(sampler, DummySampler), f"Expected DummySampler, got {type(sampler)}"
    # Check DataSource got include_images=True
    loader = sampler.scheduler.loader
    ds = loader.data_source
    assert ds.kwargs["include_images"] is True, "DataSource should have include_images=True"
    assert ds.kwargs["output_shape"] == (128, 128), f"Expected output_shape (128, 128), got {ds.kwargs['output_shape']}"


def test_make_grain_invalid_datasource(patch_grain):
    """Test that an unsupported file extension raises a ValueError.

    This ensures the factory correctly identifies and rejects file types 
    it doesn't have a registered `DataSource` for.
    """
    cfg = {
        "scheduler_class": ".invalid",
        "batch_size": 4,
    }
    with pytest.raises(ValueError, match="DataSource class .invalid not found"):
        make(cfg, use_grain_scheduler=True)


def test_make_grain_padding(patch_grain):
    """Test that Grain padding is correctly configured via `drop_remainder`.

    When `loop=False` and the dataset size isn't a multiple of the batch 
    size, Grain should be configured to NOT drop the remainder by default 
    (to ensure all data is seen).
    """
    cfg = {
        "scheduler_class": ".mat",
        "batch_size": 3,
        "flow_fields_per_batch": 1,
        "batches_per_flow_batch": 1,
        "include_images": False,
        "loop": False,
        "randomize": False,
    }

    make(cfg, use_grain_scheduler=True)

    # Verify grain.Batch was called with drop_remainder=False
    expected_batch_size = 3
    make_mod.grain.Batch.assert_called_with(
        batch_size=expected_batch_size, drop_remainder=False
    )


def test_make_grain_worker_count(patch_grain, monkeypatch):
    """Test the configuration of multi-process workers in Grain.

    Verifies that `worker_count` is passed to the `DataLoader` and that 
    a warning is issued when using multiple workers with non-episodic 
    data (due to potential serialization issues).
    """
    from unittest.mock import MagicMock

    mock_logger = MagicMock()
    monkeypatch.setattr(make_mod, "logger", mock_logger)

    cfg = {
        "scheduler_class": ".mat",
        "batch_size": 4,
        "flow_fields_per_batch": 1,
        "batches_per_flow_batch": 1,
        "worker_count": 4,
    }

    sampler = make(cfg, use_grain_scheduler=True)

    loader = sampler.scheduler.loader
    assert loader.worker_count == 4, f"Expected worker_count 4, got {loader.worker_count}"
    # Verify warning was logged
    mock_logger.warning.assert_called()
    args, _ = mock_logger.warning.call_args
    assert "This enables multiprocessing in Grain" in args[0], f"Expected warning message to mention multiprocessing, got: {args[0]}"


def test_make_grain_threading(patch_grain):
    """Test passing `num_threads` and `buffer_size` via the configuration.

    These parameters should be correctly encapsulated into Grain's 
    `ReadOptions` and passed to the `DataLoader`.
    """
    cfg = {
        "scheduler_class": ".mat",
        "batch_size": 4,
        "flow_fields_per_batch": 1,
        "batches_per_flow_batch": 1,
        "num_threads": 8,
        "buffer_size": 1000,
    }

    sampler = make(cfg, use_grain_scheduler=True)

    loader = sampler.scheduler.loader
    # Check if correct ReadOptions object was created and passed
    assert hasattr(loader.read_options, "num_threads"), "ReadOptions should have num_threads attribute"
    assert loader.read_options.num_threads == 8, f"Expected num_threads 8, got {loader.read_options.num_threads}"
    # Check buffer size (passed to kwargs of ReadOptions mock)
    assert loader.read_options.kwargs["prefetch_buffer_size"] == 1000, f"Expected prefetch_buffer_size 1000, got {loader.read_options.kwargs['prefetch_buffer_size']}"


def test_make_grain_episodic_worker_error(patch_grain):
    """Test that multiple workers are rejected for episodic data.

    Using `worker_count > 0` with episodic data is prohibited because 
    it can break the strictly sequential order required for episodes.
    """
    cfg = {
        "scheduler_class": ".mat",
        "batch_size": 4,
        "flow_fields_per_batch": 1,
        "episode_length": 5,
        "worker_count": 1,
    }
    with pytest.raises(
        ValueError, match="worker_count must be 0 when using episodic data"
    ):
        make(cfg, use_grain_scheduler=True)


def test_get_data_source_class_invalid():
    """Directly test get_data_source_class with invalid extension."""
    from synthpix.make import get_data_source_class

    with pytest.raises(ValueError, match="DataSource class .invalid not found"):
        get_data_source_class(".invalid")


def test_make_config_not_dict(monkeypatch, tmp_path):
    """Test that make raises TypeError if loaded config is not a dict."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("dummy")
    monkeypatch.setattr(make_mod, "load_configuration", lambda _: "not a dict")
    # We need to use a string to trigger load_configuration
    with pytest.raises(TypeError, match="config must be a dictionary"):
        make(str(config_file))
