
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
    def __init__(self, data_source, sampler, operations, worker_count=0, read_options=None):
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
    monkeypatch.setattr(make_mod, "load_configuration", lambda p: {}, raising=False)

def test_make_grain_basic(patch_grain, monkeypatch):
    """Test standard Grain path instantiation."""
    cfg = {
        "scheduler_class": ".mat",
        "batch_size": 4,
        "flow_fields_per_batch": 1,
        "include_images": False,
        "batches_per_flow_batch": 1,
        "file_list": ["file1.mat"],
    }
    
    sampler = make(cfg, use_grain_scheduler=True)
    
    assert isinstance(sampler, DummySampler)
    adapter = sampler.scheduler
    assert isinstance(adapter, DummyGrainAdapter)
    loader = adapter.loader
    assert isinstance(loader.data_source, DummyDataSource)
    # Check that file_list was passed correctly
    assert loader.data_source.kwargs["dataset_path"] == ["file1.mat"]

def test_make_grain_episodic(patch_grain):
    """Test Grain path with episode_length > 0."""
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
    assert isinstance(loader.data_source, DummyEpisodicDataSource)
    assert loader.data_source.episode_length == 10
    assert isinstance(loader.data_source.source, DummyDataSource)

def test_make_grain_include_images(patch_grain):
    """Test Grain path with include_images=True."""
    cfg = {
        "scheduler_class": ".mat",
        "batch_size": 4,
        "flow_fields_per_batch": 1,
        "include_images": True,
        "image_shape": (128, 128)
    }
    
    sampler = make(cfg, use_grain_scheduler=True)
    
    # RealImageSampler logic
    assert isinstance(sampler, DummySampler)
    # Check DataSource got include_images=True
    loader = sampler.scheduler.loader
    ds = loader.data_source
    assert ds.kwargs["include_images"] is True
    assert ds.kwargs["output_shape"] == (128, 128)

def test_make_grain_invalid_datasource(patch_grain):
    """Test checks for invalid data source extension."""
    cfg = {
        "scheduler_class": ".invalid",
        "batch_size": 4,
    }
    with pytest.raises(ValueError, match="DataSource class .invalid not found"):
        make(cfg, use_grain_scheduler=True)

def test_make_grain_padding(patch_grain):
    """Test behavior when dataset size is not divisible by batch size."""
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
    """Test passing worker_count and verify warning for non-episodic multi-worker."""
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
    assert loader.worker_count == 4
    # Verify warning was logged
    mock_logger.warning.assert_called()
    args, _ = mock_logger.warning.call_args
    assert "This enables multiprocessing in Grain" in args[0]

def test_make_grain_threading(patch_grain):
    """Test passing num_threads and buffer_size via config."""
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
    assert hasattr(loader.read_options, "num_threads")
    assert loader.read_options.num_threads == 8
    # Check buffer size (passed to kwargs of ReadOptions mock)
    assert loader.read_options.kwargs["prefetch_buffer_size"] == 1000

def test_make_grain_episodic_worker_error(patch_grain):
    """Test that episodic data raises error with worker_count > 0."""
    cfg = {
        "scheduler_class": ".mat",
        "batch_size": 4,
        "flow_fields_per_batch": 1,
        "episode_length": 5,
        "worker_count": 1,
    }
    with pytest.raises(ValueError, match="worker_count must be 0 when using episodic data"):
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
