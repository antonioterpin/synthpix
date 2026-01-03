import importlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from synthpix.data_sources import FileDataSource

make_mod = importlib.import_module("synthpix.make")
make = make_mod.make


class DummyScheduler:
    """Lightweight stand-in for real schedulers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def from_config(cls, cfg: dict) -> "DummyScheduler":
        # Record config passed by make()
        return cls(cfg)


class DummyPrefetchScheduler(DummyScheduler):
    pass


class DummyEpisodicScheduler(DummyScheduler):
    pass


class DummySampler:
    """Captures scheduler and batch size for inspection."""

    def __init__(
        self,
        scheduler: DummyScheduler,
        batch_size: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.kwargs = kwargs

    @classmethod
    def from_config(
        cls, scheduler: DummyScheduler, *args: Any, **kwargs: Any
    ) -> "DummySampler":
        return cls(
            scheduler=scheduler, batch_size=kwargs["config"]["batch_size"]
        )


def _patch_common(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Patch SynthPix symbols with dummy stand-ins for isolation."""
    monkeypatch.setattr(make_mod, "RealImageSampler", DummySampler)
    monkeypatch.setattr(make_mod, "SyntheticImageSampler", DummySampler)
    monkeypatch.setattr(make_mod, "MATFlowFieldScheduler", DummyScheduler)
    monkeypatch.setattr(make_mod, "NumpyFlowFieldScheduler", DummyScheduler)
    monkeypatch.setattr(
        make_mod, "PrefetchingFlowFieldScheduler", DummyPrefetchScheduler
    )
    monkeypatch.setattr(
        make_mod, "EpisodicFlowFieldScheduler", DummyEpisodicScheduler
    )
    monkeypatch.setitem(make_mod.SCHEDULERS, ".mat", DummyScheduler)
    monkeypatch.setitem(make_mod.SCHEDULERS, ".npy", DummyScheduler)
    monkeypatch.setattr(
        make_mod,
        "logger",
        SimpleNamespace(info=lambda *_: None, warning=lambda *_: None),
    )
    monkeypatch.setattr(
        make_mod, "load_configuration", lambda p: {}, raising=False
    )
    return SimpleNamespace(
        DummyScheduler=DummyScheduler,
        DummyPrefetchScheduler=DummyPrefetchScheduler,
        DummyEpisodicScheduler=DummyEpisodicScheduler,
        DummySampler=DummySampler,
    )


# ---------------------- #
# Validation and Errors  #
# ---------------------- #


@pytest.mark.parametrize("bad", [123, 3.14, ["a"], ("t",), None])
def test_rejects_non_str_or_dict(monkeypatch, bad):
    """Test that make() raises TypeError if config is not a string or a dictionary.

    The config argument must be either a path to a YAML file (string) or a 
    pre-loaded configuration dictionary.
    """
    _patch_common(monkeypatch)
    with pytest.raises(
        TypeError, match="config must be a string or a dictionary"
    ):
        make(bad)


def test_rejects_non_yaml_file(monkeypatch, tmp_path):
    """Test that make() raises ValueError if the config path does not end in .yaml."""
    _patch_common(monkeypatch)
    path = tmp_path / "config.txt"
    path.write_text("scheduler_class: '.mat'")
    with pytest.raises(ValueError, match="must point to a .yaml file"):
        make(str(path))


def test_rejects_directory_instead_of_file(monkeypatch, tmp_path):
    """Test that make() raises ValueError if a directory path is provided instead of a file."""
    _patch_common(monkeypatch)
    dirpath = tmp_path / "foo.yaml"
    dirpath.mkdir()
    with pytest.raises(ValueError, match="is not a file"):
        make(str(dirpath))


def test_raises_file_not_found(monkeypatch, tmp_path):
    """Test that make() raises FileNotFoundError if the specified YAML file does not exist."""
    _patch_common(monkeypatch)
    with pytest.raises(FileNotFoundError):
        make(str(tmp_path / "missing.yaml"))


def test_missing_scheduler_class(monkeypatch):
    _patch_common(monkeypatch)
    with pytest.raises(ValueError, match="must contain 'scheduler_class'"):
        make({})


def test_scheduler_class_not_in_mapping(monkeypatch):
    _patch_common(monkeypatch)
    cfg = {
        "scheduler_class": ".foo",
        "batch_size": 2,
        "flow_fields_per_batch": 1,
    }
    with pytest.raises(ValueError, match="not found"):
        make(cfg)


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_nonpositive_batch(monkeypatch, bad):
    _patch_common(monkeypatch)
    cfg = {
        "scheduler_class": ".mat",
        "batch_size": bad,
        "flow_fields_per_batch": 1,
    }
    with pytest.raises(
        ValueError, match="batch_size must be a positive integer"
    ):
        make(cfg)


@pytest.mark.parametrize("bad", [-3, "x"])
def test_rejects_invalid_buffer_size(monkeypatch, bad):
    _patch_common(monkeypatch)
    cfg = {
        "scheduler_class": ".mat",
        "batch_size": 2,
        "flow_fields_per_batch": 1,
        "buffer_size": bad,
    }
    with pytest.raises((ValueError, TypeError), match="buffer_size"):
        make(cfg)


@pytest.mark.parametrize("bad", [-3, "x"])
def test_rejects_invalid_episode_length(monkeypatch, bad):
    _patch_common(monkeypatch)
    cfg = {
        "scheduler_class": ".mat",
        "batch_size": 2,
        "flow_fields_per_batch": 1,
        "episode_length": bad,
    }
    with pytest.raises((ValueError, TypeError), match="episode_length"):
        make(cfg)


def test_requires_flow_fields_per_batch(monkeypatch):
    _patch_common(monkeypatch)
    cfg = {"scheduler_class": ".npy", "batch_size": 2}
    with pytest.raises(ValueError, match="flow_fields_per_batch"):
        make(cfg)


def test_requires_batches_per_flow_batch_for_synthetic(monkeypatch):
    _patch_common(monkeypatch)
    cfg = {
        "scheduler_class": ".npy",
        "batch_size": 2,
        "flow_fields_per_batch": 1,
    }
    with pytest.raises(ValueError, match="batches_per_flow_batch"):
        make(cfg, use_grain_scheduler=False)


def test_include_images_must_be_bool(monkeypatch):
    _patch_common(monkeypatch)
    cfg = {
        "scheduler_class": ".mat",
        "batch_size": 2,
        "flow_fields_per_batch": 1,
        "include_images": "yes",
    }
    with pytest.raises(TypeError, match="include_images must be a boolean"):
        make(cfg)


def test_include_images_requires_mat(monkeypatch):
    _patch_common(monkeypatch)
    cfg = {
        "scheduler_class": ".npy",
        "batch_size": 2,
        "flow_fields_per_batch": 1,
        "include_images": True,
    }
    with pytest.raises(ValueError, match="not supported for file images"):
        make(cfg)


# ---------------------- #
# Functional paths       #
# ---------------------- #


def test_make_from_yaml_path_success(monkeypatch, tmp_path):
    helpers = _patch_common(monkeypatch)
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("scheduler_class: '.npy'")
    good_cfg = {
        "scheduler_class": ".npy",
        "batch_size": 4,
        "flow_fields_per_batch": 8,
        "batches_per_flow_batch": 1,
    }
    monkeypatch.setattr(make_mod, "load_configuration", lambda _: good_cfg)
    sampler = make(str(yaml_path), use_grain_scheduler=False)
    assert isinstance(sampler, helpers.DummySampler), f"Expected DummySampler, got {type(sampler)}"
    assert isinstance(sampler.scheduler, helpers.DummyScheduler), f"Expected DummyScheduler, got {type(sampler.scheduler)}"


def test_real_sampler_with_episode(monkeypatch):
    helpers = _patch_common(monkeypatch)
    cfg = {
        "scheduler_class": ".mat",
        "batch_size": 2,
        "flow_fields_per_batch": 1,
        "include_images": True,
        "episode_length": 3,
    }
    sampler = make(cfg, use_grain_scheduler=False)
    assert isinstance(sampler.scheduler, helpers.DummyEpisodicScheduler), f"Expected DummyEpisodicScheduler, got {type(sampler.scheduler)}"
    assert isinstance(
        sampler.scheduler.kwargs["scheduler"], helpers.DummyScheduler
    ), f"Expected underlying scheduler to be DummyScheduler, got {type(sampler.scheduler.kwargs['scheduler'])}"


def test_real_sampler_with_prefetch(monkeypatch):
    helpers = _patch_common(monkeypatch)
    cfg = {
        "scheduler_class": ".mat",
        "batch_size": 2,
        "flow_fields_per_batch": 1,
        "include_images": True,
        "buffer_size": 3,
    }
    sampler = make(cfg, use_grain_scheduler=False)
    assert isinstance(sampler.scheduler, helpers.DummyPrefetchScheduler), f"Expected DummyPrefetchScheduler, got {type(sampler.scheduler)}"
    assert isinstance(
        sampler.scheduler.kwargs["scheduler"], helpers.DummyScheduler
    ), f"Expected underlying scheduler to be DummyScheduler, got {type(sampler.scheduler.kwargs['scheduler'])}"


def test_synthetic_sampler_with_prefetch(monkeypatch):
    helpers = _patch_common(monkeypatch)
    cfg = {
        "scheduler_class": ".npy",
        "batch_size": 2,
        "flow_fields_per_batch": 1,
        "batches_per_flow_batch": 1,
        "buffer_size": 2,
    }
    sampler = make(cfg, use_grain_scheduler=False)
    assert isinstance(sampler.scheduler, helpers.DummyPrefetchScheduler), f"Expected DummyPrefetchScheduler, got {type(sampler.scheduler)}"
    assert isinstance(
        sampler.scheduler.kwargs["scheduler"], helpers.DummyScheduler
    ), f"Expected underlying scheduler to be DummyScheduler, got {type(sampler.scheduler.kwargs['scheduler'])}"


def test_synthetic_sampler_with_episode(monkeypatch):
    helpers = _patch_common(monkeypatch)
    cfg = {
        "scheduler_class": ".npy",
        "batch_size": 2,
        "flow_fields_per_batch": 1,
        "batches_per_flow_batch": 2,
        "episode_length": 5,
    }
    sampler = make(cfg, use_grain_scheduler=False)
    assert isinstance(sampler.scheduler, helpers.DummyEpisodicScheduler), f"Expected DummyEpisodicScheduler, got {type(sampler.scheduler)}"
    assert isinstance(
        sampler.scheduler.kwargs["scheduler"], helpers.DummyScheduler
    ), f"Expected underlying scheduler to be DummyScheduler, got {type(sampler.scheduler.kwargs['scheduler'])}"


# ---------------------- #
# Grain Scheduler Paths  #
# ---------------------- #


def test_make_grain_scheduler_call(monkeypatch):
    """Test that make() calls make_grain_scheduler correctly with non-episodic config."""
    helpers = _patch_common(monkeypatch)

    mock_grain = SimpleNamespace(
        IndexSampler=MagicMock(),
        DataLoader=MagicMock(),
        Batch=MagicMock(),
        NoSharding=MagicMock(),
        ReadOptions=MagicMock(),
    )
    monkeypatch.setattr(make_mod, "grain", mock_grain)
    
    mock_grain_adapter = MagicMock()
    monkeypatch.setattr(make_mod, "GrainSchedulerAdapter", mock_grain_adapter)
    
    adapter_instance = MagicMock()
    mock_grain_adapter.return_value = adapter_instance
    
    class MockDataSource(FileDataSource):
        def __init__(self, **kwargs):
            pass
        def __len__(self):
            return 10
        def load_file(self, f): return {}
        @property
        def file_list(self): return []
        @property
        def include_images(self): return False

    mock_data_source_cls = MagicMock(return_value=MockDataSource())
    monkeypatch.setattr(make_mod, "get_data_source_class", lambda name: mock_data_source_cls)
    
    mock_episodic_ds_cls = MagicMock()
    monkeypatch.setattr(make_mod, "EpisodicDataSource", mock_episodic_ds_cls)

    cfg = {
        "scheduler_class": ".mat",
        "batch_size": 2,
        "flow_fields_per_batch": 1,
        "include_images": True,
        "episode_length": 0,
        "worker_count": 0,
    }

    sampler = make(cfg, use_grain_scheduler=True)
    
    mock_grain_adapter.assert_called_once()
    assert sampler.scheduler == adapter_instance, "Sampler scheduler should be the adapter instance"


def test_make_grain_scheduler_episodic_error(monkeypatch):
    """Test ValueError when worker_count > 0 with episodic data."""
    _patch_common(monkeypatch)

    mock_grain = SimpleNamespace(
        IndexSampler=MagicMock(),
        DataLoader=MagicMock(),
        Batch=MagicMock(),
        NoSharding=MagicMock(),
        ReadOptions=MagicMock(),
    )
    monkeypatch.setattr(make_mod, "grain", mock_grain)
    
    mock_episodic_ds_instance = MagicMock()
    mock_episodic_ds_instance.__len__.return_value = 10
    mock_episodic_ds_cls = MagicMock(return_value=mock_episodic_ds_instance)
    monkeypatch.setattr(make_mod, "EpisodicDataSource", mock_episodic_ds_cls)

    monkeypatch.setattr(make_mod, "get_data_source_class", MagicMock())
    
    cfg = {
        "scheduler_class": ".mat",
        "batch_size": 2,
        "flow_fields_per_batch": 1,
        "episode_length": 5,
        "worker_count": 2,
    }
    
    with pytest.raises(ValueError, match="worker_count must be 0"):
        make(cfg, use_grain_scheduler=True)


def test_make_grain_scheduler_episodic_success(monkeypatch):
    """Test successful creation of episodic Grain scheduler."""
    helpers = _patch_common(monkeypatch)
    mock_grain = SimpleNamespace(
        IndexSampler=MagicMock(),
        DataLoader=MagicMock(),
        Batch=MagicMock(),
        NoSharding=MagicMock(),
        ReadOptions=MagicMock(),
    )
    monkeypatch.setattr(make_mod, "grain", mock_grain)
    monkeypatch.setattr(make_mod, "EpisodicDataSource", MagicMock())
    monkeypatch.setattr(make_mod, "get_data_source_class", lambda _: MagicMock())
    monkeypatch.setattr(make_mod, "GrainEpisodicAdapter", MagicMock(return_value="epi_adapter"))

    cfg = {
        "scheduler_class": ".mat",
        "batch_size": 2,
        "flow_fields_per_batch": 1,
        "episode_length": 5,
        "worker_count": 0,
        "batches_per_flow_batch": 1,
    }
    sampler = make(cfg, use_grain_scheduler=True)
    assert sampler.scheduler == "epi_adapter", f"Expected sampler scheduler to be 'epi_adapter', got {sampler.scheduler}"


def test_make_grain_scheduler_warning_multi_worker(monkeypatch):
    """Test warning for multi-worker non-episodic Grain scheduler."""
    _patch_common(monkeypatch)
    mock_grain = SimpleNamespace(
        IndexSampler=MagicMock(),
        DataLoader=MagicMock(),
        Batch=MagicMock(),
        NoSharding=MagicMock(),
        ReadOptions=MagicMock(),
    )
    monkeypatch.setattr(make_mod, "grain", mock_grain)
    monkeypatch.setattr(make_mod, "get_data_source_class", lambda _: MagicMock())
    # Mock adapter to avoid isinstance(loader, grain.DataLoader) check
    monkeypatch.setattr(make_mod, "GrainSchedulerAdapter", MagicMock())
    
    # Mock logger to verify warning
    mock_logger = MagicMock()
    monkeypatch.setattr(make_mod, "logger", mock_logger)

    cfg = {
        "scheduler_class": ".mat",
        "batch_size": 2,
        "flow_fields_per_batch": 1,
        "episode_length": 0,
        "worker_count": 2,
        "batches_per_flow_batch": 1,
    }
    make(cfg, use_grain_scheduler=True)
    mock_logger.warning.assert_called()


def test_make_grain_scheduler_with_file_list(monkeypatch):
    """Test Grain scheduler with explicit file_list."""
    _patch_common(monkeypatch)
    mock_grain = SimpleNamespace(
        IndexSampler=MagicMock(),
        DataLoader=MagicMock(),
        Batch=MagicMock(),
        NoSharding=MagicMock(),
        ReadOptions=MagicMock(),
    )
    monkeypatch.setattr(make_mod, "grain", mock_grain)
    monkeypatch.setattr(make_mod, "GrainSchedulerAdapter", MagicMock())
    mock_ds_cls = MagicMock()
    monkeypatch.setattr(make_mod, "get_data_source_class", lambda _: mock_ds_cls)

    cfg = {
        "scheduler_class": ".mat",
        "batch_size": 2,
        "flow_fields_per_batch": 1,
        "file_list": ["f1.mat"],
        "batches_per_flow_batch": 1,
    }
    make(cfg, use_grain_scheduler=True)
    # Check that data source was instantiated with dataset_path
    args, kwargs = mock_ds_cls.call_args
    assert kwargs["dataset_path"] == ["f1.mat"], f"Expected dataset_path ['f1.mat'], got {kwargs['dataset_path']}"


def test_get_base_scheduler_invalid():
    """Test get_base_scheduler raises ValueError for invalid extension."""
    with pytest.raises(ValueError, match="not found"):
        make_mod.get_base_scheduler(".invalid")


def test_get_data_source_class_invalid():
    """Test get_data_source_class raises ValueError for invalid extension."""
    with pytest.raises(ValueError, match="not found"):
        make_mod.get_data_source_class(".invalid")


def test_get_data_source_class_valid():
    """Test get_data_source_class returns correct class for valid extension."""
    from synthpix.data_sources import MATDataSource
    assert make_mod.get_data_source_class(".mat") == MATDataSource, "get_data_source_class('.mat') should return MATDataSource"


def test_make_rejects_non_dict_config_loaded(monkeypatch, tmp_path):
    """Test that make() raises TypeError if loaded config is not a dict."""
    _patch_common(monkeypatch)
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("foo")
    # Mock load_configuration to return something else
    monkeypatch.setattr(make_mod, "load_configuration", lambda _: "not a dict")
    
    with pytest.raises(TypeError, match="must be a dictionary"):
        make(str(yaml_path))

