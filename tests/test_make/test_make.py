import importlib
from types import SimpleNamespace
from typing import Any

import pytest

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
    _patch_common(monkeypatch)
    with pytest.raises(
        TypeError, match="config must be a string or a dictionary"
    ):
        make(bad)


def test_rejects_non_yaml_file(monkeypatch, tmp_path):
    _patch_common(monkeypatch)
    path = tmp_path / "config.txt"
    path.write_text("scheduler_class: '.mat'")
    with pytest.raises(ValueError, match="must point to a .yaml file"):
        make(str(path))


def test_rejects_directory_instead_of_file(monkeypatch, tmp_path):
    _patch_common(monkeypatch)
    dirpath = tmp_path / "foo.yaml"
    dirpath.mkdir()
    with pytest.raises(ValueError, match="is not a file"):
        make(str(dirpath))


def test_raises_file_not_found(monkeypatch, tmp_path):
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
        make(cfg)


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
    sampler = make(str(yaml_path))
    assert isinstance(sampler, helpers.DummySampler)
    assert isinstance(sampler.scheduler, helpers.DummyScheduler)


def test_real_sampler_with_episode(monkeypatch):
    helpers = _patch_common(monkeypatch)
    cfg = {
        "scheduler_class": ".mat",
        "batch_size": 2,
        "flow_fields_per_batch": 1,
        "include_images": True,
        "episode_length": 3,
    }
    sampler = make(cfg)
    assert isinstance(sampler.scheduler, helpers.DummyEpisodicScheduler)
    assert isinstance(
        sampler.scheduler.kwargs["scheduler"], helpers.DummyScheduler
    )


def test_real_sampler_with_prefetch(monkeypatch):
    helpers = _patch_common(monkeypatch)
    cfg = {
        "scheduler_class": ".mat",
        "batch_size": 2,
        "flow_fields_per_batch": 1,
        "include_images": True,
        "buffer_size": 3,
    }
    sampler = make(cfg)
    assert isinstance(sampler.scheduler, helpers.DummyPrefetchScheduler)
    assert isinstance(
        sampler.scheduler.kwargs["scheduler"], helpers.DummyScheduler
    )


def test_synthetic_sampler_with_prefetch(monkeypatch):
    helpers = _patch_common(monkeypatch)
    cfg = {
        "scheduler_class": ".npy",
        "batch_size": 2,
        "flow_fields_per_batch": 1,
        "batches_per_flow_batch": 1,
        "buffer_size": 2,
    }
    sampler = make(cfg)
    assert isinstance(sampler.scheduler, helpers.DummyPrefetchScheduler)
    assert isinstance(
        sampler.scheduler.kwargs["scheduler"], helpers.DummyScheduler
    )


def test_synthetic_sampler_with_episode(monkeypatch):
    helpers = _patch_common(monkeypatch)
    cfg = {
        "scheduler_class": ".npy",
        "batch_size": 2,
        "flow_fields_per_batch": 1,
        "batches_per_flow_batch": 2,
        "episode_length": 5,
    }
    sampler = make(cfg)
    assert isinstance(sampler.scheduler, helpers.DummyEpisodicScheduler)
    assert isinstance(
        sampler.scheduler.kwargs["scheduler"], helpers.DummyScheduler
    )
