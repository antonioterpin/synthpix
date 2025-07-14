from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# import the factory under test
make_mod = importlib.import_module("synthpix.make")
make = make_mod.make


class DummyScheduler:
    """Lightweight stand‑in for real schedulers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def from_config(cls, cfg: dict) -> "DummyScheduler":
        return cls(cfg)


class DummyPrefetchScheduler(DummyScheduler):
    pass


class DummyEpisodicScheduler(DummyScheduler):
    pass


class DummySampler:
    """Captures scheduler and batch size so tests can inspect them."""

    def __init__(
        self, scheduler: DummyScheduler, batch_size: int, *args: Any, **kwargs: Any
    ):
        self.scheduler = scheduler
        self.batch_size = batch_size

    @classmethod
    def from_config(
        cls, scheduler: DummyScheduler, *args: Any, **kwargs: Any
    ) -> "DummySampler":
        return cls(scheduler=scheduler, batch_size=kwargs["config"]["batch_size"])


def _patch_common(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace heavy SynthPix classes with the test doubles defined above."""

    # samplers
    monkeypatch.setattr(make_mod, "RealImageSampler", DummySampler, raising=True)
    monkeypatch.setattr(make_mod, "SyntheticImageSampler", DummySampler, raising=True)

    # schedulers and wrappers
    monkeypatch.setattr(make_mod, "MATFlowFieldScheduler", DummyScheduler, raising=True)
    monkeypatch.setattr(
        make_mod, "NumpyFlowFieldScheduler", DummyScheduler, raising=True
    )
    monkeypatch.setattr(
        make_mod, "PrefetchingFlowFieldScheduler", DummyPrefetchScheduler, raising=True
    )
    monkeypatch.setattr(
        make_mod, "EpisodicFlowFieldScheduler", DummyEpisodicScheduler, raising=True
    )

    # mapping used inside make.py
    monkeypatch.setitem(make_mod.SCHEDULERS, ".mat", DummyScheduler)
    monkeypatch.setitem(make_mod.SCHEDULERS, ".npy", DummyScheduler)

    # other heavy bits
    monkeypatch.setattr(
        make_mod, "generate_images_from_flow", lambda *_a, **_k: None, raising=True
    )
    monkeypatch.setattr(
        make_mod,
        "logger",
        SimpleNamespace(info=lambda *_: None, warning=lambda *_: None),
    )

    return SimpleNamespace(
        DummyScheduler=DummyScheduler,
        DummyPrefetchScheduler=DummyPrefetchScheduler,
        DummyEpisodicScheduler=DummyEpisodicScheduler,
        DummySampler=DummySampler,
    )


@pytest.mark.parametrize("bad_config", [123, 3.14, ["list"], ("tuple",), None])
def test_make_rejects_non_str_dict(monkeypatch: pytest.MonkeyPatch, bad_config):
    _patch_common(monkeypatch)
    with pytest.raises(TypeError):
        make(bad_config)  # type: ignore[arg-type]


def test_make_rejects_non_yaml_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_common(monkeypatch)
    bad_path = tmp_path / "config.txt"
    bad_path.write_text("scheduler_class: '.npy'")
    with pytest.raises(ValueError, match="must point to a .yaml file"):
        make(str(bad_path))


def test_make_directory_path_not_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_common(monkeypatch)
    cfg_dir = tmp_path / "cfg_dir"
    cfg_dir.mkdir()
    fake_yaml_dir = cfg_dir.with_suffix(".yaml")
    fake_yaml_dir.mkdir()
    with pytest.raises(ValueError, match="is not a file"):
        make(str(fake_yaml_dir))


def test_make_file_not_found(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_common(monkeypatch)
    with pytest.raises(FileNotFoundError):
        make(str(tmp_path / "missing.yaml"))


def test_make_dict_missing_scheduler_class(monkeypatch: pytest.MonkeyPatch):
    _patch_common(monkeypatch)
    with pytest.raises(ValueError, match="must contain 'scheduler_class'"):
        make({})  # type: ignore[arg-type]


def test_make_invalid_images_from_file_type(monkeypatch: pytest.MonkeyPatch):
    _patch_common(monkeypatch)
    cfg = {"scheduler_class": ".npy", "batch_size": 2, "flow_fields_per_batch": 4}
    with pytest.raises(TypeError, match="images_from_file must be a boolean"):
        make(cfg, images_from_file="yes")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_value", [-1, -5])
def test_make_invalid_buffer_size(monkeypatch: pytest.MonkeyPatch, bad_value: int):
    _patch_common(monkeypatch)
    cfg = {"scheduler_class": ".npy", "batch_size": 2, "flow_fields_per_batch": 4}
    with pytest.raises(ValueError, match="buffer_size must be a non-negative integer"):
        make(cfg, buffer_size=bad_value)


@pytest.mark.parametrize("bad_value", [-1, -10])
def test_make_invalid_episode_length(monkeypatch: pytest.MonkeyPatch, bad_value: int):
    _patch_common(monkeypatch)
    cfg = {"scheduler_class": ".mat", "batch_size": 2}
    with pytest.raises(
        ValueError, match="episode_length must be a non-negative integer"
    ):
        make(cfg, images_from_file=True, episode_length=bad_value)


@pytest.mark.parametrize(
    "cfg, images_from_file",
    [
        (
            {"scheduler_class": ".npy", "batch_size": 2, "flow_fields_per_batch": 4},
            True,
        ),
        ({"scheduler_class": ".mat", "batch_size": 2}, False),
    ],
)
def test_make_images_from_file_scheduler_mismatch(
    monkeypatch: pytest.MonkeyPatch, cfg: dict, images_from_file: bool
):
    helpers = _patch_common(monkeypatch)
    if images_from_file:
        with pytest.raises(ValueError):
            make(cfg, images_from_file=True)
    else:
        sampler = make(cfg)
        assert isinstance(sampler, helpers.DummySampler)


def test_make_yaml_path_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    helpers = _patch_common(monkeypatch)

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("# dummy")

    cfg_dict = {
        "scheduler_class": ".npy",
        "batch_size": 4,
        "flow_fields_per_batch": 8,
        "batches_per_flow_batch": 1,
    }
    monkeypatch.setattr(
        make_mod, "load_configuration", lambda _p: cfg_dict, raising=True
    )

    sampler = make(str(yaml_path))
    assert isinstance(sampler, helpers.DummySampler)
    assert isinstance(sampler.scheduler, helpers.DummyScheduler)


def test_make_yaml_loader_returns_bad_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _patch_common(monkeypatch)

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("# dummy")
    monkeypatch.setattr(
        make_mod, "load_configuration", lambda _p: [1, 2, 3], raising=True
    )

    with pytest.raises(TypeError, match="dataset_config must be a dictionary"):
        make(str(yaml_path))


def test_make_real_sampler_with_episode(monkeypatch: pytest.MonkeyPatch):
    helpers = _patch_common(monkeypatch)

    cfg = {"scheduler_class": ".mat", "batch_size": 4}
    sampler = make(cfg, images_from_file=True, episode_length=5)

    assert isinstance(sampler, helpers.DummySampler)
    assert isinstance(sampler.scheduler, helpers.DummyEpisodicScheduler)
    assert isinstance(sampler.scheduler.args[0], helpers.DummyScheduler)


def test_make_synthetic_sampler_with_prefetch(monkeypatch: pytest.MonkeyPatch):
    helpers = _patch_common(monkeypatch)

    cfg = {
        "scheduler_class": ".npy",
        "batch_size": 4,
        "flow_fields_per_batch": 8,
        "batches_per_flow_batch": 1,
    }
    sampler = make(cfg, buffer_size=3)

    assert isinstance(sampler, helpers.DummySampler)
    assert isinstance(sampler.scheduler, helpers.DummyPrefetchScheduler)
    assert sampler.scheduler.kwargs["buffer_size"] == 3
    assert isinstance(sampler.scheduler.kwargs.get("scheduler"), helpers.DummyScheduler)


def test_make_synthetic_sampler_episode_no_prefetch(monkeypatch: pytest.MonkeyPatch):
    helpers = _patch_common(monkeypatch)

    cfg = {
        "scheduler_class": ".npy",
        "batch_size": 4,
        "flow_fields_per_batch": 8,
        "batches_per_flow_batch": 2,
    }
    sampler = make(cfg, episode_length=3)

    assert isinstance(sampler, helpers.DummySampler)
    assert isinstance(sampler.scheduler, helpers.DummyEpisodicScheduler)
    assert isinstance(sampler.scheduler.args[0], helpers.DummyScheduler)


def test_make_real_sampler_prefetch(monkeypatch: pytest.MonkeyPatch):
    helpers = _patch_common(monkeypatch)

    cfg = {"scheduler_class": ".mat", "batch_size": 4}
    sampler = make(cfg, images_from_file=True, buffer_size=2)

    assert isinstance(sampler, helpers.DummySampler)
    assert isinstance(sampler.scheduler, helpers.DummyPrefetchScheduler)
    assert sampler.scheduler.kwargs["buffer_size"] == 2
    assert isinstance(sampler.scheduler.args[0], helpers.DummyScheduler)
    # include_images should be set to True automatically
    assert cfg["include_images"] is True


def test_make_scheduler_class_not_in_mapping(monkeypatch: pytest.MonkeyPatch):
    _patch_common(monkeypatch)
    cfg = {"scheduler_class": ".foo", "batch_size": 2, "flow_fields_per_batch": 4}
    with pytest.raises(ValueError):
        make(cfg)
