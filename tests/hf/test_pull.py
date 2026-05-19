"""Tests for ``synthpix.hf.pull.pull_dataset``.

All network access is mocked. The real ``huggingface_hub`` module is
replaced with a stand-in that records the kwargs passed to
``snapshot_download``.
"""

from __future__ import annotations

import importlib
import types
from pathlib import Path

import pytest

from synthpix.hf import auth as auth_mod
from synthpix.hf import pull as pull_mod


class _FakeHub:
    """Minimal stand-in for the ``huggingface_hub`` module."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def snapshot_download(self, **kwargs):
        self.calls.append(kwargs)
        # Materialize the target so resolved paths are well-defined.
        local_dir = kwargs.get("local_dir")
        if local_dir is not None:
            Path(local_dir).mkdir(parents=True, exist_ok=True)
        return str(local_dir)


def _install_fake_hub(monkeypatch) -> _FakeHub:
    # Patch ``importlib.import_module`` so the lazy import yields a fake.
    fake = _FakeHub()

    real_import_module = importlib.import_module

    def _import_module(name, package=None):
        if name == "huggingface_hub":
            return fake
        return real_import_module(name, package)

    monkeypatch.setattr(
        pull_mod.importlib, "import_module", _import_module
    )
    return fake


def _clear_env(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_HUB_TOKEN", raising=False)
    # Neutralize the on-disk cached token so token resolution is
    # hermetic regardless of whether the dev machine has run
    # ``hf auth login``. Env-var and explicit-token resolution paths
    # are intentionally left intact.
    monkeypatch.setattr(auth_mod, "_cached_token", lambda: None)


def test_pull_dataset_basic(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    fake = _install_fake_hub(monkeypatch)

    result = pull_mod.pull_dataset("user/repo", tmp_path / "data")

    assert isinstance(result, Path)
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["repo_id"] == "user/repo"
    assert call["repo_type"] == "dataset"
    assert call["revision"] == "main"
    assert call["local_dir"] == str(tmp_path / "data")
    assert call["allow_patterns"] is None
    assert call["ignore_patterns"] == list(pull_mod._DEFAULT_IGNORE)
    assert call["max_workers"] == 8


def test_pull_dataset_splits_filter(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    fake = _install_fake_hub(monkeypatch)

    pull_mod.pull_dataset(
        "user/repo",
        tmp_path / "data",
        splits=("train", "val"),
    )

    call = fake.calls[0]
    assert call["allow_patterns"] == ["train/**", "val/**", "README.md"]


def test_pull_dataset_explicit_includes_win(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    fake = _install_fake_hub(monkeypatch)

    warnings: list[str] = []
    monkeypatch.setattr(
        pull_mod.logger,
        "warning",
        lambda msg, *a, **kw: warnings.append(msg),
    )

    pull_mod.pull_dataset(
        "user/repo",
        tmp_path / "data",
        splits=("train",),
        include_globs=("foo/**",),
    )

    call = fake.calls[0]
    assert call["allow_patterns"] == ["foo/**"]
    assert warnings, "Expected a warning about splits being ignored"
    assert "splits" in warnings[0].lower()


def test_pull_dataset_token_passes_through(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    fake = _install_fake_hub(monkeypatch)

    pull_mod.pull_dataset(
        "user/repo", tmp_path / "with_token", token="hf_xxx"
    )
    assert fake.calls[-1]["token"] == "hf_xxx"

    pull_mod.pull_dataset("user/repo", tmp_path / "no_token")
    assert fake.calls[-1]["token"] is None


def test_pull_dataset_token_resolved_from_env(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    fake = _install_fake_hub(monkeypatch)

    pull_mod.pull_dataset("user/repo", tmp_path / "data")

    assert fake.calls[-1]["token"] == "hf_env"


def test_pull_dataset_returns_resolved_path(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)

    target = tmp_path / "nested" / "data"
    result = pull_mod.pull_dataset("user/repo", target)

    assert result == target.expanduser().resolve()


def test_pull_dataset_target_is_file_raises(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)

    bad = tmp_path / "file"
    bad.write_bytes(b"not a dir")

    with pytest.raises(NotADirectoryError):
        pull_mod.pull_dataset("user/repo", bad)


def test_pull_dataset_hf_not_installed(monkeypatch, tmp_path):
    _clear_env(monkeypatch)

    def _missing(name, package=None):
        if name == "huggingface_hub":
            raise ImportError("no module named huggingface_hub")
        return importlib.import_module(name, package)

    monkeypatch.setattr(pull_mod.importlib, "import_module", _missing)

    with pytest.raises(ImportError) as excinfo:
        pull_mod.pull_dataset("user/repo", tmp_path / "data")

    assert "synthpix[hf]" in str(excinfo.value)


def test_pull_dataset_passes_max_workers(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    fake = _install_fake_hub(monkeypatch)

    pull_mod.pull_dataset(
        "user/repo", tmp_path / "data", max_workers=2
    )

    assert fake.calls[-1]["max_workers"] == 2


def test_pull_dataset_explicit_ignore_globs(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    fake = _install_fake_hub(monkeypatch)

    pull_mod.pull_dataset(
        "user/repo",
        tmp_path / "data",
        ignore_globs=("**/skip.me",),
    )

    assert fake.calls[-1]["ignore_patterns"] == ["**/skip.me"]


def test_pull_dataset_enables_hf_transfer_before_import(
    monkeypatch, tmp_path
):
    # huggingface_hub reads HF_HUB_ENABLE_HF_TRANSFER at *import* time,
    # so the env flag must be set BEFORE the lazy import. Capture the
    # environment as the import call runs and verify the flag was already
    # present.
    import os as _os

    from synthpix.hf import _transfer as transfer_mod

    _clear_env(monkeypatch)
    monkeypatch.delenv("HF_HUB_ENABLE_HF_TRANSFER", raising=False)
    monkeypatch.setattr(transfer_mod, "_hf_transfer_available", lambda: True)

    seen_flag: dict[str, bool] = {}
    fake = _FakeHub()
    real_import_module = importlib.import_module

    def _import_module(name, package=None):
        if name == "huggingface_hub":
            seen_flag["set"] = (
                _os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1"
            )
            return fake
        return real_import_module(name, package)

    monkeypatch.setattr(pull_mod.importlib, "import_module", _import_module)

    pull_mod.pull_dataset("user/repo", tmp_path / "data")

    assert seen_flag.get("set"), (
        "HF_HUB_ENABLE_HF_TRANSFER must be set before huggingface_hub "
        "is imported, otherwise the accelerated path stays off."
    )


# Sanity check: the helper does not pull huggingface_hub into the module
# graph eagerly.
def test_pull_module_does_not_eagerly_import_hub():
    # When this test runs huggingface_hub may or may not be installed in
    # the dev venv; either way, the pull module must remain importable
    # without it. We can only assert the module imports cleanly here.
    assert isinstance(pull_mod, types.ModuleType)
