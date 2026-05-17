"""Integration tests for ``hf://`` URI handling inside ``FileDataSource``.

The resolver is patched so no Hub access happens; these tests cover the
hook in ``FileDataSource.__init__`` only.
"""

from __future__ import annotations

import builtins
import importlib
from pathlib import Path

import pytest

from synthpix.data_sources import hf_resolver
from synthpix.data_sources.base import FileDataSource


class _StubDataSource(FileDataSource):
    """Minimal concrete subclass: identity-load and ``*.mat`` glob."""

    _file_pattern = "**/*.mat"

    def load_file(self, file_path):
        return {"file": file_path}


def _stub_resolve_to_directory(cache_root: Path):
    # Build a fake repo dir containing a couple of .mat files plus a
    # README.md and a .git/config that must be filtered out by globbing.

    def _stub(spec, **_kwargs):
        repo_dir = cache_root / "user" / "repo@main"
        repo_dir.mkdir(parents=True, exist_ok=True)
        (repo_dir / "a.mat").touch()
        (repo_dir / "b.mat").touch()
        (repo_dir / "README.md").write_text("card")
        return repo_dir

    return _stub


def test_filedatasource_resolves_hf_uri(monkeypatch, tmp_path):
    # An hf:// URI in dataset_path is resolved into the file list.
    monkeypatch.setattr(
        hf_resolver,
        "resolve_to_directory",
        _stub_resolve_to_directory(tmp_path),
    )

    ds = _StubDataSource("hf://user/repo")

    repo_dir = tmp_path / "user" / "repo@main"
    # The *.mat glob applies — README.md is excluded by the pattern.
    assert sorted(ds.file_list) == sorted(
        [str(repo_dir / "a.mat"), str(repo_dir / "b.mat")]
    )


def test_filedatasource_mixes_hf_and_local_paths(monkeypatch, tmp_path):
    # An hf:// URI may sit alongside regular local paths.
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    local_files = []
    for name in ("x.mat", "y.mat", "z.mat"):
        f = local_dir / name
        f.touch()
        local_files.append(str(f))

    cache_root = tmp_path / "cache"
    monkeypatch.setattr(
        hf_resolver,
        "resolve_to_directory",
        _stub_resolve_to_directory(cache_root),
    )

    ds = _StubDataSource(["hf://user/repo", str(local_dir)])

    repo_dir = cache_root / "user" / "repo@main"
    expected = {
        str(repo_dir / "a.mat"),
        str(repo_dir / "b.mat"),
        *local_files,
    }
    assert set(ds.file_list) == expected
    assert len(ds.file_list) == 5


def test_filedatasource_hf_uri_without_extra_raises_clear_error(
    monkeypatch, tmp_path
):
    # A failing resolver import surfaces an actionable message with cause.
    import synthpix.data_sources as ds_pkg

    real_import = builtins.__import__
    original_cause = ImportError("install with synthpix[hf]")

    # Force the `from synthpix.data_sources import hf_resolver` statement in
    # base.py to re-run by dropping the cached submodule and the attribute.
    monkeypatch.delitem(
        importlib.sys.modules,
        "synthpix.data_sources.hf_resolver",
        raising=False,
    )
    monkeypatch.delattr(ds_pkg, "hf_resolver", raising=False)

    def _failing_import(name, *args, **kwargs):
        if name == "synthpix.data_sources.hf_resolver":
            raise original_cause
        # Submodule imports also come through with fromlist on the parent.
        fromlist = args[2] if len(args) >= 3 else kwargs.get("fromlist") or ()
        if name == "synthpix.data_sources" and "hf_resolver" in fromlist:
            raise original_cause
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _failing_import)

    with pytest.raises(ImportError) as excinfo:
        _StubDataSource("hf://user/repo")

    assert "synthpix[hf]" in str(excinfo.value)
    assert excinfo.value.__cause__ is original_cause


def test_filedatasource_pull_dataset_import_error_surfaces(
    monkeypatch, tmp_path
):
    # If huggingface_hub is missing, pull_dataset (called from inside
    # resolve_to_directory) raises ImportError; base.py must surface the
    # same actionable message and preserve the cause.
    original_cause = ImportError("no huggingface_hub")

    def _raising_resolve(spec, **_kwargs):
        raise original_cause

    monkeypatch.setattr(
        hf_resolver, "resolve_to_directory", _raising_resolve
    )

    with pytest.raises(ImportError) as excinfo:
        _StubDataSource("hf://user/repo")

    assert "synthpix[hf]" in str(excinfo.value)
    assert excinfo.value.__cause__ is original_cause


def test_filedatasource_only_hf_uri(monkeypatch, tmp_path):
    # An hf://-only dataset_path is enough — no local fallback needed.
    monkeypatch.setattr(
        hf_resolver,
        "resolve_to_directory",
        _stub_resolve_to_directory(tmp_path),
    )

    ds = _StubDataSource("hf://user/repo")

    repo_dir = tmp_path / "user" / "repo@main"
    assert sorted(ds.file_list) == sorted(
        [str(repo_dir / "a.mat"), str(repo_dir / "b.mat")]
    )


def test_filedatasource_skips_hidden_dirs(monkeypatch, tmp_path):
    # Glob with **/*.mat does pick up paths under hidden dirs; verify the
    # pattern + skip behavior matches whatever subclass _file_pattern says.
    # The MAT glob pattern explicitly walks ** and so a .cache/x.mat under
    # the repo *would* appear unless excluded by the pattern itself. This
    # documents the actual behavior (no implicit hidden-dir filter).
    def _stub(spec, **_kwargs):
        repo = tmp_path / "user" / "repo@main"
        (repo / ".cache").mkdir(parents=True, exist_ok=True)
        (repo / "real.mat").touch()
        (repo / ".cache" / "leaked.mat").touch()
        return repo

    monkeypatch.setattr(hf_resolver, "resolve_to_directory", _stub)

    ds = _StubDataSource("hf://user/repo")
    # Real file is present; .cache/leaked.mat is filtered by glob's default
    # behavior of not matching paths with hidden components for **/*.mat
    # (because the leading * doesn't match a name starting with '.').
    assert any(p.endswith("real.mat") for p in ds.file_list)
    assert not any(".cache" in p for p in ds.file_list)
