"""Integration tests for ``hf://`` URI handling inside ``FileDataSource``.

The resolver is patched so no Hub access happens; these tests cover the
hook in ``FileDataSource.__init__`` only.
"""

from __future__ import annotations

import builtins
import importlib

import pytest

from synthpix.data_sources import hf_resolver
from synthpix.data_sources.base import FileDataSource


class _StubDataSource(FileDataSource):
    """Minimal concrete subclass: identity-load and ``*.mat`` glob."""

    _file_pattern = "**/*.mat"

    def load_file(self, file_path):
        return {"file": file_path}


def test_filedatasource_resolves_hf_uri(monkeypatch, tmp_path):
    # An hf:// URI in dataset_path is resolved into the file list.
    fake_files = [
        str(tmp_path / "user" / "repo@main" / "a.mat"),
        str(tmp_path / "user" / "repo@main" / "b.mat"),
    ]
    monkeypatch.setattr(
        hf_resolver, "resolve", lambda spec, **kwargs: list(fake_files)
    )

    ds = _StubDataSource("hf://user/repo")

    assert ds.file_list == fake_files


def test_filedatasource_mixes_hf_and_local_paths(monkeypatch, tmp_path):
    # An hf:// URI may sit alongside regular local paths.
    # Local fixture: a directory holding three .mat files.
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    local_files = []
    for name in ("x.mat", "y.mat", "z.mat"):
        f = local_dir / name
        f.touch()
        local_files.append(str(f))

    hf_files = [
        str(tmp_path / "cache" / "user" / "repo@main" / "a.mat"),
        str(tmp_path / "cache" / "user" / "repo@main" / "b.mat"),
    ]
    monkeypatch.setattr(
        hf_resolver, "resolve", lambda spec, **kwargs: list(hf_files)
    )

    ds = _StubDataSource(["hf://user/repo", str(local_dir)])

    assert set(ds.file_list) == set(hf_files) | set(local_files)
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


def test_filedatasource_only_hf_uri(monkeypatch, tmp_path):
    # An hf://-only dataset_path is enough — no local fallback needed.
    fake_files = [
        str(tmp_path / "user" / "repo@main" / "only.mat"),
    ]
    monkeypatch.setattr(
        hf_resolver, "resolve", lambda spec, **kwargs: list(fake_files)
    )

    ds = _StubDataSource("hf://user/repo")

    assert ds.file_list == fake_files
