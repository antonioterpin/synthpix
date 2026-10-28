"""Integration tests for ``hf://`` URI handling inside
``FileDataSource``.

The resolver is patched so no Hub access happens; these tests cover
the hook in ``FileDataSource.__init__`` only.
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from typing import Any

import pytest

from synthpix.data_sources import hf_resolver
from synthpix.data_sources.base import FileDataSource


class _StubDataSource(FileDataSource):
    """Minimal concrete subclass: identity-load and ``*.mat`` glob."""

    _file_pattern = "**/*.mat"

    def load_file(self, file_path: Path) -> dict[str, Any]:
        """Load file.

        Args:
            file_path: File path used by this test.

        Returns:
            The result used by the test assertions.
        """
        return {"file": file_path}


def _stub_resolve_to_directory(cache_root: Path) -> Any:
    """Stub resolve to directory.

    Args:
        cache_root: Cache root for this test case.

    Returns:
        The value produced for the test scenario.
    """
    # Build a fake repo dir containing a couple of .mat files plus a
    # README.md and a .git/config that must be filtered out by globbing.

    def _stub(spec: Any, **_kwargs: Any) -> Any:
        """Stub.

        Args:
            spec: Spec for this test case.
            **_kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        repo_dir = cache_root / "user" / "repo@main"
        repo_dir.mkdir(parents=True, exist_ok=True)
        (repo_dir / "a.mat").touch()
        (repo_dir / "b.mat").touch()
        (repo_dir / "README.md").write_text("card")
        return repo_dir

    return _stub


def test_filedatasource_resolves_hf_uri(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify filedatasource resolves hf uri.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
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


def test_filedatasource_mixes_hf_and_local_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify filedatasource mixes hf and local paths.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
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


def test_filedatasource_hf_resolver_module_import_failure_surfaces_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify filedatasource hf resolver module import failure surfaces
    clear error.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    # Defensive coverage: if `from synthpix.data_sources import hf_resolver`
    # itself raises ImportError, the guard in base.py still surfaces the
    # actionable `synthpix[hf]` hint and preserves __cause__.
    #
    # NOTE: this is NOT the realistic missing-`[hf]`-extra path. Today the
    # hf_resolver module loads fine without huggingface_hub — the real
    # ImportError fires inside pull_dataset's lazy import on the call arm,
    # which is covered by
    # test_filedatasource_pull_dataset_import_error_surfaces.
    # This test pins the defensive arm against a future refactor that adds
    # an eager huggingface_hub (or other optional) import anywhere in the
    # hf_resolver module-load chain.
    import synthpix.data_sources as ds_pkg  # noqa: PLC0415 - optional or intentionally lazy import

    real_import = builtins.__import__
    original_cause = ImportError("simulated resolver-module import failure")

    # Force the `from synthpix.data_sources import hf_resolver` statement in
    # base.py to re-run by dropping the cached submodule and the attribute.
    monkeypatch.delitem(
        sys.modules,
        "synthpix.data_sources.hf_resolver",
        raising=False,
    )
    monkeypatch.delattr(ds_pkg, "hf_resolver", raising=False)

    def _failing_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Failing import.

        Args:
            name: Name for this test case.
            *args: Positional arguments accepted by the test double.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.

        Raises:
            original_cause: When the simulated failure is triggered.
        """
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify filedatasource pull dataset import error surfaces.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    # If huggingface_hub is missing, pull_dataset (called from inside
    # resolve_to_directory) raises ImportError; base.py must surface the
    # same actionable message and preserve the cause.
    original_cause = ImportError("no huggingface_hub")

    def _raising_resolve(spec: Any, **_kwargs: Any) -> None:
        """Raising resolve.

        Args:
            spec: Spec for this test case.
            **_kwargs: Keyword arguments accepted by the test double.

        Raises:
            original_cause: When the simulated failure is triggered.
        """
        raise original_cause

    monkeypatch.setattr(hf_resolver, "resolve_to_directory", _raising_resolve)

    with pytest.raises(ImportError) as excinfo:
        _StubDataSource("hf://user/repo")

    assert "synthpix[hf]" in str(excinfo.value)
    assert excinfo.value.__cause__ is original_cause


def test_filedatasource_only_hf_uri(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify filedatasource only hf uri.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
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


def test_filedatasource_skips_hidden_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify filedatasource skips hidden dirs.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """

    # Glob with **/*.mat does pick up paths under hidden dirs; verify the
    # pattern + skip behavior matches whatever subclass _file_pattern says.
    # The MAT glob pattern explicitly walks ** and so a .cache/x.mat under
    # the repo *would* appear unless excluded by the pattern itself. This
    # documents the actual behavior (no implicit hidden-dir filter).
    def _stub(spec: Any, **_kwargs: Any) -> Any:
        """Stub.

        Args:
            spec: Spec for this test case.
            **_kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
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
