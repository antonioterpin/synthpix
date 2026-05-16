"""Tests for the ``hf://`` URI resolver.

All network access is mocked. ``pull_dataset`` is patched so the resolver
runs the URI parsing, cache layout, and file-enumeration logic without
touching the Hub.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synthpix.data_sources import hf_resolver


def _fake_pull(call_log: list[dict]):
    # Return a stub that materializes ``local_dir`` and records its kwargs.

    def _stub(repo_id, local_dir, *, revision="main", token=None, **kwargs):
        call_log.append(
            {
                "repo_id": repo_id,
                "local_dir": Path(local_dir),
                "revision": revision,
                "token": token,
                **kwargs,
            }
        )
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        return Path(local_dir).resolve()

    return _stub


def _clear_cache_env(monkeypatch) -> None:
    monkeypatch.delenv("SYNTHPIX_HF_CACHE", raising=False)


def test_resolve_basic(monkeypatch, tmp_path):
    # Plain hf://user/repo URI returns sorted data file paths.
    _clear_cache_env(monkeypatch)
    calls: list[dict] = []

    def _stub(repo_id, local_dir, *, revision="main", token=None, **kwargs):
        calls.append({"repo_id": repo_id, "local_dir": Path(local_dir)})
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "b.mat").touch()
        (Path(local_dir) / "a.mat").touch()
        return Path(local_dir).resolve()

    monkeypatch.setattr(hf_resolver, "pull_dataset", _stub)

    result = hf_resolver.resolve("hf://user/repo", cache_dir=tmp_path)

    assert calls[0]["repo_id"] == "user/repo"
    expected_dir = (tmp_path / "user" / "repo@main").resolve()
    assert calls[0]["local_dir"].resolve() == expected_dir
    assert result == sorted(
        [str(expected_dir / "a.mat"), str(expected_dir / "b.mat")]
    )
    # All paths absolute.
    assert all(Path(p).is_absolute() for p in result)


def test_resolve_revision(monkeypatch, tmp_path):
    # hf://user/repo@v1 forwards revision and lands in a @v1 cache dir.
    _clear_cache_env(monkeypatch)
    calls: list[dict] = []
    monkeypatch.setattr(hf_resolver, "pull_dataset", _fake_pull(calls))

    hf_resolver.resolve("hf://user/repo@v1", cache_dir=tmp_path)

    assert calls[0]["revision"] == "v1"
    assert "@v1" in calls[0]["local_dir"].name
    expected_dir = (tmp_path / "user" / "repo@v1").resolve()
    assert calls[0]["local_dir"].resolve() == expected_dir


def test_resolve_subpath(monkeypatch, tmp_path):
    # A :subpath URI narrows the returned files to that subtree.
    _clear_cache_env(monkeypatch)
    calls: list[dict] = []

    def _stub(repo_id, local_dir, *, revision="main", token=None, **kwargs):
        calls.append({"local_dir": Path(local_dir)})
        root = Path(local_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "train").mkdir()
        (root / "val").mkdir()
        (root / "train" / "a.mat").touch()
        (root / "train" / "b.mat").touch()
        (root / "val" / "c.mat").touch()
        (root / "top.mat").touch()
        return root.resolve()

    monkeypatch.setattr(hf_resolver, "pull_dataset", _stub)

    result = hf_resolver.resolve("hf://user/repo:train", cache_dir=tmp_path)

    expected_train = (tmp_path / "user" / "repo@main" / "train").resolve()
    assert result == sorted(
        [str(expected_train / "a.mat"), str(expected_train / "b.mat")]
    )


def test_resolve_revision_and_subpath(monkeypatch, tmp_path):
    # Combined @revision:subpath forwards both.
    _clear_cache_env(monkeypatch)
    calls: list[dict] = []

    def _stub(repo_id, local_dir, *, revision="main", token=None, **kwargs):
        calls.append({"revision": revision, "local_dir": Path(local_dir)})
        root = Path(local_dir)
        (root / "splits").mkdir(parents=True, exist_ok=True)
        (root / "splits" / "x.mat").touch()
        return root.resolve()

    monkeypatch.setattr(hf_resolver, "pull_dataset", _stub)

    result = hf_resolver.resolve(
        "hf://user/repo@dev:splits", cache_dir=tmp_path
    )

    assert calls[0]["revision"] == "dev"
    assert calls[0]["local_dir"].name == "repo@dev"
    expected = (tmp_path / "user" / "repo@dev" / "splits" / "x.mat").resolve()
    assert result == [str(expected)]


def test_resolve_subpath_missing_raises(monkeypatch, tmp_path):
    # If the pulled tree lacks the requested subpath, raise.
    _clear_cache_env(monkeypatch)

    def _stub(repo_id, local_dir, *, revision="main", token=None, **kwargs):
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "other.mat").touch()
        return Path(local_dir).resolve()

    monkeypatch.setattr(hf_resolver, "pull_dataset", _stub)

    with pytest.raises(FileNotFoundError, match="subpath does not exist"):
        hf_resolver.resolve("hf://user/repo:missing", cache_dir=tmp_path)


def test_resolve_skips_metadata_files(monkeypatch, tmp_path):
    # Dotfiles and root README.md never appear in the result.
    _clear_cache_env(monkeypatch)

    def _stub(repo_id, local_dir, *, revision="main", token=None, **kwargs):
        root = Path(local_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "README.md").write_text("card")
        (root / ".gitattributes").write_text("")
        (root / ".DS_Store").write_text("")
        (root / "data.mat").touch()
        sub = root / "train"
        sub.mkdir()
        # Nested README.md is data-adjacent and should be kept.
        (sub / "README.md").write_text("nested-keep")
        (sub / "x.mat").touch()
        return root.resolve()

    monkeypatch.setattr(hf_resolver, "pull_dataset", _stub)

    result = hf_resolver.resolve("hf://user/repo", cache_dir=tmp_path)

    repo_root = (tmp_path / "user" / "repo@main").resolve()
    assert str(repo_root / "README.md") not in result
    assert str(repo_root / ".gitattributes") not in result
    assert str(repo_root / ".DS_Store") not in result
    assert str(repo_root / "data.mat") in result
    # Nested README.md is kept (only the *root* one is metadata).
    assert str(repo_root / "train" / "README.md") in result
    assert str(repo_root / "train" / "x.mat") in result


@pytest.mark.parametrize(
    "uri, fragment",
    [
        ("hf://no-slash", "repo name"),
        ("hf://", "owner"),
        ("hf://user/repo:..", "'..'"),
        ("hf://user/repo@", "revision"),
    ],
)
def test_resolve_invalid_uri(monkeypatch, tmp_path, uri, fragment):
    # Each malformed URI raises ValueError naming the broken component.
    _clear_cache_env(monkeypatch)
    monkeypatch.setattr(hf_resolver, "pull_dataset", _fake_pull([]))

    with pytest.raises(ValueError, match=fragment):
        hf_resolver.resolve(uri, cache_dir=tmp_path)


def test_resolve_cache_dir_explicit(monkeypatch, tmp_path):
    # Explicit cache_dir wins and is used verbatim (after expand/resolve).
    _clear_cache_env(monkeypatch)
    calls: list[dict] = []
    monkeypatch.setattr(hf_resolver, "pull_dataset", _fake_pull(calls))

    hf_resolver.resolve("hf://user/repo", cache_dir=tmp_path)

    expected = (tmp_path / "user" / "repo@main").resolve()
    assert calls[0]["local_dir"].resolve() == expected


def test_resolve_cache_dir_env(monkeypatch, tmp_path):
    # SYNTHPIX_HF_CACHE provides the cache root when no arg is passed.
    monkeypatch.setenv("SYNTHPIX_HF_CACHE", str(tmp_path))
    calls: list[dict] = []
    monkeypatch.setattr(hf_resolver, "pull_dataset", _fake_pull(calls))

    hf_resolver.resolve("hf://user/repo")

    expected = (tmp_path / "user" / "repo@main").resolve()
    assert calls[0]["local_dir"].resolve() == expected


def test_resolve_cache_dir_default(monkeypatch, tmp_path):
    # Default cache root lives under ``~/.cache/synthpix/hf``.
    _clear_cache_env(monkeypatch)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(hf_resolver.Path, "home", classmethod(lambda cls: fake_home))

    calls: list[dict] = []
    monkeypatch.setattr(hf_resolver, "pull_dataset", _fake_pull(calls))

    hf_resolver.resolve("hf://user/repo")

    expected_prefix = (fake_home / ".cache" / "synthpix" / "hf").resolve()
    assert str(calls[0]["local_dir"].resolve()).startswith(str(expected_prefix))


def test_resolve_token_passes_through(monkeypatch, tmp_path):
    # The token kwarg reaches pull_dataset verbatim.
    _clear_cache_env(monkeypatch)
    calls: list[dict] = []
    monkeypatch.setattr(hf_resolver, "pull_dataset", _fake_pull(calls))

    hf_resolver.resolve(
        "hf://user/repo", cache_dir=tmp_path, token="hf_xxx"
    )

    assert calls[0]["token"] == "hf_xxx"


def test_resolve_idempotent(monkeypatch, tmp_path):
    # Two calls invoke pull_dataset twice; idempotency lives in pull itself.
    _clear_cache_env(monkeypatch)
    calls: list[dict] = []
    monkeypatch.setattr(hf_resolver, "pull_dataset", _fake_pull(calls))

    hf_resolver.resolve("hf://user/repo", cache_dir=tmp_path)
    hf_resolver.resolve("hf://user/repo", cache_dir=tmp_path)

    assert len(calls) == 2
    assert calls[0]["local_dir"].resolve() == calls[1]["local_dir"].resolve()
