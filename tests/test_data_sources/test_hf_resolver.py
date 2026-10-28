"""Tests for the ``hf://`` URI resolver.

All network access is mocked. ``pull_dataset`` is patched so the
resolver runs the URI parsing, cache layout, and file-enumeration
logic without touching the Hub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from synthpix.data_sources import hf_resolver


def _fake_pull(call_log: list[dict]) -> Any:
    """Fake pull.

    Args:
        call_log: Call log for this test case.

    Returns:
        The value produced for the test scenario.
    """
    # Return a stub that materializes ``local_dir`` and records its kwargs.

    def _stub(
        repo_id: str,
        local_dir: Path,
        *,
        revision: str = "main",
        token: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Stub.

        Args:
            repo_id: Repo id for this test case.
            local_dir: Local dir for this test case.
            revision: Revision for this test case.
            token: Token for this test case.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
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


def _clear_cache_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear cache env.

    Args:
        monkeypatch: Fixture that restores patched state after the test.
    """
    monkeypatch.delenv("SYNTHPIX_HF_CACHE", raising=False)


def test_resolve_basic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify resolve basic.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    # Plain hf://user/repo URI returns sorted data file paths.
    _clear_cache_env(monkeypatch)
    calls: list[dict] = []

    def _stub(
        repo_id: str,
        local_dir: Path,
        *,
        revision: str = "main",
        token: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Stub.

        Args:
            repo_id: Repo id for this test case.
            local_dir: Local dir for this test case.
            revision: Revision for this test case.
            token: Token for this test case.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
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


def test_resolve_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify resolve revision.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    # hf://user/repo@v1 forwards revision and lands in a @v1 cache dir.
    _clear_cache_env(monkeypatch)
    calls: list[dict] = []
    monkeypatch.setattr(hf_resolver, "pull_dataset", _fake_pull(calls))

    hf_resolver.resolve("hf://user/repo@v1", cache_dir=tmp_path)

    assert calls[0]["revision"] == "v1"
    assert "@v1" in calls[0]["local_dir"].name
    expected_dir = (tmp_path / "user" / "repo@v1").resolve()
    assert calls[0]["local_dir"].resolve() == expected_dir


def test_resolve_subpath(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify resolve subpath.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    # A :subpath URI narrows the returned files to that subtree.
    _clear_cache_env(monkeypatch)
    calls: list[dict] = []

    def _stub(
        repo_id: str,
        local_dir: Path,
        *,
        revision: str = "main",
        token: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Stub.

        Args:
            repo_id: Repo id for this test case.
            local_dir: Local dir for this test case.
            revision: Revision for this test case.
            token: Token for this test case.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
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


def test_resolve_revision_and_subpath(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify resolve revision and subpath.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    # Combined @revision:subpath forwards both.
    _clear_cache_env(monkeypatch)
    calls: list[dict] = []

    def _stub(
        repo_id: str,
        local_dir: Path,
        *,
        revision: str = "main",
        token: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Stub.

        Args:
            repo_id: Repo id for this test case.
            local_dir: Local dir for this test case.
            revision: Revision for this test case.
            token: Token for this test case.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
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


def test_resolve_subpath_missing_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify resolve subpath missing raises.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    # If the pulled tree lacks the requested subpath, raise.
    _clear_cache_env(monkeypatch)

    def _stub(
        repo_id: str,
        local_dir: Path,
        *,
        revision: str = "main",
        token: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Stub.

        Args:
            repo_id: Repo id for this test case.
            local_dir: Local dir for this test case.
            revision: Revision for this test case.
            token: Token for this test case.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "other.mat").touch()
        return Path(local_dir).resolve()

    monkeypatch.setattr(hf_resolver, "pull_dataset", _stub)

    with pytest.raises(FileNotFoundError, match="subpath does not exist"):
        hf_resolver.resolve("hf://user/repo:missing", cache_dir=tmp_path)


def test_resolve_skips_metadata_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify resolve skips metadata files.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    # Dotfiles and root README.md never appear in the result.
    _clear_cache_env(monkeypatch)

    def _stub(
        repo_id: str,
        local_dir: Path,
        *,
        revision: str = "main",
        token: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Stub.

        Args:
            repo_id: Repo id for this test case.
            local_dir: Local dir for this test case.
            revision: Revision for this test case.
            token: Token for this test case.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
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
def test_resolve_invalid_uri(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, uri: str, fragment: str
) -> None:
    """Verify resolve invalid uri.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
        uri: Uri used by this test.
        fragment: Fragment used by this test.
    """
    # Each malformed URI raises ValueError naming the broken component.
    _clear_cache_env(monkeypatch)
    monkeypatch.setattr(hf_resolver, "pull_dataset", _fake_pull([]))

    with pytest.raises(ValueError, match=fragment):
        hf_resolver.resolve(uri, cache_dir=tmp_path)


def test_resolve_cache_dir_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify resolve cache dir explicit.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    # Explicit cache_dir wins and is used verbatim (after expand/resolve).
    _clear_cache_env(monkeypatch)
    calls: list[dict] = []
    monkeypatch.setattr(hf_resolver, "pull_dataset", _fake_pull(calls))

    hf_resolver.resolve("hf://user/repo", cache_dir=tmp_path)

    expected = (tmp_path / "user" / "repo@main").resolve()
    assert calls[0]["local_dir"].resolve() == expected


def test_resolve_cache_dir_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify resolve cache dir env.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    # SYNTHPIX_HF_CACHE provides the cache root when no arg is passed.
    monkeypatch.setenv("SYNTHPIX_HF_CACHE", str(tmp_path))
    calls: list[dict] = []
    monkeypatch.setattr(hf_resolver, "pull_dataset", _fake_pull(calls))

    hf_resolver.resolve("hf://user/repo")

    expected = (tmp_path / "user" / "repo@main").resolve()
    assert calls[0]["local_dir"].resolve() == expected


def test_resolve_cache_dir_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify resolve cache dir default.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    # Default cache root lives under ``~/.cache/synthpix/hf``.
    _clear_cache_env(monkeypatch)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(
        hf_resolver.Path, "home", classmethod(lambda cls: fake_home)
    )

    calls: list[dict] = []
    monkeypatch.setattr(hf_resolver, "pull_dataset", _fake_pull(calls))

    hf_resolver.resolve("hf://user/repo")

    expected_prefix = (fake_home / ".cache" / "synthpix" / "hf").resolve()
    assert str(calls[0]["local_dir"].resolve()).startswith(str(expected_prefix))


def test_resolve_token_passes_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify resolve token passes through.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    # The token kwarg reaches pull_dataset verbatim.
    _clear_cache_env(monkeypatch)
    calls: list[dict] = []
    monkeypatch.setattr(hf_resolver, "pull_dataset", _fake_pull(calls))

    hf_resolver.resolve("hf://user/repo", cache_dir=tmp_path, token="hf_xxx")

    assert calls[0]["token"] == "hf_xxx"


@pytest.mark.parametrize(
    "uri",
    [
        # owner/name path-traversal attempts caught by sanitization.
        "hf://../bad/repo:train",
        "hf://bad/../sneak",
    ],
)
def test_resolve_rejects_path_traversal_in_repo_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, uri: str
) -> None:
    """Verify resolve rejects path traversal in repo id.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
        uri: Uri used by this test.
    """
    _clear_cache_env(monkeypatch)
    monkeypatch.setattr(hf_resolver, "pull_dataset", _fake_pull([]))

    with pytest.raises(ValueError, match="invalid hf:// URI"):
        hf_resolver.resolve(uri, cache_dir=tmp_path)


def test_resolve_flattens_slash_in_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify resolve flattens slash in revision.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    # ``refs/pr/123`` is a legal upstream revision but must not create
    # nested directories under the cache root.
    _clear_cache_env(monkeypatch)
    calls: list[dict] = []
    monkeypatch.setattr(hf_resolver, "pull_dataset", _fake_pull(calls))

    hf_resolver.resolve("hf://user/repo@refs/pr/123", cache_dir=tmp_path)

    expected = (tmp_path / "user" / "repo@refs__pr__123").resolve()
    assert calls[0]["local_dir"].resolve() == expected
    # The original revision is still what we hand to pull_dataset.
    assert calls[0]["revision"] == "refs/pr/123"


def test_resolve_rejects_double_dot_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify resolve rejects double dot revision.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    _clear_cache_env(monkeypatch)
    monkeypatch.setattr(hf_resolver, "pull_dataset", _fake_pull([]))

    with pytest.raises(ValueError, match="invalid hf:// URI"):
        hf_resolver.resolve("hf://user/repo@..", cache_dir=tmp_path)


def test_resolve_skips_files_in_hidden_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify resolve skips files in hidden directories.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    # _is_metadata_file must exclude *any* file inside a hidden directory,
    # not just files whose own name begins with '.'.
    _clear_cache_env(monkeypatch)

    def _stub(
        repo_id: str,
        local_dir: Path,
        *,
        revision: str = "main",
        token: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Stub.

        Args:
            repo_id: Repo id for this test case.
            local_dir: Local dir for this test case.
            revision: Revision for this test case.
            token: Token for this test case.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        root = Path(local_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / ".git").mkdir()
        (root / ".git" / "config").write_text("[core]")
        (root / "data.mat").touch()
        return root.resolve()

    monkeypatch.setattr(hf_resolver, "pull_dataset", _stub)

    result = hf_resolver.resolve("hf://user/repo", cache_dir=tmp_path)

    repo_root = (tmp_path / "user" / "repo@main").resolve()
    assert str(repo_root / "data.mat") in result
    # File under .git/ has a non-hidden basename but a hidden ancestor.
    assert str(repo_root / ".git" / "config") not in result


def test_resolve_to_directory_returns_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify resolve to directory returns path.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    _clear_cache_env(monkeypatch)
    calls: list[dict] = []
    monkeypatch.setattr(hf_resolver, "pull_dataset", _fake_pull(calls))

    result = hf_resolver.resolve_to_directory(
        "hf://user/repo", cache_dir=tmp_path
    )

    expected = (tmp_path / "user" / "repo@main").resolve()
    assert result == expected


def test_resolve_to_directory_applies_subpath(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify resolve to directory applies subpath.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    _clear_cache_env(monkeypatch)

    def _stub(repo_id: str, local_dir: Path, **kwargs: Any) -> Any:
        """Stub.

        Args:
            repo_id: Repo id for this test case.
            local_dir: Local dir for this test case.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        root = Path(local_dir)
        (root / "train").mkdir(parents=True, exist_ok=True)
        return root.resolve()

    monkeypatch.setattr(hf_resolver, "pull_dataset", _stub)

    result = hf_resolver.resolve_to_directory(
        "hf://user/repo:train", cache_dir=tmp_path
    )

    expected = (tmp_path / "user" / "repo@main" / "train").resolve()
    assert result == expected


def test_resolve_to_directory_missing_subpath_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify resolve to directory missing subpath raises.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    _clear_cache_env(monkeypatch)

    def _stub(repo_id: str, local_dir: Path, **kwargs: Any) -> Any:
        """Stub.

        Args:
            repo_id: Repo id for this test case.
            local_dir: Local dir for this test case.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        return Path(local_dir).resolve()

    monkeypatch.setattr(hf_resolver, "pull_dataset", _stub)

    with pytest.raises(FileNotFoundError, match="subpath does not exist"):
        hf_resolver.resolve_to_directory(
            "hf://user/repo:missing", cache_dir=tmp_path
        )


def test_resolve_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify resolve idempotent.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    # Two calls invoke pull_dataset twice; idempotency lives in pull itself.
    _clear_cache_env(monkeypatch)
    calls: list[dict] = []
    monkeypatch.setattr(hf_resolver, "pull_dataset", _fake_pull(calls))

    hf_resolver.resolve("hf://user/repo", cache_dir=tmp_path)
    hf_resolver.resolve("hf://user/repo", cache_dir=tmp_path)

    assert len(calls) == 2
    assert calls[0]["local_dir"].resolve() == calls[1]["local_dir"].resolve()


def test_resolve_pinned_commit_pulls_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify resolve pinned commit pulls once.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    # A commit-hash revision is pulled once; a branch is pulled every time.
    _clear_cache_env(monkeypatch)
    calls: list[dict] = []
    monkeypatch.setattr(hf_resolver, "pull_dataset", _fake_pull(calls))
    sha = "a" * 40

    for _ in range(2):
        hf_resolver.resolve_to_directory(f"hf://o/n@{sha}", cache_dir=tmp_path)
        hf_resolver.resolve_to_directory("hf://o/n@main", cache_dir=tmp_path)

    assert [c["revision"] for c in calls] == [sha, "main", "main"]
