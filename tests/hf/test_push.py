"""Tests for ``synthpix.hf.push.push_dataset``.

All network access is mocked. The real ``huggingface_hub`` module is
replaced with a stand-in that records the kwargs passed to
``HfApi.create_repo`` and ``HfApi.upload_folder``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

from synthpix.hf import push as push_mod
from synthpix.hf.card import DatasetCardMeta


class _FakeCommitInfo:
    """Fake Commit Info test double."""

    def __init__(self, oid: str = "deadbeef") -> None:
        """Initialize the test double.

        Args:
            oid: Oid for this test case.
        """
        self.oid = oid


class _FakeHfApi:
    """Fake Hf Api test double."""

    def __init__(self, token: str | None = None, **_: object) -> None:
        """Initialize the test double.

        Args:
            token: Token for this test case.
            **_: Keyword arguments accepted by the test double.
        """
        self.token = token
        self.create_repo_calls: list[dict] = []
        self.upload_folder_calls: list[dict] = []
        self.list_repo_files_calls: list[dict] = []
        self._instances_list = _FakeHfApiCollector._INSTANCES
        self._instances_list.append(self)
        # Configurable behaviors via class-level overrides.
        self.list_repo_files_return: list[str] = []
        self.list_repo_files_raises: BaseException | None = None
        self.upload_folder_return: object = _FakeCommitInfo()

    def create_repo(self, **kwargs: Any) -> dict[str, Any]:
        """Create repo.

        Args:
            **kwargs: Kwargs used by this test.

        Returns:
            The result used by the test assertions.
        """
        self.create_repo_calls.append(kwargs)
        return {"url": "https://huggingface.co/datasets/" + kwargs["repo_id"]}

    def upload_folder(self, **kwargs: Any) -> Any:
        """Upload folder.

        Args:
            **kwargs: Kwargs used by this test.

        Returns:
            The result used by the test assertions.
        """
        self.upload_folder_calls.append(kwargs)
        return self.upload_folder_return

    def list_repo_files(self, repo_id: str, **kwargs: Any) -> Any:
        """List repo files.

        Args:
            repo_id: Repo id used by this test.
            **kwargs: Kwargs used by this test.

        Returns:
            The result used by the test assertions.

        Raises:
            self.list_repo_files_raises: When the simulated failure is
                triggered.
        """
        self.list_repo_files_calls.append({"repo_id": repo_id, **kwargs})
        if self.list_repo_files_raises is not None:
            raise self.list_repo_files_raises
        return list(self.list_repo_files_return)


class _FakeHfApiCollector:
    """Fake Hf Api Collector test double."""

    # Module-level shared list so tests can find the instance that
    # ``push_dataset`` constructed.
    _INSTANCES: ClassVar[list[_FakeHfApi]] = []


class _FakeRepoNotFoundError(Exception):
    """Fake Repo Not Found Error test double."""

    pass


class _FakeErrorsModule:
    """Expose the Hub exception used by push tests.

    Attributes:
        RepositoryNotFoundError: Error raised for a missing repository.
    """

    RepositoryNotFoundError = _FakeRepoNotFoundError


class _FakeHub:
    """Minimal stand-in for ``huggingface_hub``."""

    def __init__(self) -> None:
        """Initialize the test double."""
        self.HfApi = _FakeHfApi
        self.errors = _FakeErrorsModule


def _install_fake_hub(monkeypatch: pytest.MonkeyPatch) -> _FakeHub:
    """Install fake hub.

    Args:
        monkeypatch: Fixture that restores patched state after the test.

    Returns:
        The value produced for the test scenario.
    """
    _FakeHfApiCollector._INSTANCES.clear()
    fake = _FakeHub()
    real_import_module = importlib.import_module

    def _import_module(name: str, package: str | None = None) -> Any:
        """Import module.

        Args:
            name: Name for this test case.
            package: Package for this test case.

        Returns:
            The value produced for the test scenario.
        """
        if name == "huggingface_hub":
            return fake
        if name == "huggingface_hub.errors":
            return fake.errors
        return real_import_module(name, package)

    monkeypatch.setattr(push_mod.importlib, "import_module", _import_module)
    return fake


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear env.

    Args:
        monkeypatch: Fixture that restores patched state after the test.
    """
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_HUB_TOKEN", raising=False)


def _populate(local_dir: Path) -> None:
    """Populate.

    Args:
        local_dir: Local dir for this test case.
    """
    train = local_dir / "train"
    train.mkdir(parents=True)
    (train / "flow.mat").write_bytes(b"\x00" * 16)


def _populate_piv_tree(root: Path) -> None:
    """Build a tree mirroring scripts/download_piv_1.py's output layout.

    Splits at ``{train,val,test,tune}/<scenario>/Re####/*.mat`` plus a
    ``splits/`` dir, a top-level README.md, and the ``raw_class1/`` /
    ``packed_class1/`` staging dirs that must NOT be uploaded.

    Args:
        root: Root for this test case.
    """
    (root / "train" / "backstep" / "Re1000").mkdir(parents=True)
    (root / "train" / "backstep" / "Re1000" / "flow_0001.mat").write_bytes(
        b"\x00" * 8
    )
    (root / "train" / "cylinder").mkdir(parents=True)
    (root / "train" / "cylinder" / "flow_0002.mat").write_bytes(b"\x00" * 8)
    (root / "val" / "uniform").mkdir(parents=True)
    (root / "val" / "uniform" / "flow_0003.mat").write_bytes(b"\x00" * 8)
    (root / "test" / "DNS" / "Re3900").mkdir(parents=True)
    (root / "test" / "DNS" / "Re3900" / "flow_0004.mat").write_bytes(
        b"\x00" * 8
    )
    (root / "tune").mkdir(parents=True)
    (root / "tune" / "flow_0005.mat").write_bytes(b"\x00" * 8)
    (root / "splits").mkdir(parents=True)
    (root / "splits" / "train.txt").write_text("flow_0001.mat\n")
    (root / "splits" / "val.txt").write_text("flow_0003.mat\n")
    (root / "README.md").write_text("# card\n")
    # Excluded staging dirs.
    (root / "raw_class1" / "PIV_zips").mkdir(parents=True)
    (root / "raw_class1" / "PIV_zips" / "a.zip").write_bytes(b"PK")
    (root / "packed_class1" / "backstep").mkdir(parents=True)
    (root / "packed_class1" / "backstep" / "flow_0001.mat").write_bytes(b"\x00")
    # Junk that must be ignored even though it sits under a split.
    (root / "train" / ".DS_Store").write_bytes(b"junk")
    (root / "train" / "backstep" / "__pycache__").mkdir(parents=True)
    (root / "train" / "backstep" / "__pycache__" / "x.pyc").write_bytes(b"\x00")


def test_filter_local_files_matches_real_piv_layout(tmp_path: Path) -> None:
    """Verify filter local files matches real piv layout.

    Args:
        tmp_path: Temporary directory for this test.
    """
    _populate_piv_tree(tmp_path)

    selected = push_mod._filter_local_files(
        tmp_path, push_mod._DEFAULT_INCLUDE, push_mod._DEFAULT_IGNORE
    )

    assert selected == sorted(
        [
            "README.md",
            "splits/train.txt",
            "splits/val.txt",
            "test/DNS/Re3900/flow_0004.mat",
            "train/backstep/Re1000/flow_0001.mat",
            "train/cylinder/flow_0002.mat",
            "tune/flow_0005.mat",
            "val/uniform/flow_0003.mat",
        ]
    )
    # Staging dirs and junk must never be selected.
    assert not any(s.startswith("raw_class1/") for s in selected)
    assert not any(s.startswith("packed_class1/") for s in selected)
    assert not any(".DS_Store" in s for s in selected)
    assert not any("__pycache__" in s for s in selected)


def test_push_basic_private(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify push basic private.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)

    result = push_mod.push_dataset(tmp_path, "user/repo")

    [api] = _FakeHfApiCollector._INSTANCES
    assert len(api.create_repo_calls) == 1
    create = api.create_repo_calls[0]
    assert create["private"] is True
    assert create["exist_ok"] is True
    assert create["repo_type"] == "dataset"
    assert create["repo_id"] == "user/repo"

    assert len(api.upload_folder_calls) == 1
    up = api.upload_folder_calls[0]
    assert up["repo_id"] == "user/repo"
    assert up["repo_type"] == "dataset"
    assert up["folder_path"] == str(tmp_path)
    assert up["revision"] == "main"
    assert up["allow_patterns"] == list(push_mod._DEFAULT_INCLUDE)
    assert up["ignore_patterns"] == list(push_mod._DEFAULT_IGNORE)
    assert up["commit_message"] == "Upload via synthpix-hf"
    # hub 1.x: upload_folder has no parallelism kwarg.
    assert "num_workers" not in up
    assert "max_workers" not in up

    assert result == "deadbeef"


def test_push_public_without_allow_public_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify push public without allow public raises.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)

    with pytest.raises(PermissionError):
        push_mod.push_dataset(tmp_path, "user/repo", private=False)

    assert _FakeHfApiCollector._INSTANCES == []


@pytest.mark.parametrize("private_value", [0, "", None])
def test_push_safety_gate_uses_truthiness_not_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, private_value: Any
) -> None:
    """Verify push safety gate uses truthiness not identity.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
        private_value: Private value used by this test.
    """
    # Regression: the gate must not be bypassed by a falsy-but-not-False
    # ``private`` value (0, "", None). Identity comparison
    # (``private is False``) would let these through.
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)

    with pytest.raises(PermissionError):
        push_mod.push_dataset(tmp_path, "user/repo", private=private_value)

    assert _FakeHfApiCollector._INSTANCES == []


def test_push_public_with_allow_public_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify push public with allow public succeeds.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)

    push_mod.push_dataset(
        tmp_path, "user/repo", private=False, allow_public=True
    )

    [api] = _FakeHfApiCollector._INSTANCES
    assert api.create_repo_calls[0]["private"] is False


def test_push_local_dir_must_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify push local dir must exist.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)

    with pytest.raises(ValueError):
        push_mod.push_dataset(tmp_path / "nope", "user/repo")


def test_push_local_dir_must_be_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify push local dir must be directory.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    f = tmp_path / "file.txt"
    f.write_text("x")

    with pytest.raises(ValueError):
        push_mod.push_dataset(f, "user/repo")


def test_push_repo_id_must_have_owner_name_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify push repo id must have owner name shape.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)

    with pytest.raises(ValueError):
        push_mod.push_dataset(tmp_path, "just-a-name")


def test_push_writes_card_when_meta_provided(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify push writes card when meta provided.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)

    meta = DatasetCardMeta(
        name="example",
        source_url="https://example.org/src",
        citation="My citation",
        pretty_name="Example Dataset",
    )

    push_mod.push_dataset(tmp_path, "user/repo", card_meta=meta)

    readme = tmp_path / "README.md"
    assert readme.exists()
    assert "My citation" in readme.read_text()


def test_push_dry_run_does_not_write_card(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify push dry run does not write card.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    # Regression: dry-run must not mutate the working tree. Even when
    # ``card_meta`` is provided, ``README.md`` must not be rewritten by
    # a previewing call.
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("preserve-me")

    meta = DatasetCardMeta(
        name="example",
        source_url="https://example.org/src",
        citation="My citation",
    )

    push_mod.push_dataset(tmp_path, "user/repo", card_meta=meta, dry_run=True)

    assert readme.read_text() == "preserve-me"


def test_push_card_overwrite_logs_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify push card overwrite logs warning.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)

    readme = tmp_path / "README.md"
    readme.write_text("pre-existing content")

    warnings: list[str] = []
    monkeypatch.setattr(
        push_mod.logger,
        "warning",
        lambda msg, *a, **kw: warnings.append(msg),
    )

    meta = DatasetCardMeta(
        name="example",
        source_url="https://example.org/src",
        citation="My citation",
    )

    push_mod.push_dataset(tmp_path, "user/repo", card_meta=meta)

    assert readme.read_text() != "pre-existing content"
    assert any("overwriting" in w.lower() for w in warnings)


def test_push_card_skipped_when_meta_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify push card skipped when meta none.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)

    readme = tmp_path / "README.md"
    readme.write_text("untouched content")

    push_mod.push_dataset(tmp_path, "user/repo")

    assert readme.read_text() == "untouched content"


def test_push_dry_run_lists_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify push dry run lists files.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
        capsys: Capture of standard output and standard error.
    """
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)
    # Add another file so we can assert both new + unchanged appear.
    (tmp_path / "train" / "extra.mat").write_bytes(b"\x00" * 8)

    # Configure the FakeHfApi to claim one of the local files exists remote.
    def _factory_token(token: str | None = None, **_kwargs: Any) -> Any:
        """Factory token.

        Args:
            token: Token for this test case.
            **_kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        api = _FakeHfApi(token=token)
        api.list_repo_files_return = ["train/flow.mat"]
        return api

    monkeypatch.setattr(
        push_mod.importlib.import_module("huggingface_hub"),
        "HfApi",
        _factory_token,
    )

    result = push_mod.push_dataset(tmp_path, "user/repo", dry_run=True)

    captured = capsys.readouterr()
    assert result == "dry-run"
    assert "new files: 1" in captured.out
    assert "unchanged files: 1" in captured.out
    # No upload_folder call was made.
    [api] = _FakeHfApiCollector._INSTANCES
    assert api.upload_folder_calls == []
    assert len(api.list_repo_files_calls) == 1


def test_push_dry_run_repo_not_found_treated_as_all_new(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify push dry run repo not found treated as all new.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
        capsys: Capture of standard output and standard error.
    """
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)

    def _factory_token(token: str | None = None, **_kwargs: Any) -> Any:
        """Factory token.

        Args:
            token: Token for this test case.
            **_kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        api = _FakeHfApi(token=token)
        api.list_repo_files_raises = _FakeRepoNotFoundError("missing")
        return api

    monkeypatch.setattr(
        push_mod.importlib.import_module("huggingface_hub"),
        "HfApi",
        _factory_token,
    )

    result = push_mod.push_dataset(tmp_path, "user/repo", dry_run=True)

    captured = capsys.readouterr()
    assert result == "dry-run"
    assert "new files: 1" in captured.out
    assert "unchanged files: 0" in captured.out


def test_push_token_passes_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify push token passes through.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)

    push_mod.push_dataset(tmp_path, "user/repo", token="hf_xxx")

    [api] = _FakeHfApiCollector._INSTANCES
    assert api.token == "hf_xxx"


def test_push_token_resolved_from_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify push token resolved from env.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    _clear_env(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)

    push_mod.push_dataset(tmp_path, "user/repo")

    [api] = _FakeHfApiCollector._INSTANCES
    assert api.token == "hf_env"


def test_push_hf_not_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify push hf not installed.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    _clear_env(monkeypatch)
    _populate(tmp_path)

    real_import_module = importlib.import_module

    def _missing(name: str, package: str | None = None) -> Any:
        """Missing.

        Args:
            name: Name for this test case.
            package: Package for this test case.

        Returns:
            The value produced for the test scenario.

        Raises:
            ImportError: When the simulated failure is triggered.
        """
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError("no module named huggingface_hub")
        return real_import_module(name, package)

    monkeypatch.setattr(push_mod.importlib, "import_module", _missing)
    # Also make the bare ``import huggingface_hub`` in auth._cached_token
    # behave as if the library were absent, so token resolution degrades
    # gracefully instead of finding the real (installed) hub.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    with pytest.raises(ImportError) as excinfo:
        push_mod.push_dataset(tmp_path, "user/repo")

    assert "synthpix[hf]" in str(excinfo.value)


class _StrictHfApi(_FakeHfApi):
    """Like ``_FakeHfApi`` but ``upload_folder`` rejects unknown kwargs.

    Mirrors the real hub 2.0 ``HfApi.upload_folder`` signature, so
    passing an unsupported kwarg (e.g. ``num_workers``) is caught
    without a network call.
    """

    _ALLOWED: ClassVar[set[str]] = {
        "repo_id",
        "folder_path",
        "path_in_repo",
        "commit_message",
        "commit_description",
        "token",
        "repo_type",
        "revision",
        "create_pr",
        "parent_commit",
        "allow_patterns",
        "ignore_patterns",
        "delete_patterns",
        "run_as_future",
    }

    def upload_folder(self, **kwargs: Any) -> Any:
        """Upload folder.

        Args:
            **kwargs: Kwargs used by this test.

        Returns:
            The result used by the test assertions.

        Raises:
            TypeError: When the simulated failure is triggered.
        """
        unexpected = set(kwargs) - self._ALLOWED
        if unexpected:
            raise TypeError(
                "upload_folder() got an unexpected keyword argument "
                f"{sorted(unexpected)[0]!r}"
            )
        return super().upload_folder(**kwargs)


def _install_strict_hub(monkeypatch: pytest.MonkeyPatch) -> _FakeHub:
    """Install strict hub.

    Args:
        monkeypatch: Fixture that restores patched state after the test.

    Returns:
        The value produced for the test scenario.
    """
    fake = _install_fake_hub(monkeypatch)
    fake.HfApi = _StrictHfApi
    return fake


def test_push_upload_folder_no_unsupported_kwargs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``upload_folder`` only gets kwargs the real hub 2.0 accepts.

    Args:
        monkeypatch: Fixture that restores patched state after the test.
        tmp_path: Temporary directory for this test.
    """
    _clear_env(monkeypatch)
    _install_strict_hub(monkeypatch)
    _populate(tmp_path)

    result = push_mod.push_dataset(tmp_path, "user/repo")

    [api] = _FakeHfApiCollector._INSTANCES
    assert len(api.upload_folder_calls) == 1
    up = api.upload_folder_calls[0]
    assert "num_workers" not in up
    assert "max_workers" not in up
    assert result == "deadbeef"


def test_push_returns_oid_not_commit_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: hub-1.x ``CommitInfo`` is a ``str`` whose value is the
    commit URL, not the oid.

    Because ``CommitInfo`` subclasses ``str``, ``isinstance(ci, str)``
    is ``True``; the old branch order then returned the full commit URL
    instead of the short commit sha.

    Args:
        monkeypatch: Fixture that restores patched state after the test.
        tmp_path: Temporary directory for this test.
    """
    _clear_env(monkeypatch)
    fake = _install_fake_hub(monkeypatch)
    _populate(tmp_path)

    class _StrCommitInfo(str):
        """String commit URL with the Hub commit identifier.

        Attributes:
            oid: Commit identifier returned by the upload.
        """

        oid: str

        # Mirrors huggingface_hub.CommitInfo: a str whose value is the
        # commit URL, with an ``oid`` attribute holding the sha.
        def __new__(cls, url: str, oid: str) -> Any:
            """Construct the test value.

            Args:
                url: Url for this test case.
                oid: Oid for this test case.

            Returns:
                The value produced for the test scenario.
            """
            obj = super().__new__(cls, url)
            obj.oid = oid
            return obj

    commit_url = "https://huggingface.co/datasets/user/repo/commit/abc123sha"

    def _factory(token: str | None = None, **_kwargs: Any) -> Any:
        """Factory.

        Args:
            token: Token for this test case.
            **_kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        api = _FakeHfApi(token=token)
        api.upload_folder_return = _StrCommitInfo(commit_url, "abc123sha")
        return api

    monkeypatch.setattr(fake, "HfApi", _factory)

    result = push_mod.push_dataset(tmp_path, "user/repo")

    assert result == "abc123sha"
    assert result != commit_url


def test_push_token_never_logged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify push token never logged.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
        capsys: Capture of standard output and standard error.
        caplog: Capture of emitted log records.
    """
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)

    seen_log_messages: list[str] = []
    for level in ("info", "warning", "error", "debug"):
        original = getattr(push_mod.logger, level)

        def _spy(msg: str, *a: Any, __orig: Any = original, **kw: Any) -> Any:
            """Spy.

            Args:
                msg: Msg for this test case.
                *a: Positional arguments accepted by the test double.
                __orig: Orig for this test case.
                **kw: Keyword arguments accepted by the test double.

            Returns:
                The value produced for the test scenario.
            """
            seen_log_messages.append(str(msg))
            return __orig(msg, *a, **kw)

        monkeypatch.setattr(push_mod.logger, level, _spy)

    push_mod.push_dataset(tmp_path, "user/repo", token="hf_secret")

    captured = capsys.readouterr()
    assert "hf_secret" not in captured.out
    assert "hf_secret" not in captured.err
    for msg in seen_log_messages:
        assert "hf_secret" not in msg
    for record in caplog.records:
        assert "hf_secret" not in record.getMessage()
