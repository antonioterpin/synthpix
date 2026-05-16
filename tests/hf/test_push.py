"""Tests for ``synthpix.hf.push.push_dataset``.

All network access is mocked. The real ``huggingface_hub`` module is
replaced with a stand-in that records the kwargs passed to
``HfApi.create_repo`` and ``HfApi.upload_folder``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from synthpix.hf import push as push_mod
from synthpix.hf.card import DatasetCardMeta


class _FakeCommitInfo:
    def __init__(self, oid: str = "deadbeef") -> None:
        self.oid = oid


class _FakeHfApi:
    def __init__(self, token=None, **_: object) -> None:
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

    def create_repo(self, **kwargs):
        self.create_repo_calls.append(kwargs)
        return {"url": "https://huggingface.co/datasets/" + kwargs["repo_id"]}

    def upload_folder(self, **kwargs):
        self.upload_folder_calls.append(kwargs)
        return self.upload_folder_return

    def list_repo_files(self, repo_id, **kwargs):
        self.list_repo_files_calls.append(
            {"repo_id": repo_id, **kwargs}
        )
        if self.list_repo_files_raises is not None:
            raise self.list_repo_files_raises
        return list(self.list_repo_files_return)


class _FakeHfApiCollector:
    # Module-level shared list so tests can find the instance that
    # ``push_dataset`` constructed.
    _INSTANCES: list[_FakeHfApi] = []


class _FakeRepoNotFound(Exception):
    pass


class _FakeErrorsModule:
    RepositoryNotFoundError = _FakeRepoNotFound


class _FakeHub:
    """Minimal stand-in for ``huggingface_hub``."""

    def __init__(self) -> None:
        self.HfApi = _FakeHfApi
        self.errors = _FakeErrorsModule


def _install_fake_hub(monkeypatch) -> _FakeHub:
    _FakeHfApiCollector._INSTANCES.clear()
    fake = _FakeHub()
    real_import_module = importlib.import_module

    def _import_module(name, package=None):
        if name == "huggingface_hub":
            return fake
        if name == "huggingface_hub.errors":
            return fake.errors
        return real_import_module(name, package)

    monkeypatch.setattr(
        push_mod.importlib, "import_module", _import_module
    )
    return fake


def _clear_env(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_HUB_TOKEN", raising=False)


def _populate(local_dir: Path) -> None:
    train = local_dir / "train"
    train.mkdir(parents=True)
    (train / "flow.mat").write_bytes(b"\x00" * 16)


def test_push_basic_private(monkeypatch, tmp_path):
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

    assert result == "deadbeef"


def test_push_public_without_allow_public_raises(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)

    with pytest.raises(PermissionError):
        push_mod.push_dataset(tmp_path, "user/repo", private=False)

    assert _FakeHfApiCollector._INSTANCES == []


def test_push_public_with_allow_public_succeeds(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)

    push_mod.push_dataset(
        tmp_path, "user/repo", private=False, allow_public=True
    )

    [api] = _FakeHfApiCollector._INSTANCES
    assert api.create_repo_calls[0]["private"] is False


def test_push_local_dir_must_exist(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)

    with pytest.raises(ValueError):
        push_mod.push_dataset(tmp_path / "nope", "user/repo")


def test_push_local_dir_must_be_directory(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    f = tmp_path / "file.txt"
    f.write_text("x")

    with pytest.raises(ValueError):
        push_mod.push_dataset(f, "user/repo")


def test_push_repo_id_must_have_owner_name_shape(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)

    with pytest.raises(ValueError):
        push_mod.push_dataset(tmp_path, "just-a-name")


def test_push_writes_card_when_meta_provided(monkeypatch, tmp_path):
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


def test_push_card_overwrite_logs_warning(monkeypatch, tmp_path):
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


def test_push_card_skipped_when_meta_none(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)

    readme = tmp_path / "README.md"
    readme.write_text("untouched content")

    push_mod.push_dataset(tmp_path, "user/repo")

    assert readme.read_text() == "untouched content"


def test_push_dry_run_lists_files(monkeypatch, tmp_path, capsys):
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)
    # Add another file so we can assert both new + unchanged appear.
    (tmp_path / "train" / "extra.mat").write_bytes(b"\x00" * 8)

    # Configure the FakeHfApi to claim one of the local files exists remote.
    def _factory_token(token=None, **_kwargs):
        api = _FakeHfApi(token=token)
        api.list_repo_files_return = ["train/flow.mat"]
        return api

    monkeypatch.setattr(
        push_mod.importlib.import_module("huggingface_hub"),
        "HfApi",
        _factory_token,
    )

    result = push_mod.push_dataset(
        tmp_path, "user/repo", dry_run=True
    )

    captured = capsys.readouterr()
    assert result == "dry-run"
    assert "new files: 1" in captured.out
    assert "unchanged files: 1" in captured.out
    # No upload_folder call was made.
    [api] = _FakeHfApiCollector._INSTANCES
    assert api.upload_folder_calls == []
    assert len(api.list_repo_files_calls) == 1


def test_push_dry_run_repo_not_found_treated_as_all_new(
    monkeypatch, tmp_path, capsys
):
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)

    def _factory_token(token=None, **_kwargs):
        api = _FakeHfApi(token=token)
        api.list_repo_files_raises = _FakeRepoNotFound("missing")
        return api

    monkeypatch.setattr(
        push_mod.importlib.import_module("huggingface_hub"),
        "HfApi",
        _factory_token,
    )

    result = push_mod.push_dataset(
        tmp_path, "user/repo", dry_run=True
    )

    captured = capsys.readouterr()
    assert result == "dry-run"
    assert "new files: 1" in captured.out
    assert "unchanged files: 0" in captured.out


def test_push_token_passes_through(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)

    push_mod.push_dataset(tmp_path, "user/repo", token="hf_xxx")

    [api] = _FakeHfApiCollector._INSTANCES
    assert api.token == "hf_xxx"


def test_push_token_resolved_from_env(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)

    push_mod.push_dataset(tmp_path, "user/repo")

    [api] = _FakeHfApiCollector._INSTANCES
    assert api.token == "hf_env"


def test_push_hf_not_installed(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    _populate(tmp_path)

    def _missing(name, package=None):
        if name == "huggingface_hub":
            raise ImportError("no module named huggingface_hub")
        return importlib.import_module(name, package)

    monkeypatch.setattr(push_mod.importlib, "import_module", _missing)

    with pytest.raises(ImportError) as excinfo:
        push_mod.push_dataset(tmp_path, "user/repo")

    assert "synthpix[hf]" in str(excinfo.value)


def test_push_token_never_logged(monkeypatch, tmp_path, capsys, caplog):
    _clear_env(monkeypatch)
    _install_fake_hub(monkeypatch)
    _populate(tmp_path)

    seen_log_messages: list[str] = []
    for level in ("info", "warning", "error", "debug"):
        original = getattr(push_mod.logger, level)

        def _spy(msg, *a, __orig=original, **kw):
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
