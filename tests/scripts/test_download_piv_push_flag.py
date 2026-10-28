"""Tests for the ``--push-to`` opt-in flag on
``scripts/download_piv_1.py``.

The download/convert/split steps are short-circuited so that no
Google Drive traffic occurs. ``synthpix.hf.push_dataset`` is patched
to a recording stub.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "download_piv_1.py"


def _load_script_module() -> ModuleType:
    """Import ``download_piv_1.py`` as a module without running
    ``__main__``.

    The script lives under ``scripts/`` and imports a sibling ``utils``
    module via ``from utils import ...``; we therefore add the script's
    parent directory to ``sys.path`` before loading.

    Returns:
        ModuleType: The loaded module object.
    """
    scripts_dir = str(_SCRIPT_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    spec = importlib.util.spec_from_file_location(
        "download_piv_1_test", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    """Script.

    Returns:
        The result used by the test assertions.
    """
    try:
        return _load_script_module()
    except ImportError as exc:
        pytest.skip(f"download_piv_1 deps not installed: {exc}")


def _make_args(**overrides: Any) -> Any:
    """Make args.

    Args:
        **overrides: Keyword arguments accepted by the test double.

    Returns:
        The value produced for the test scenario.
    """
    base = dict(
        push_to=None,
        push_public=False,
        allow_public=False,
        push_token=None,
        no_push_card=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_maybe_push_no_op_without_push_to(
    monkeypatch: pytest.MonkeyPatch, script: Any, tmp_path: Path
) -> None:
    """Verify maybe push no op without push to.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        script: Script used by this test.
        tmp_path: Temporary directory for this test.
    """
    called: list = []

    def _spy(*args: Any, **kwargs: Any) -> str:
        """Spy.

        Args:
            *args: Positional arguments accepted by the test double.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        called.append((args, kwargs))
        return "sha"

    # Patch the push_dataset symbol accessed via ``from synthpix.hf import ...``
    # at call time. We replace the attribute on the imported ``synthpix.hf``
    # package so the lazy import inside ``_maybe_push`` picks up the spy.
    import synthpix.hf as hf_pkg  # noqa: PLC0415 - optional or intentionally lazy import

    monkeypatch.setattr(hf_pkg, "push_dataset", _spy)

    args = _make_args()
    script._maybe_push(args, tmp_path)

    assert called == []


def test_maybe_push_invokes_push_dataset_with_card(
    monkeypatch: pytest.MonkeyPatch, script: Any, tmp_path: Path
) -> None:
    """Verify maybe push invokes push dataset with card.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        script: Script used by this test.
        tmp_path: Temporary directory for this test.
    """
    captured: dict = {}

    def _spy(*, local_dir: Path, repo_id: str, **kwargs: Any) -> str:
        """Spy.

        Args:
            local_dir: Local dir for this test case.
            repo_id: Repo id for this test case.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        captured["local_dir"] = local_dir
        captured["repo_id"] = repo_id
        captured.update(kwargs)
        return "sha-123"

    import synthpix.hf as hf_pkg  # noqa: PLC0415 - optional or intentionally lazy import

    monkeypatch.setattr(hf_pkg, "push_dataset", _spy)

    args = _make_args(push_to="user/my-piv")
    script._maybe_push(args, tmp_path)

    assert captured["repo_id"] == "user/my-piv"
    assert captured["local_dir"] == tmp_path
    meta = captured["card_meta"]
    assert meta is not None
    assert meta.source_url == script._PUSH_SOURCE_URL
    assert "Dense motion estimation" in meta.citation


def test_maybe_push_no_card_passes_none(
    monkeypatch: pytest.MonkeyPatch, script: Any, tmp_path: Path
) -> None:
    """Verify maybe push no card passes none.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        script: Script used by this test.
        tmp_path: Temporary directory for this test.
    """
    captured: dict = {}

    def _spy(*, local_dir: Path, repo_id: str, **kwargs: Any) -> str:
        """Spy.

        Args:
            local_dir: Local dir for this test case.
            repo_id: Repo id for this test case.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        captured.update(kwargs)
        return "sha"

    import synthpix.hf as hf_pkg  # noqa: PLC0415 - optional or intentionally lazy import

    monkeypatch.setattr(hf_pkg, "push_dataset", _spy)

    args = _make_args(push_to="user/my-piv", no_push_card=True)
    script._maybe_push(args, tmp_path)

    assert captured["card_meta"] is None


def test_push_public_requires_allow_public_exits_nonzero(
    tmp_path: Path,
) -> None:
    """Verify push public requires allow public exits nonzero.

    Args:
        tmp_path: Temporary directory for this test.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--out-dir",
            str(out_dir),
            "--push-to",
            "user/repo",
            "--push-public",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "--allow-public" in proc.stderr
