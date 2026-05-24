"""Tests for the ``--push-to`` opt-in flag on ``scripts/download_piv_1.py``.

The download/convert/split steps are short-circuited so that no Google
Drive traffic occurs. ``synthpix.hf.push_dataset`` is patched to a
recording stub.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "download_piv_1.py"


def _load_script_module() -> ModuleType:
    """Import ``download_piv_1.py`` as a module without running ``__main__``.

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
    try:
        return _load_script_module()
    except ImportError as exc:
        pytest.skip(f"download_piv_1 deps not installed: {exc}")


def _make_args(**overrides):
    base = dict(
        push_to=None,
        push_public=False,
        allow_public=False,
        push_token=None,
        no_push_card=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_maybe_push_no_op_without_push_to(monkeypatch, script, tmp_path):
    called: list = []

    def _spy(*args, **kwargs):
        called.append((args, kwargs))
        return "sha"

    # Patch the push_dataset symbol accessed via ``from synthpix.hf import ...``
    # at call time. We replace the attribute on the imported ``synthpix.hf``
    # package so the lazy import inside ``_maybe_push`` picks up the spy.
    import synthpix.hf as hf_pkg

    monkeypatch.setattr(hf_pkg, "push_dataset", _spy)

    args = _make_args()
    script._maybe_push(args, tmp_path)

    assert called == []


def test_maybe_push_invokes_push_dataset_with_card(
    monkeypatch, script, tmp_path
):
    captured: dict = {}

    def _spy(*, local_dir, repo_id, **kwargs):
        captured["local_dir"] = local_dir
        captured["repo_id"] = repo_id
        captured.update(kwargs)
        return "sha-123"

    import synthpix.hf as hf_pkg

    monkeypatch.setattr(hf_pkg, "push_dataset", _spy)

    args = _make_args(push_to="user/my-piv")
    script._maybe_push(args, tmp_path)

    assert captured["repo_id"] == "user/my-piv"
    assert captured["local_dir"] == tmp_path
    meta = captured["card_meta"]
    assert meta is not None
    assert meta.source_url == script._PUSH_SOURCE_URL
    assert "Dense motion estimation" in meta.citation


def test_maybe_push_no_card_passes_none(monkeypatch, script, tmp_path):
    captured: dict = {}

    def _spy(*, local_dir, repo_id, **kwargs):
        captured.update(kwargs)
        return "sha"

    import synthpix.hf as hf_pkg

    monkeypatch.setattr(hf_pkg, "push_dataset", _spy)

    args = _make_args(push_to="user/my-piv", no_push_card=True)
    script._maybe_push(args, tmp_path)

    assert captured["card_meta"] is None


def test_push_public_requires_allow_public_exits_nonzero(tmp_path):
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
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "--allow-public" in proc.stderr
