"""Tests for ``synthpix.hf.auth`` token resolution."""

from __future__ import annotations

import sys
import types

import pytest

from synthpix.hf import auth


def _clear_env(monkeypatch):
    """Remove every HF-related env var to start each test from a blank slate."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_HUB_TOKEN", raising=False)


def test_resolve_token_prefers_explicit_argument(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "env-token")
    monkeypatch.setenv("HF_HUB_TOKEN", "hub-token")

    result = auth.resolve_token("explicit-token")

    assert result == "explicit-token"


def test_resolve_token_uses_hf_token_when_no_explicit(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "env-token")
    monkeypatch.setenv("HF_HUB_TOKEN", "hub-token")

    result = auth.resolve_token()

    assert result == "env-token"


def test_resolve_token_uses_hf_hub_token_when_hf_token_missing(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HF_HUB_TOKEN", "hub-token")

    result = auth.resolve_token()

    assert result == "hub-token"


def test_resolve_token_ignores_empty_strings(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("HF_HUB_TOKEN", "hub-token")

    result = auth.resolve_token(explicit="")

    assert result == "hub-token"


def test_resolve_token_falls_back_to_cached(monkeypatch):
    pytest.importorskip("huggingface_hub")
    _clear_env(monkeypatch)

    import huggingface_hub  # noqa: PLC0415

    monkeypatch.setattr(
        huggingface_hub, "get_token", lambda: "cached-token", raising=False
    )

    result = auth.resolve_token()

    assert result == "cached-token"


def test_resolve_token_uses_modern_get_token_without_hffolder(monkeypatch):
    """Regression: hub 1.x removed ``HfFolder``; ``get_token`` must be used.

    Installs a fake ``huggingface_hub`` module that exposes ``get_token``
    but NOT ``HfFolder`` (mirroring huggingface_hub >= 1.0). The old
    ``_cached_token`` did ``from huggingface_hub import HfFolder`` which
    raised inside the lazy ``__getattr__`` and was swallowed, so the
    cached token was silently never found.
    """
    _clear_env(monkeypatch)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.get_token = lambda: "cli-cached-token"  # type: ignore[attr-defined]
    assert not hasattr(fake_hub, "HfFolder")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    result = auth.resolve_token()

    assert result == "cli-cached-token"


def test_resolve_token_cached_empty_string_is_ignored(monkeypatch):
    """An empty cached token must be treated as missing (-> ``None``)."""
    _clear_env(monkeypatch)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.get_token = lambda: ""  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    assert auth.resolve_token() is None


def test_resolve_token_returns_none_when_nothing_set(monkeypatch):
    _clear_env(monkeypatch)

    # Force the lazy import to fail so we exercise the safe branch when
    # ``huggingface_hub`` is missing and no env/argument is provided.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    result = auth.resolve_token()

    assert result is None


def test_resolve_token_does_not_raise_when_hub_missing(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    # Should not raise even though we have nothing to fall back to.
    result = auth.resolve_token()

    assert result is None
