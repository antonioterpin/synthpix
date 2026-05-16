"""Tests for ``synthpix.hf.auth`` token resolution."""

from __future__ import annotations

import sys

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

    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub.HfFolder, "get_token", lambda: "cached-token"
    )

    result = auth.resolve_token()

    assert result == "cached-token"


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
