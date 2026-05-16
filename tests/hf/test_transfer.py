"""Tests for ``synthpix.hf._transfer`` env-var toggling."""

from __future__ import annotations

import os

from synthpix.hf import _transfer


_FLAG = "HF_HUB_ENABLE_HF_TRANSFER"


def test_enable_hf_transfer_sets_env_when_available(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    monkeypatch.setattr(
        _transfer, "_hf_transfer_available", lambda: True
    )

    assert _FLAG not in os.environ
    with _transfer.enable_hf_transfer():
        assert os.environ[_FLAG] == "1"
    assert _FLAG not in os.environ


def test_enable_hf_transfer_no_op_when_missing(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    monkeypatch.setattr(
        _transfer, "_hf_transfer_available", lambda: False
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        _transfer.logger,
        "warning",
        lambda msg, *a, **kw: warnings.append(msg),
    )

    with _transfer.enable_hf_transfer():
        # Env var must not be set when the library is unavailable.
        assert _FLAG not in os.environ

    assert _FLAG not in os.environ
    assert warnings, "Expected a warning when hf_transfer is missing"
    assert "hf_transfer" in warnings[0]


def test_enable_hf_transfer_restores_existing_value(monkeypatch):
    monkeypatch.setenv(_FLAG, "0")
    monkeypatch.setattr(
        _transfer, "_hf_transfer_available", lambda: True
    )

    with _transfer.enable_hf_transfer():
        assert os.environ[_FLAG] == "1"

    assert os.environ[_FLAG] == "0"
