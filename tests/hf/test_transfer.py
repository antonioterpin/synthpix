"""Tests for ``synthpix.hf._transfer`` env-var toggling."""

from __future__ import annotations

import os
import threading
import time

from synthpix.hf import _transfer


_FLAG = "HF_HUB_ENABLE_HF_TRANSFER"


def test_enable_hf_transfer_sets_env_when_available(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    monkeypatch.setattr(_transfer, "_hf_transfer_available", lambda: True)

    assert _FLAG not in os.environ
    with _transfer.enable_hf_transfer():
        assert os.environ[_FLAG] == "1"
    assert _FLAG not in os.environ


def test_enable_hf_transfer_no_op_when_missing(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    monkeypatch.setattr(_transfer, "_hf_transfer_available", lambda: False)

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
    monkeypatch.setattr(_transfer, "_hf_transfer_available", lambda: True)

    with _transfer.enable_hf_transfer():
        assert os.environ[_FLAG] == "1"

    assert os.environ[_FLAG] == "0"


def test_no_op_branch_clears_stale_flag(monkeypatch):
    # If ``hf_transfer`` is missing but the environment still carries a
    # stale ``HF_HUB_ENABLE_HF_TRANSFER=1``, ``huggingface_hub`` will try
    # the accelerated path and fail. The context manager must clear it for
    # the duration of the block and restore it afterwards.
    monkeypatch.setenv(_FLAG, "1")
    monkeypatch.setattr(_transfer, "_hf_transfer_available", lambda: False)
    monkeypatch.setattr(
        _transfer.logger, "warning", lambda *a, **kw: None
    )

    with _transfer.enable_hf_transfer():
        assert _FLAG not in os.environ

    assert os.environ[_FLAG] == "1"


def test_nested_enable_is_reentrant(monkeypatch):
    # Nested calls must not let an inner exit pop the flag while an outer
    # context still expects it to be set.
    monkeypatch.delenv(_FLAG, raising=False)
    monkeypatch.setattr(_transfer, "_hf_transfer_available", lambda: True)

    with _transfer.enable_hf_transfer():
        assert os.environ[_FLAG] == "1"
        with _transfer.enable_hf_transfer():
            assert os.environ[_FLAG] == "1"
        # Inner exit must not pop the flag while the outer is still active.
        assert os.environ[_FLAG] == "1"
    assert _FLAG not in os.environ


def test_overlapping_threads_do_not_leak_flag(monkeypatch):
    # Two threads entering and leaving overlapping contexts must not leak
    # the flag or restore it in the wrong order.
    monkeypatch.delenv(_FLAG, raising=False)
    monkeypatch.setattr(_transfer, "_hf_transfer_available", lambda: True)

    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        with _transfer.enable_hf_transfer():
            time.sleep(0.01)
            assert os.environ[_FLAG] == "1"

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert _FLAG not in os.environ
