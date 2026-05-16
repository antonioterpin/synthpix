"""Live, network-touching tests for ``synthpix.hf.pull_dataset``.

Skipped by default. To run, set ``HF_TOKEN`` in the environment and pass
``--run-hf-live`` on the pytest command line, e.g.::

    HF_TOKEN=hf_xxx uv run pytest tests/hf/test_pull_live.py \
        -m hf_live --run-hf-live

The chosen fixture should be a small, stable, public Hugging Face dataset.
If you are unsure which dataset to use, leave this test as the placeholder
skip below and fill in a known-good identifier before flipping the switch.
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.hf_live
def test_pull_dataset_live_roundtrip(tmp_path, request):
    # Round-trip a small, stable, public dataset. Fill in
    # ``fixture_repo_id`` with a known-good public HF dataset before
    # flipping the gate; otherwise this test stays as a deliberate skip so
    # CI does not pin on a fixture that might be renamed upstream.
    if not request.config.getoption("--run-hf-live", default=False):
        pytest.skip("requires --run-hf-live")
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("requires HF_TOKEN in env")

    fixture_repo_id: str | None = None
    if fixture_repo_id is None:
        pytest.skip(
            "fill in a known-good public HF dataset before running"
        )

    from synthpix.hf import pull_dataset

    target = pull_dataset(fixture_repo_id, tmp_path / "data")

    files = [p for p in target.rglob("*") if p.is_file()]
    assert files, "Expected at least one file to land under local_dir"
