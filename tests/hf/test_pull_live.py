"""Live, network-touching tests for ``synthpix.hf.pull_dataset``.

Skipped by default. To run, set ``HF_TOKEN`` in the environment and pass
``--run-hf-live`` on the pytest command line, e.g.::

    HF_TOKEN=hf_xxx uv run pytest tests/hf/test_pull_live.py \
        -m hf_live --run-hf-live

The fixture repo defaults to ``lhoestq/demo1`` — a tiny, stable, public
demo dataset on the Hub that has been around since 2021. Override with
``SYNTHPIX_HF_LIVE_REPO=<owner>/<dataset>`` to point at a private synthpix
mirror or any other dataset of your choice.
"""

from __future__ import annotations

import os

import pytest

# TODO(synthpix#259): swap to a synthpix-owned mirror once one is published
# so the live smoke test no longer pins on an upstream-owned repo.
_DEFAULT_LIVE_REPO = "lhoestq/demo1"


@pytest.mark.hf_live
def test_pull_dataset_live_roundtrip(tmp_path):
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("requires HF_TOKEN in env")

    repo_id = os.environ.get("SYNTHPIX_HF_LIVE_REPO", _DEFAULT_LIVE_REPO)

    from synthpix.hf import pull_dataset

    target = pull_dataset(repo_id, tmp_path / "data")

    files = [p for p in target.rglob("*") if p.is_file()]
    assert files, "Expected at least one file to land under local_dir"
