"""Live, network-touching tests for ``synthpix.hf.push_dataset``.

Skipped by default. To run, set ``HF_TOKEN`` in the environment and pass
``--run-hf-live`` on the pytest command line, e.g.::

    HF_TOKEN=hf_xxx uv run pytest tests/hf/test_push_live.py \
        -m hf_live --run-hf-live

The test pushes a tiny fixture to a unique private repo derived from
the authenticated user, pulls it back, asserts byte equality, and
cleans up the repo in a finalizer.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime

import h5py
import numpy as np
import pytest


@pytest.mark.hf_live
def test_push_dataset_live_roundtrip(tmp_path, request):
    if not request.config.getoption("--run-hf-live", default=False):
        pytest.skip("requires --run-hf-live")
    token = os.environ.get("HF_TOKEN")
    if not token:
        pytest.skip("requires HF_TOKEN in env")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        pytest.skip("huggingface_hub is not installed")

    api = HfApi(token=token)
    try:
        username = api.whoami()["name"]
    except Exception as exc:
        pytest.skip(f"could not resolve username via whoami(): {exc}")

    repo_id = f"{username}/synthpix-ci-push-{uuid.uuid4().hex[:8]}"

    # Build a tiny train/flow_0001.mat fixture with a MATLAB v7.3 header.
    train = tmp_path / "src" / "train"
    train.mkdir(parents=True)
    mat_path = train / "flow_0001.mat"
    h = w = 8
    with h5py.File(mat_path, "w", libver="latest", userblock_size=512) as f:
        f.create_dataset(
            "I0",
            data=np.random.randint(0, 255, size=(h, w), dtype=np.uint8),
        )
        f.create_dataset(
            "I1",
            data=np.random.randint(0, 255, size=(h, w), dtype=np.uint8),
        )
        f.create_dataset(
            "V", data=np.random.rand(h, w, 2).astype(np.float32)
        )
    header = (
        f"MATLAB 7.3 MAT-file, Platform: Python-h5py, "
        f"Created on {datetime.now():%c}"
    ).encode("ascii").ljust(116, b" ")
    header += b" " * (512 - 116)
    with open(mat_path, "r+b") as fp:
        fp.write(header)

    original_bytes = mat_path.read_bytes()

    def _cleanup():
        try:
            api.delete_repo(
                repo_id=repo_id, repo_type="dataset", missing_ok=True
            )
        except Exception:
            pass

    request.addfinalizer(_cleanup)

    from synthpix.hf import pull_dataset, push_dataset

    push_dataset(
        tmp_path / "src",
        repo_id,
        private=True,
        token=token,
    )

    dst = tmp_path / "dst"
    pull_dataset(repo_id, dst, token=token)

    pulled = dst / "train" / "flow_0001.mat"
    assert pulled.exists(), f"pulled file missing: {pulled}"
    assert pulled.read_bytes() == original_bytes
