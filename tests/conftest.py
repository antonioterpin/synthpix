"""Conftest.py for Sampler and Scheduler tests."""

from __future__ import annotations

import gc
import os
import uuid
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image

from synthpix.scheduler import HDF5FlowFieldScheduler

# ──────────────────────────────────────────────────────────────────────────────
# Helper: per-worker root directory
# ──────────────────────────────────────────────────────────────────────────────
_WORKER = os.getenv("PYTEST_XDIST_WORKER", "master")
ROOT = Path(os.getenv("PYTEST_SCRATCH_DIR", Path.cwd() / ".pytest_scratch"))
ROOT.mkdir(exist_ok=True)
WORKER_ROOT = ROOT / f"{_WORKER}_{datetime.now():%Y%m%d_%H%M%S}"
WORKER_ROOT.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Generic random HDF5 creator
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def generate_hdf5_file(tmp_path_factory):
    """Return a callable that writes an HDF5 file and yields its Path."""

    def _generate(stem: str, dims: dict[str, int]) -> Path:
        folder = tmp_path_factory.mktemp("hdf5")
        path = folder / f"{stem}_{uuid.uuid4().hex}.h5"

        with h5py.File(path, "w") as f:
            data = np.random.rand(
                dims["x_dim"], dims["y_dim"], dims["z_dim"], dims["features"]
            ).astype(np.float32)
            f.create_dataset("flow", data=data)
        return str(path)

    return _generate


# ──────────────────────────────────────────────────────────────────────────────
# Default dimensions
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def hdf5_test_dims() -> dict[str, int]:
    """Return default dimensions for HDF5 test files."""
    CI = os.getenv("CI") == "true"
    return {
        "x_dim": 128 if CI else 1536,
        "y_dim": 10 if CI else 100,
        "z_dim": 128 if CI else 2048,
        "features": 2,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Single temporary HDF5 file
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def temp_file(request, hdf5_test_dims, generate_hdf5_file):
    """Create a temporary HDF5 file with specified dimensions."""
    dims = getattr(request, "param", hdf5_test_dims)
    yield generate_hdf5_file(stem="flow_data", dims=dims)


# ──────────────────────────────────────────────────────────────────────────────
# Multiple HDF5 files
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def mock_hdf5_files(request, hdf5_test_dims, generate_hdf5_file):
    """Create multiple temporary HDF5 files with specified dimensions."""
    num_files = getattr(request, "param", 1)
    paths = [
        generate_hdf5_file(stem=f"flow_data_{i}", dims=hdf5_test_dims)
        for i in range(num_files)
    ]
    yield paths, hdf5_test_dims


# ──────────────────────────────────────────────────────────────────────────────
# Scheduler
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def temp_file_module(request, hdf5_test_dims, generate_hdf5_file):
    """Create a temporary HDF5 file for module scope tests."""
    dims = getattr(request, "param", hdf5_test_dims)
    yield generate_hdf5_file(stem="flow_data_module", dims=dims)


@pytest.fixture
def scheduler(temp_file_module, request):
    """Create a scheduler for module scope tests."""
    randomize = False
    loop = False

    if hasattr(request, "param"):
        randomize = request.param.get("randomize", False)
        loop = request.param.get("loop", False)

    yield HDF5FlowFieldScheduler(
        [temp_file_module], randomize=randomize, loop=loop
    )


# ──────────────────────────────────────────────────────────────────────────────
# Text file
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def temp_txt_file(tmp_path):
    """Create a temporary text file with some content."""
    path = tmp_path / "mock_data.txt"
    path.write_text("This is a test file.")
    yield [str(path)]


# ──────────────────────────────────────────────────────────────────────────────
# Numpy helpers
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def numpy_test_dims():
    """Return default dimensions for Numpy test files."""
    return {"height": 64, "width": 64}


@pytest.fixture
def mock_numpy_files(tmp_path, numpy_test_dims, request):
    """Create multiple temporary Numpy files with random data."""
    param = getattr(request, "param", 2)
    
    if isinstance(param, dict):
        num_files = param.get("num_files", 2)
        dims = param.get("dims", numpy_test_dims)
        h, w = dims["height"], dims["width"]
    else:
        num_files = param
        dims = numpy_test_dims
        h, w = dims["height"], dims["width"]

    paths = []
    for t in range(1, num_files + 1):
        img0 = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
        img1 = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
        Image.fromarray(img0).save(tmp_path / f"img_{t - 1}.jpg")
        Image.fromarray(img1).save(tmp_path / f"img_{t}.jpg")

        flow = np.random.rand(h, w, 2).astype(np.float32)
        flow_path = tmp_path / f"flow_{t}.npy"
        np.save(flow_path, flow)
        paths.append(flow_path)

    yield [str(p) for p in paths], dims


# ──────────────────────────────────────────────────────────────────────────────
# .mat helpers
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def mat_test_dims():
    """Return default dimensions for .mat test files."""
    return {"height": 64, "width": 64}


@pytest.fixture
def mock_mat_files(tmp_path, mat_test_dims, request):
    """Create multiple temporary .mat files with random data."""
    param = getattr(request, "param", 2)
    
    if isinstance(param, dict):
        num_files = param.get("num_files", 2)
        dims = param.get("dims", mat_test_dims)
        h, w = dims["height"], dims["width"]
    else:
        num_files = param
        dims = mat_test_dims
        h, w = mat_test_dims["height"], mat_test_dims["width"]

    paths = []
    for t in range(1, num_files + 1):
        mat_path = tmp_path / f"flow_{t:04d}.mat"
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

        # write fake MATLAB header
        header = (
            (
                f"MATLAB 7.3 MAT-file, Platform: Python-h5py, "
                f"Created on {datetime.now():%c}"
            )
            .encode("ascii")
            .ljust(116, b" ")
        )
        header += b" " * (512 - 116)
        with open(mat_path, "r+b") as fp:
            fp.write(header)

        paths.append(mat_path)

    yield [str(p) for p in paths], dims


# ──────────────────────────────────────────────────────────────────────────────
# Collection modifier
# ──────────────────────────────────────────────────────────────────────────────
def pytest_collection_modifyitems(config, items):
    """Skip tests unless explicitly selected with -m run_explicitly."""
    if config.getoption("-m") and "run_explicitly" in config.getoption("-m"):
        return
    skip = pytest.mark.skip(
        reason="Skipped unless explicitly selected with -m run_explicitly"
    )
    for item in items:
        if "run_explicitly" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def clear_after_test():
    """Clear JAX backends and free memory after each test."""
    yield  # --- run the test ---
    # Free Python-side references
    gc.collect()

    # Finish goggles session
    try:
        import goggles as gg

        gg.finish()
    except Exception:
        # Silently continue if goggles is not available or fails
        pass
