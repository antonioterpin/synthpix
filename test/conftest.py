"""Conftest.py for Sampler and Scheduler tests."""
import os
import tempfile

import h5py
import numpy as np
import pytest

from src.sym.scheduler import HDF5FlowFieldScheduler


@pytest.fixture(scope="module")
def hdf5_test_dims(request):
    """Fixture to provide default dimensions for HDF5 files.

    The dimensions are set based on the CI environment variable,
    but can be overridden by passing parameters in the request.
    The parameters need to be a dictionary with keys:
    'x_dim', 'y_dim', 'z_dim', and 'features'.
    """
    CI = os.environ.get("CI") == "true"
    default_dims = {
        "x_dim": 1536 if not CI else 128,
        "y_dim": 100 if not CI else 10,
        "z_dim": 2048 if not CI else 128,
        "features": 3,
    }
    return default_dims


@pytest.fixture(scope="module")
def generate_hdf5_file():
    """Fixture to generate a temporary HDF5 file with random data."""

    def _generate(filename="test.h5", dims=None):
        path = os.path.join(tempfile.gettempdir(), filename)
        with h5py.File(path, "w") as f:
            data = np.random.rand(
                dims["x_dim"], dims["y_dim"], dims["z_dim"], dims["features"]
            ).astype(np.float32)
            f.create_dataset("flow", data=data)
        return path

    return _generate


@pytest.fixture
def mock_hdf5_files(request, hdf5_test_dims, generate_hdf5_file, num_files=1):
    """Fixture to create multiple HDF5 files for testing."""
    if isinstance(request.param, int):
        num_files = request.param

    files = []
    try:
        for i in range(num_files):
            path = generate_hdf5_file(f"test_file_{i}.h5", dims=hdf5_test_dims)
            files.append(path)
        yield files, hdf5_test_dims
    finally:
        for f in files:
            if os.path.exists(f):
                os.remove(f)


@pytest.fixture(scope="module")
def temp_file(request, generate_hdf5_file, hdf5_test_dims):
    """Fixture to create a temporary HDF5 file for testing."""
    if hasattr(request, "param"):
        dims = request.param
    else:
        dims = hdf5_test_dims
    filename = "mock_data_tmp.h5" #TODO change name
    path = generate_hdf5_file(filename, dims=dims)
    try:
        yield path
    finally:
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture
def temp_txt_file(request):
    """Fixture to create a temporary text file."""
    filename = "mock_data.txt"
    path = os.path.join(tempfile.gettempdir(), filename)
    with open(path, "w") as f:
        f.write("This is a test file.")
    try:
        yield [path]
    finally:
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture(scope="module")
def scheduler(temp_file, request):
    """Fixture to create an HDF5FlowFieldScheduler for testing."""
    # Default parameters for the scheduler
    randomize = (
        request.param.get("randomize", False) if hasattr(request, "param") else False
    )
    loop = request.param.get("loop", False) if hasattr(request, "param") else False

    # Create the scheduler using the temporary HDF5 file
    scheduler_instance = HDF5FlowFieldScheduler(
        [temp_file], randomize=randomize, loop=loop
    )
    return scheduler_instance
