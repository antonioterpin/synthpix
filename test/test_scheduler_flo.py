# tests/test_scheduler_flo.py
import os
import struct

import numpy as np
import pytest

from synthpix.scheduler import FloFlowFieldScheduler

TAG = FloFlowFieldScheduler.TAG_FLOAT


def _write_flo(path, data, tag=TAG, truncate=False):
    """Write a Middlebury .flo file (optionally truncated for error cases)."""
    h, w, _ = data.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<f", tag))
        f.write(struct.pack("<i", w))
        f.write(struct.pack("<i", h))
        payload = data.astype(np.float32).tobytes(order="C")
        if truncate:
            payload = payload[: len(payload) // 2]
        f.write(payload)


@pytest.fixture
def flo_file(tmp_path):
    arr = np.random.rand(4, 5, 2).astype(np.float32)
    p = tmp_path / "flow_valid.flo"
    _write_flo(p, arr)
    return str(p), arr


@pytest.fixture
def mock_flo_files(request, tmp_path):
    n = request.param
    h, w = 3, 6
    files = []
    for i in range(n):
        arr = np.random.rand(h, w, 2).astype(np.float32)
        p = tmp_path / f"flow_{i}.flo"
        _write_flo(p, arr)
        files.append(str(p))
    return files, {"height": h, "width": w}


def test_load_file_roundtrip(flo_file):
    path, original = flo_file
    sch = FloFlowFieldScheduler([path])
    loaded = sch.load_file(path)
    assert np.allclose(loaded, original)
    assert loaded.shape == (4, 5, 2)


def test_wrong_tag_raises(tmp_path, flo_file):
    good_path, _ = flo_file
    bad_path = tmp_path / "bad_tag.flo"
    _write_flo(bad_path, np.zeros((2, 2, 2), np.float32), tag=123.0)
    sch = FloFlowFieldScheduler([good_path])
    with pytest.raises(ValueError, match="wrong tag"):
        sch.load_file(str(bad_path))


def test_illegal_dimensions_raises(tmp_path, flo_file):
    good_path, _ = flo_file
    bad_path = tmp_path / "bad_dim.flo"
    _write_flo(bad_path, np.zeros((2, 0, 2), np.float32))
    sch = FloFlowFieldScheduler([good_path])
    with pytest.raises(ValueError, match="illegal dimensions"):
        sch.load_file(str(bad_path))


def test_unexpected_eof_raises(tmp_path, flo_file):
    good_path, _ = flo_file
    bad_path = tmp_path / "eof.flo"
    _write_flo(bad_path, np.random.rand(4, 4, 2), truncate=True)
    sch = FloFlowFieldScheduler([good_path])
    with pytest.raises(IOError, match="unexpected EOF"):
        sch.load_file(str(bad_path))


@pytest.mark.parametrize("mock_flo_files", [3], indirect=True)
def test_iteration_and_shape(mock_flo_files):
    files, dims = mock_flo_files
    sch = FloFlowFieldScheduler.from_config(
        {"scheduler_files": files, "randomize": False, "loop": False}
    )
    flows = list(sch)
    assert len(flows) == len(files) * dims["width"]
    for flow in flows:
        assert flow.shape == (dims["height"], dims["width"], 2)


@pytest.mark.parametrize("mock_flo_files", [4], indirect=True)
def test_get_batch(mock_flo_files):
    files, dims = mock_flo_files
    sch = FloFlowFieldScheduler(files, loop=False)
    total = len(files) * dims["width"]
    seen = 0
    while True:
        try:
            batch = sch.get_batch(1)
            assert batch.shape == (1, dims["height"], dims["width"], 2)
            seen += 1
        except StopIteration:
            break
    assert seen == total


@pytest.mark.parametrize("mock_flo_files", [2], indirect=True)
def test_looping_restarts_after_exhaustion(mock_flo_files):
    files, dims = mock_flo_files
    sch = FloFlowFieldScheduler(files, loop=True)
    total = len(files) * dims["width"]

    first_pass = [next(sch) for _ in range(total)]
    second_pass = [next(sch) for _ in range(total)]

    assert len(first_pass) == len(second_pass) == total
    # Ensure we really restarted from the beginning
    assert np.allclose(first_pass[0], second_pass[0])


def test_directory_constructor(tmp_path):
    h, w = 2, 2
    for i in range(3):
        _write_flo(tmp_path / f"f{i}.flo", np.random.rand(h, w, 2).astype(np.float32))
    sch = FloFlowFieldScheduler(str(tmp_path))
    assert sorted(os.path.basename(p) for p in sch.file_list) == sorted(
        os.listdir(tmp_path)
    )


def test_invalid_extension_rejected(tmp_path):
    bad = tmp_path / "flow.txt"
    bad.write_text("fake")
    with pytest.raises(ValueError, match=r"\.flo"):
        FloFlowFieldScheduler([str(bad)])


@pytest.mark.parametrize("mock_flo_files", [2], indirect=True)
def test_get_flow_fields_shape_ok(mock_flo_files):
    files, dims = mock_flo_files
    sch = FloFlowFieldScheduler(files)
    assert sch.get_flow_fields_shape() == (dims["height"], dims["width"], 2)


@pytest.mark.parametrize("mock_flo_files", [1], indirect=True)
def test_get_flow_fields_shape_bad_tag(tmp_path, mock_flo_files):
    files, dims = mock_flo_files
    bad = tmp_path / "bad_tag.flo"
    h, w = dims["height"], dims["width"]
    _write_flo(bad, np.zeros((h, w, 2), np.float32), tag=123.0)  # helper from file
    sch = FloFlowFieldScheduler([str(bad)] + files)
    with pytest.raises(ValueError, match="Bad .*tag"):
        sch.get_flow_fields_shape()


@pytest.mark.parametrize("mock_flo_files", [2], indirect=True)
def test_loop_restarts(mock_flo_files):
    files, dims = mock_flo_files
    sch = FloFlowFieldScheduler(files, loop=True)
    total = len(files) * dims["width"]
    first_epoch = [next(sch) for _ in range(total)]
    second_epoch = [next(sch) for _ in range(total)]
    assert np.allclose(first_epoch[0], second_epoch[0])


def test_load_file_missing_file(tmp_path, flo_file):
    valid_path, _ = flo_file
    sch = FloFlowFieldScheduler([valid_path])
    missing = tmp_path / "missing.flo"
    with pytest.raises(FileNotFoundError):
        sch.load_file(str(missing))
