import jax.numpy as jnp
import pytest

from src.sym.dataset_query import query_2d_flow_field


# Dummy implementations to bypass external dependencies.
class DummyDataset:
    pass


def dummy_turb_dataset(dataset_title, output_path, auth_token):
    return DummyDataset()


def dummy_getData(
    dataset, variable, time, temporal_method, spatial_method, spatial_operator, points
):
    # Return a dummy flow field (same shape as points)
    return jnp.ones(points.shape)


# Patch the external dependencies.
@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    monkeypatch.setattr("src.sym.dataset_query.turb_dataset", dummy_turb_dataset)
    monkeypatch.setattr("src.sym.dataset_query.getData", dummy_getData)


@pytest.mark.parametrize("nx, nz", [(8, 4), (16, 8), (32, 16)])
def test_query_valid(nx, nz):
    """Test that a valid query returns a flow field with the correct shape."""
    result = query_2d_flow_field(nx=nx, nz=nz)
    expected_shape = (nx * nz, 3)
    assert result.shape == expected_shape


@pytest.mark.parametrize("nx", [0, -1])
def test_invalid_nx(nx):
    """Test that passing an invalid nx value raises a ValueError."""
    with pytest.raises(ValueError, match="nx must be a positive integer."):
        query_2d_flow_field(nx=nx)


@pytest.mark.parametrize("nz", [0, -5])
def test_invalid_nz(nz):
    """Test that passing an invalid nz value raises a ValueError."""
    with pytest.raises(ValueError, match="nz must be a positive integer."):
        query_2d_flow_field(nz=nz)


@pytest.mark.parametrize("y", [-0.1, 1.1])
def test_invalid_y(y):
    """Test that passing an invalid y value raises a ValueError."""
    with pytest.raises(ValueError, match="y must be between 0 and 1."):
        query_2d_flow_field(y=y)
