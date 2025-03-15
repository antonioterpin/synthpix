"""Query the JHTDB database for data."""
import jax.numpy as jnp
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData


# TODO: extend to 3d flow field and different axes.
def query_2d_flow_field(
    variable: str = "velocity",
    auth_token: str = "ch.ethz.aterpin-24c9605d",
    dataset_title: str = "channel",
    output_path: str = "./giverny_output",
    temporal_method: str = "none",
    spatial_method: str = "lag8",
    spatial_operator: str = "field",
    time: float = 1.0,
    nx: int = 1936,
    nz: int = 1216,
    y: float = 0.9,
    jax: bool = True,
) -> jnp.ndarray:
    """Query a 2d flow field from the JHTDB database.

    Query the flow field at a set of points evenly spaced over a 2D plane
    lying along one of the primary axes from the JHTDB database.

    Args:
        auth_token (str): The authorization token for the JHTDB database.
        dataset_title (str): The title of the dataset to be queried.
        output_path (str): The path to the output directory.
        variable (str): The variable to be queried.
        temporal_method (str): The temporal method to be used for the query.
        spatial_method (str): The spatial method to be used for the query.
        spatial_operator (str): The spatial operator to be used for the query.
        time (float): The time to be queried.
        nx (int): The number of points along the x-axis.
        nz (int): The number of points along the z-axis.

    Returns:
        jnp.ndarray: The flow field at the queried points.
    """
    # Argument checks using exceptions instead of asserts
    if nx < 1:
        raise ValueError("nx must be a positive integer.")
    if nz < 1:
        raise ValueError("nz must be a positive integer.")
    if y < 0 or y > 1:
        raise ValueError("y must be between 0 and 1.")

    # Instantiate the dataset.
    try:
        dataset = turb_dataset(
            dataset_title=dataset_title, output_path=output_path, auth_token=auth_token
        )
    except Exception as e:
        raise ValueError(f"The dataset could not be instantiated: {e}") from e

    # Generate the points with 0.0024 * jnp.pi spacing and split them into two sets.
    spacing = 0.0024 * jnp.pi
    x_points_1 = jnp.linspace(0.0, (nx // 2 - 1) * spacing, nx // 2, dtype=jnp.float64)
    x_points_2 = jnp.linspace(
        (nx // 2) * spacing, (nx - 1) * spacing, nx - nx // 2, dtype=jnp.float64
    )
    y_points = y
    z_points = jnp.linspace(0.0, (nz - 1) * spacing, nz, dtype=jnp.float64)

    points_1 = jnp.stack(
        [
            jnp.repeat(jnp.array(x_points_1), nz),
            jnp.full((nx // 2 * nz,), y_points),
            jnp.tile(jnp.array(z_points), nx // 2),
        ],
        axis=1,
    )

    # Query the data from JHTDB.
    try:
        result_1 = getData(
            dataset,
            variable,
            time,
            temporal_method,
            spatial_method,
            spatial_operator,
            points_1,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to query data from JHTDB: {e}") from e

    points_2 = jnp.stack(
        [
            jnp.repeat(jnp.array(x_points_2), nz),
            jnp.full(((nx - nx // 2) * nz,), y_points),
            jnp.tile(jnp.array(z_points), (nx - nx // 2)),
        ],
        axis=1,
    )
    # Query the data from JHTDB.
    try:
        result_2 = getData(
            dataset,
            variable,
            time,
            temporal_method,
            spatial_method,
            spatial_operator,
            points_2,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to query data from JHTDB: {e}") from e

    # Convert the result to a jax numpy array.
    result_1 = jnp.array(result_1)
    result_2 = jnp.array(result_2)

    # Concatenate the two results.
    result = jnp.concatenate([result_1, result_2], axis=0)

    # Reshape the result to a grid.
    result = result.reshape((nx, nz, 3))

    # Only keep the x and z components of the velocity.
    result = result[:, :, [0, 2]]

    return result
