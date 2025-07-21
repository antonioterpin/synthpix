import re
from test.test_sampler import dummy_img_gen_fn

import jax.numpy as jnp
import numpy as np
import pytest

from synthpix.sampler import SyntheticImageSampler
from synthpix.utils import load_configuration

sampler_config = load_configuration("config/test_data.yaml")


@pytest.fixture
def mock_invalid_mask_file(tmp_path, numpy_test_dims, request):
    """Create and save a numpy invalid mask to a temporary file."""
    shape = numpy_test_dims["height"], numpy_test_dims["width"]
    value = getattr(request, "param", 2)

    path = tmp_path / "invalid_mask.npy"
    arr = np.full(shape, value, dtype=float)
    np.save(path, arr)

    yield str(path), numpy_test_dims


@pytest.fixture
def mock_mask_file(tmp_path, numpy_test_dims):
    """Create and save a numpy mask to a temporary file."""
    shape = numpy_test_dims["height"], numpy_test_dims["width"]

    path = tmp_path / "mask.npy"
    # Create a mask with some ones and zeros
    arr = np.random.choice([0, 1], size=shape, p=[0.5, 0.5]).astype(int)
    np.save(path, arr)

    yield str(path), numpy_test_dims


@pytest.mark.parametrize(
    "mask",
    [
        1,
        [],
        {},
        jnp.ones((256, 256)),
        jnp.full((256, 256), 0.5),
    ],
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_mask_type(mask, scheduler):
    """Test that invalid mask raises a ValueError."""
    with pytest.raises(
        ValueError, match="mask must be a string representing the mask path."
    ):
        config = sampler_config.copy()
        config["mask"] = mask
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
        )


@pytest.mark.parametrize(
    "mask",
    [
        "invalid_mask_path",
        "non_existent_mask_path.png",
        "mask_with_invalid_format.txt",
        "mask_with_invalid_format.jpg",
        "mask_with_invalid_format.jpeg",
    ],
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_mask_path(mask, scheduler):
    """Test that invalid mask path raises a ValueError."""
    with pytest.raises(ValueError, match=f"Mask file {mask} does not exist."):
        config = sampler_config.copy()
        config["mask"] = mask
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
        )


@pytest.mark.parametrize("image_shape", [(256, 256), (128, 128), (512, 512)])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_mask_shape(scheduler, mock_invalid_mask_file, image_shape):
    """Test that mask with invalid shape raises a ValueError."""
    # Create a dummy mask with an invalid shape
    mask = jnp.array(np.load(mock_invalid_mask_file[0]))
    if mask.shape != image_shape:
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Mask shape {mask.shape} does not match image shape " f"{image_shape}."
            ),
        ):
            config = sampler_config.copy()
            config["mask"] = mock_invalid_mask_file[0]
            config["image_shape"] = image_shape
            SyntheticImageSampler.from_config(
                scheduler=scheduler,
                img_gen_fn=dummy_img_gen_fn,
                config=config,
            )


@pytest.mark.parametrize(
    "mock_invalid_mask_file", [1.1, -1, 2, 0.5, 1e-10], indirect=True
)
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_invalid_mask_values(scheduler, mock_invalid_mask_file):
    """Test that mask with invalid values raises a ValueError."""
    # Create a dummy mask with an invalid shape
    mask = jnp.array(np.load(mock_invalid_mask_file[0]))

    with pytest.raises(ValueError, match="Mask must only contain 0 and 1 values."):
        config = sampler_config.copy()
        config["mask"] = mock_invalid_mask_file[0]
        config["image_shape"] = mask.shape  # Use the shape of the mask
        SyntheticImageSampler.from_config(
            scheduler=scheduler,
            img_gen_fn=dummy_img_gen_fn,
            config=config,
        )


@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_mask_is_right(scheduler, mock_mask_file):
    """Test that correct mask gets loaded."""
    # Create a dummy mask with a valid shape
    mask = jnp.array(np.load(mock_mask_file[0]))

    config = sampler_config.copy()
    config["mask"] = mock_mask_file[0]
    config["image_shape"] = mask.shape  # Use the shape of the mask
    sampler = SyntheticImageSampler.from_config(
        scheduler=scheduler,
        img_gen_fn=dummy_img_gen_fn,
        config=config,
    )

    assert isinstance(sampler.mask, jnp.ndarray)
    assert sampler.mask.shape == mask.shape
    assert jnp.array_equal(
        sampler.mask, mask
    ), "Mask loaded from file does not match the expected mask."
