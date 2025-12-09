import re
from synthpix.types import ImageGenerationSpecification

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from synthpix.data_generate import (
    generate_images_from_flow,
    input_check_gen_img_from_flow,
)
from synthpix.sampler import SyntheticImageSampler

# Import existing modules
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
            config=config,
        )


@pytest.mark.parametrize("image_shape", [(256, 256), (128, 128), (512, 512)])
@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
@pytest.mark.parametrize("mock_invalid_mask_file", [0.0, 1.0, 0.0], indirect=True)
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
            config=config,
        )


@pytest.mark.parametrize(
    "scheduler", [{"randomize": False, "loop": False}], indirect=True
)
def test_mask_is_correct(scheduler, mock_mask_file):
    """Test that correct mask gets loaded."""
    # Create a dummy mask with a valid shape
    mask = jnp.array(np.load(mock_mask_file[0]))

    config = sampler_config.copy()
    config["mask"] = mock_mask_file[0]
    config["image_shape"] = mask.shape  # Use the shape of the mask
    sampler = SyntheticImageSampler.from_config(
        scheduler=scheduler,
        config=config,
    )

    assert isinstance(sampler.mask_images, jnp.ndarray)
    assert sampler.mask_images.shape == mask.shape
    assert jnp.array_equal(
        sampler.mask_images, mask
    ), "Mask loaded from file does not match the expected mask."


def test_input_check_gen_img_from_flow_logs_mask(monkeypatch):
    """Test that the input_check_gen_img_from_flow function logs the mask shape."""
    flow_field = jnp.zeros((1, 8, 8, 2))
    image_shape = (4, 4)
    mask = jnp.ones(image_shape)

    import synthpix.data_generate as generate_mod

    # Collect debug messages
    logged = []
    monkeypatch.setattr(generate_mod.logger, "debug", lambda msg: logged.append(msg))

    # Call the function to test
    generate_mod.input_check_gen_img_from_flow(
        flow_field=flow_field,
        parameters=ImageGenerationSpecification(
            image_shape=image_shape,
        ),
        mask=mask,
    )

    # Check if the mask shape was logged
    expected_msg = f"Masking out {16 - jnp.sum(mask)} pixels " "in the images."
    assert any(
        expected_msg in m for m in logged
    ), f"Expected :{expected_msg}, got: {logged}"


@pytest.mark.parametrize("mask", ["a", [1, 2]])
def test_invalid_mask_type_in_generate(mask):
    """Test that invalid mask type raise a ValueError."""
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(ValueError, match="mask must be a jnp.ndarray or None."):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            mask=mask,
            parameters=ImageGenerationSpecification(
                image_shape=image_shape,
            ),
        )


@pytest.mark.parametrize(
    "mask",
    [
        jnp.zeros((356, 128)),
        jnp.ones((128, 1)),
        jnp.ones((128, 128, 1)),
    ],  # Invalid shapes
)
def test_invalid_mask_shape_in_generate(mask):
    """Test that invalid mask shape raise a ValueError."""
    flow_field = jnp.zeros((1, 128, 128, 2))
    image_shape = (128, 128)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"mask shape {mask.shape} does not match image_shape {image_shape}."
        ),
    ):
        input_check_gen_img_from_flow(
            flow_field=flow_field,
            mask=mask,
            parameters=ImageGenerationSpecification(
                image_shape=image_shape,
            ),
        )


@pytest.mark.parametrize(
    "mask",
    [
        jnp.ones((32, 32)),  # no mask: nothing is masked out
        jnp.pad(jnp.ones((16, 32)), ((0, 16), (0, 0))),  # bottom half masked
        jnp.eye(32),  # diagonal only
        jnp.zeros((32, 32)),  # all masked (everything should be zero)
    ],
)
def test_mask_applies_zeros(mask):
    key = jax.random.PRNGKey(1)
    flow_field = jnp.zeros((1, 32, 32, 2))  # no motion
    image_shape = (32, 32)
    image_offset = (0, 0)

    images1, images2, _ = generate_images_from_flow(
        key=key,
        flow_field=flow_field,
        parameters=ImageGenerationSpecification(
            image_shape=image_shape,
            img_offset=image_offset,
            batch_size=1,
        ),
        position_bounds=image_shape,
        mask=mask,
    )
    images1 = images1[0]  # Remove batch dimension
    images2 = images2[0]  # Remove batch dimension

    # Masked regions (where mask == 0) should be exactly zero
    masked1 = images1[mask == 0]
    masked2 = images2[mask == 0]
    assert jnp.all(masked1 == 0), f"Masked region in images1 is not zero: {masked1}"
    assert jnp.all(masked2 == 0), f"Masked region in images2 is not zero: {masked2}"

    # Unmasked regions (where mask == 1) should be nonzero
    # for at least some pixels (unless all masked)
    if jnp.any(mask == 1):
        unmasked1 = images1[mask == 1]
        unmasked2 = images2[mask == 1]
        assert jnp.any(unmasked1 != 0), "All unmasked region in images1 is zero"
        assert jnp.any(unmasked2 != 0), "All unmasked region in images2 is zero"
