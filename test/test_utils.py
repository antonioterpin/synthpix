import pytest

from src.utils import compute_image_scaled_height, is_int


@pytest.mark.parametrize(
    "original_height, original_width, new_width, expected_height",
    [
        (200, 400, 800, 400),
        (100, 400, 800, 200),
        (50, 200, 100, 25),
        (300, 600, 900, 450),
    ],
)
def test_compute_image_scaled_height(
    original_height, original_width, new_width, expected_height
):
    """Test the compute_image_scaled_height function with various inputs.

    Args:
        original_height (int): The original height of the image.
        original_width (int): The original width of the image.
        new_width (int): The new width to scale the image to.
        expected_height (int): The expected height after scaling.
    """
    assert (
        compute_image_scaled_height(original_height, original_width, new_width)
        == expected_height
    )


@pytest.mark.parametrize(
    "original_height, original_width, new_width",
    [
        (0, 400, 800),
        (100, 0, 800),
        (50, 200, 0),
        (-300, 600, 900),
        (300, -600, 900),
        (300, 600, -900),
    ],
)
def test_compute_image_scaled_height_exceptions(
    original_height, original_width, new_width
):
    """Test the compute_image_scaled_height function with invalid inputs.

    Args:
        original_height (int): The original height of the image.
        original_width (int): The original width of the image.
        new_width (int): The new width to scale the image to.
    """
    with pytest.raises(ValueError):
        compute_image_scaled_height(original_height, original_width, new_width)


@pytest.mark.parametrize(
    "val, expected",
    [
        (1, True),
        (1.0, True),
        (1.000000001, True),
        (1.00000001, True),
        (1.5, False),
        (0, True),
        (-1, True),
        (-1.0, True),
        (-1.000000001, True),
        (-1.00000001, True),
        (1e9, True),
        (1e9 + 0.000000001, True),
        (1e9 + 0.00000001, True),
    ],
)
def test_is_int(val, expected):
    """Test the is_int function with various inputs.

    Args:
        val (Union[int, float]): The value to check.
        expected (bool): The expected result.
    """
    assert is_int(val) == expected
