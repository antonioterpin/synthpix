import builtins
import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from synthpix.sanity import (
    calculate_min_and_max_speeds,
    missing_speeds_panel,
    update_config_file,
)
from synthpix.utils import load_configuration


def test_update_config_file():
    """Test the update_config_file function."""
    # Create a temporary configuration file based on test_data.yaml
    base_config_path = os.path.join("config", "test_data.yaml")
    tmp_path = tempfile.gettempdir()
    temp_config_path = os.path.join(tmp_path, "temp_config.yaml")
    try:
        base_config = load_configuration(base_config_path)
        with open(temp_config_path, "w") as temp_file:
            yaml.safe_dump(base_config, temp_file)
        # Define the updates to be made
        updates = {
            "max_speed_x": 15.0,
            "max_speed_y": 20.0,
            "min_speed_x": -15.0,
            "min_speed_y": -20.0,
        }
        # Call the function to update the configuration file
        update_config_file(temp_config_path, updates)
        # Reload the updated configuration file
        updated_config = load_configuration(temp_config_path)
        # Assert that the updates were applied correctly
        for key, value in updates.items():
            assert updated_config[key] == value
        # Assert that other keys remain unchanged
        for key in base_config:
            if key not in updates:
                assert updated_config[key] == base_config[key]
    finally:
        # Ensure the temporary file is deleted
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def test_convert_to_standard_type(tmp_path):
    # 1. create a minimal temporary YAML config
    cfg_path = tmp_path / "sample_cfg.yaml"
    cfg_path.write_text("placeholder: 42\n")

    # 2. values that trigger every branch
    updates = {
        "np_float": np.float32(1.23),
        "jnp_float": jnp.float32(4.56),
        "np_int": np.int64(7),
        "jnp_int": jnp.int32(8),
        "np_array": np.array([1, 2, 3]),
        "jnp_array": jnp.array([4, 5, 6]),
        "string_val": "hello",
    }

    # 3. run the helper (this calls convert_to_standard_type internally)
    update_config_file(str(cfg_path), updates)

    # 4. reload and assert conversions
    cfg = load_configuration(str(cfg_path))

    assert (
        isinstance(cfg["np_float"], float)
        and pytest.approx(cfg["np_float"]) == 1.23
    )
    assert (
        isinstance(cfg["jnp_float"], float)
        and pytest.approx(cfg["jnp_float"]) == 4.56
    )
    assert isinstance(cfg["np_int"], int) and cfg["np_int"] == 7
    assert isinstance(cfg["jnp_int"], int) and cfg["jnp_int"] == 8

    assert cfg["np_array"] == [1, 2, 3], (
        "np.ndarray should be converted to list"
    )
    assert cfg["jnp_array"] == [4, 5, 6], (
        "jnp.ndarray should be converted to list"
    )

    # fallback branch: untouched
    assert cfg["string_val"] == "hello", "string should remain unchanged"


@pytest.mark.parametrize("mock_hdf5_files", [2], indirect=True)
def test_calculate_min_and_max_speeds(mock_hdf5_files):
    """Test the calculate_min_and_max_speeds function."""
    files, dims = mock_hdf5_files

    # Call the function to calculate speeds
    result = calculate_min_and_max_speeds(files)

    # Assert the results
    assert "min_speed_x" in result
    assert "max_speed_x" in result
    assert "min_speed_y" in result
    assert "max_speed_y" in result

    # Ensure the values are within expected ranges based on the mock data
    assert result["min_speed_x"] <= result["max_speed_x"]
    assert result["min_speed_y"] <= result["max_speed_y"]


@pytest.mark.parametrize(
    "cfg_name, cfg_content, expected_exception",
    [
        ("no_key", {}, ValueError),
        ("not_a_list", {"file_list": "not_a_list"}, ValueError),
        ("non_string_items", {"file_list": [123, 456]}, ValueError),
        ("empty_list", {"file_list": []}, ValueError),
        (
            "nonexistent_file",
            {"file_list": ["nonexistent_file.h5"]},
            ValueError,
        ),
    ],
)
def test_invalid_inputs_missing_speeds_panel(
    tmp_path, cfg_name, cfg_content, expected_exception
):
    """Test with invalid inputs for missing_speeds_panel."""
    cfg_file = tmp_path / f"{cfg_name}.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg_content))
    with pytest.raises(expected_exception):
        missing_speeds_panel(str(cfg_file))


def test_not_h5_inputs_missing_speeds_panel(tmp_path):
    """Test with non-HDF5 file inputs for missing_speeds_panel."""
    cfg_file = tmp_path / "not_h5.yaml"
    cfg_content = {"file_list": [str(cfg_file)]}
    cfg_file.write_text(yaml.safe_dump(cfg_content))
    with pytest.raises(ValueError, match=f"File {cfg_file} is not a .h5 file"):
        missing_speeds_panel(str(cfg_file))


def _fake_calculate(_files):
    """Stand-in for calculate_min_and_max_speeds."""
    return {
        "max_speed_x": 10.0,
        "max_speed_y": 20.0,
        "min_speed_x": -5.0,
        "min_speed_y": -10.0,
    }


@pytest.mark.parametrize(
    "cli_answers, expect_exception, expect_speeds",
    [
        (["1", "y"], None, (10.0, 20.0, -5.0, -10.0)),  # happy path
        (["1", "n"], RuntimeError, None),  # abort after calc
        (["2"], RuntimeError, None),  # manual edit branch
        (["garbage"], RuntimeError, None),  # invalid choice
    ],
)
@pytest.mark.parametrize("mock_hdf5_files", [1], indirect=True)
def test_missing_speeds_panel_all_branches(
    monkeypatch,
    mock_hdf5_files,
    tmp_path,
    cli_answers,
    expect_exception,
    expect_speeds,
):
    """Drive every branch in `missing_speeds_panel` by faking user input."""
    files, _dims = mock_hdf5_files
    # Write initial config with only file_list
    cfg_dict = {"file_list": files}
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg_dict))

    # 1. stub heavy/irrelevant helpers
    monkeypatch.setattr(
        "synthpix.sanity.calculate_min_and_max_speeds", _fake_calculate
    )
    monkeypatch.setattr(
        "synthpix.sanity.update_config_file", lambda *_, **__: None
    )
    # ensure update_config_file would target our config file path
    import synthpix.sanity as sp_sanity

    monkeypatch.setattr(sp_sanity, "config_path", str(cfg_file), raising=False)

    # 2. simulate successive `input()` calls
    answers = iter(cli_answers)
    monkeypatch.setattr(builtins, "input", lambda *_args: next(answers))

    # 3. assert behaviour
    if expect_exception is None:
        speeds = missing_speeds_panel(str(cfg_file))
        assert speeds == expect_speeds
    else:
        with pytest.raises(expect_exception):
            missing_speeds_panel(str(cfg_file))


@pytest.mark.parametrize("mock_hdf5_files", [1], indirect=True)
def test_missing_speeds_panel_when_values_present(
    monkeypatch, mock_hdf5_files, tmp_path
):
    """Fast-path: nothing is missing -> no prompts."""
    files, _dims = mock_hdf5_files
    # Write config with all required keys
    cfg = {
        "file_list": files,
        "max_speed_x": 1.0,
        "max_speed_y": 2.0,
        "min_speed_x": -1.0,
        "min_speed_y": -2.0,
    }
    cfg_file = tmp_path / "full_cfg.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg))

    # Patch input to fail if called
    monkeypatch.setattr(
        builtins,
        "input",
        lambda *_: pytest.fail("input() should not be called"),
    )

    result = missing_speeds_panel(str(cfg_file))
    assert result == (1.0, 2.0, -1.0, -2.0)
