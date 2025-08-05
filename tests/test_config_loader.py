import pytest
import yaml
import os
from src.utils.config_loader import load_config

@pytest.fixture
def config_path():
    """Provides the path to the test configuration file."""
    return "configs/config.yaml"

def test_load_config_successfully(config_path):
    """Tests that the configuration file can be loaded without errors."""
    config = load_config(config_path)
    assert config is not None

def test_config_is_dict(config_path):
    """Tests that the loaded configuration is a dictionary."""
    config = load_config(config_path)
    assert isinstance(config, dict)

def test_config_has_top_level_keys(config_path):
    """Tests that the configuration has the expected top-level keys."""
    config = load_config(config_path)
    assert "data" in config
    assert "model" in config
    assert "training" in config

def test_config_nested_key_value(config_path):
    """Tests that a nested key has the correct value."""
    config = load_config(config_path)
    assert config["model"]["params"]["learning_rate"] == 0.001

def test_load_config_file_not_found():
    """Tests that a FileNotFoundError is raised for a non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config.yaml")
