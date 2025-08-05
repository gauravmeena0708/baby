import yaml
import os

def load_config(config_path: str):
    """
    Loads a YAML configuration file.

    Args:
        config_path: The path to the configuration file.

    Returns:
        A dictionary with the configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None
