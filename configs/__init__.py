# Configs package
import yaml
import os

def load_config(config_name: str = "default"):
    """Load a config file."""
    config_path = os.path.join(os.path.dirname(__file__), f"{config_name}.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_default_config():
    """Get default configuration."""
    return load_config("default")