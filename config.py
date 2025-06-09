from pathlib import Path

import yaml

CONFIG = None


# Load the configuration file
# Global variable imported to several files
def init_config():
    global CONFIG
    with Path("config.yaml").open() as file:
        CONFIG = yaml.safe_load(file)


def get_config():
    return CONFIG
