from pathlib import Path

import yaml

# Load the configuration file
# Global variable imported to several files
with Path("config.yaml").open() as file:
    CONFIG = yaml.safe_load(file)

