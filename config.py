import yaml

# Load the configuration file
# Global variable imported to several files
with open('config.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)
