import yaml
import torch

# Load the configuration file
# Global variable imported to several files
with open('config.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)

# Decide the device dynamically
device = "cuda" if torch.cuda.is_available() else "cpu"

# Update the configuration with the runtime-decided device
CONFIG['TRAINING']['DEVICE'] = device



if __name__ == "__main__":
    print("Hello, World!")