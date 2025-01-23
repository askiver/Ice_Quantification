import logging

import torch

from config import CONFIG
from data_preparation import DataPreparation
from model import AutoEncoder, VariationalAutoEncoder, SnowRanker
from trainer import Trainer
from utils import show_autoencoder_results

# Decide the device dynamically
device = "cuda" if torch.cuda.is_available() else "cpu"


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level (e.g., INFO, DEBUG)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Format for the log messages
        handlers=[
            logging.StreamHandler()  # This ensures output goes to the console (stdout)
        ]
    )


# Update the configuration with the runtime-decided device
CONFIG["TRAINING"]["DEVICE"] = device

if __name__ == "__main__":
    setup_logger()
    model = SnowRanker(3, 5).to(device)
    data_preparation = DataPreparation()
    train_loader, val_loader, test_loader = data_preparation.create_dataloaders()
    trainer = Trainer(model)
    trainer.train(train_loader, val_loader)
    #show_autoencoder_results(model, test_loader)
