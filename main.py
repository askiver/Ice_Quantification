import logging

import torch

from config import CONFIG
from data_preparation import DataPreparation
from model import AutoEncoder, VariationalAutoEncoder, SnowRanker
from trainer import Trainer
import wandb
from utils import show_autoencoder_results, evaluate_model_accuracy, visualize_predictions, evaluate_and_sort_results

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

def init_wandb(model):
    wandb.init(
        project="snowranker",
        entity="askis-ntnu",)

    wandb.config.update(CONFIG)

    wandb.watch(model)



# Update the configuration with the runtime-decided device
CONFIG["TRAINING"]["DEVICE"] = device

if __name__ == "__main__":
    setup_logger()
    model = SnowRanker().to(device)
    init_wandb(model)
    data_preparation = DataPreparation()
    train_loader, val_loader, test_loader = data_preparation.create_dataloaders()
    trainer = Trainer(model)
    trainer.train(train_loader, val_loader, save_model=True)
    # load model
    #model.load_state_dict(torch.load("models/model.pth", weights_only=True))
    evaluate_model_accuracy(model, test_loader)
    visualize_predictions(model, test_loader)
    evaluate_and_sort_results(model, test_loader)
    #show_autoencoder_results(model, test_loader)
