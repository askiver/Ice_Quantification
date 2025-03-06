import logging
import random
import socket

import torch

import wandb
from config import get_config, init_config
from data_preparation import DataPreparation
from model import AutoEncoder, SnowRanker, VariationalAutoEncoder, Vision_Transformer
from trainer import Trainer
from utils import evaluate_and_sort_results, evaluate_model_accuracy, show_autoencoder_results, visualize_predictions, \
    show_label_counts


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level (e.g., INFO, DEBUG)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Format for the log messages
        handlers=[
            logging.StreamHandler()  # This ensures output goes to the console (stdout)
        ],
    )


def init_wandb(model):
    config = get_config()
    wandb.login("never", config["WANDB"]["API_KEY"])
    wandb.init(project=config["WANDB"]["PROJECT"], entity=config["WANDB"]["USERNAME"])

    wandb.config.update(config)

    wandb.watch(model)


def set_seeds():
    torch.manual_seed(0)
    random.seed(0)
    # torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    setup_logger()
    init_config()
    run_table = wandb.Table(columns=["Parameter", "Value"])
    config = get_config()
    # Decide the device dynamically
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Update the configuration with the runtime-decided device
    config["TRAINING"]["DEVICE"] = device

    # model = SnowRanker().to(device)
    vit_model = config["MODEL"]["VIT"]["PRE_TRAINED"]
    model = Vision_Transformer(pretrained_model=vit_model).to(device)
    if isinstance(model, Vision_Transformer):
        config["IMAGE"]["HEIGHT"] = 224
        config["IMAGE"]["WIDTH"] = 224
        config["IMAGE"]["CENTER_CROP"] = True
        config["IMAGE"]["NORMALIZE"] = True

    if config["TRAINING"]["LOSS"] != "PairWise":
        config["TRAINING"]["BATCH_SIZE"] = 1

    run_table.add_data("Device", device)
    # Add model name to the table
    run_table.add_data("Model Name", model.name)
    # Add pc name to the table
    run_table.add_data("PC Name", socket.gethostname())

    init_wandb(model)
    data_preparation = DataPreparation()
    train_loader, val_loader, test_loader, transform = data_preparation.create_dataloaders()
    trainer = Trainer(model)
    best_model = trainer.train(train_loader, val_loader, save_model=True)
    # load model
    # model.load_state_dict(torch.load("models/model.pth", weights_only=True))
    # log test accuracy
    run_table.add_data("Test Accuracy", str(evaluate_model_accuracy(best_model, test_loader)))
    wandb.log({"Test Accuracy Table": run_table})

    visualize_predictions(best_model, test_loader)
    evaluate_and_sort_results(best_model, transform, test_loader)
    #show_autoencoder_results(model, test_loader)
    show_label_counts()
    wandb.finish()
