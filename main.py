import logging
import random
import socket

import numpy as np
import torch

import wandb
from config import get_config, init_config
from data_preparation import DataPreparation
from model import Vision_Transformer
from trainer import Trainer
from utils import (
    calculate_mean_and_std,
    evaluate_and_sort_results,
    evaluate_model_accuracy,
    kendall_tau,
    show_label_counts,
    test_model_on_snow_scenes,
    visualize_predictions,
)


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
    wandb.init(project=config["WANDB"]["PROJECT"],
               entity=config["WANDB"]["USERNAME"],
               name=config["WANDB"]["RUN_NAME"],
               mode="disabled" if config["WANDB"]["DISABLED"] else "online")

    wandb.config.update(config)

    wandb.watch(model, log=None)


def set_seeds():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    # torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    set_seeds()
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
    reference_image = config["IMAGE"]["REFERENCE_IMAGE"]
    model = Vision_Transformer(pretrained_model=vit_model, reference_image=reference_image).to(device)
    if isinstance(model, Vision_Transformer):
        config["IMAGE"]["HEIGHT"] = 224
        config["IMAGE"]["WIDTH"] = 224
        config["IMAGE"]["CENTER_CROP"] = True
        config["IMAGE"]["NORMALIZE"] = True

    run_table.add_data("Device", device)
    # Add model name to the table
    run_table.add_data("Model Name", model.name)
    # Add pc name to the table
    run_table.add_data("PC Name", socket.gethostname())

    init_wandb(model)
    data_preparation = DataPreparation()
    # Create dataloaders
    train_loader, val_loader, test_loader, transform, val_pair_loader = data_preparation.create_dataloaders()
    #hogaliden_loader = data_preparation.create_hogaliden_dataloader()
    trainer = Trainer(model)
    best_model = trainer.train(train_loader, val_loader, save_model=True)

    if config["TRAINING"]["LOSS"] != "PairWise":
        val_loader = val_pair_loader
    # calculate mean and std on validation set
    mean, std = calculate_mean_and_std(best_model, val_loader, transform, device)
    # log mean and std
    run_table.add_data("Validation mean", str(mean.item()))
    run_table.add_data("Validation std", str(std.item()))
    # log test accuracy
    run_table.add_data("Test Accuracy", str(evaluate_model_accuracy(best_model, test_loader, device)))
    run_table.add_data("Kendall tau", str(kendall_tau(best_model, test_loader, device)))

    # Log accuracy and kendall tau for hogaliden dataset
    #run_table.add_data("Hogaliden Accuracy", str(evaluate_model_accuracy(best_model, hogaliden_loader, device)))
    #run_table.add_data("Hogaliden Kendall tau", str(kendall_tau(best_model, hogaliden_loader, device)))

    wandb.log({"Test Accuracy Table": run_table})


    visualize_predictions(best_model, test_loader)
    evaluate_and_sort_results(best_model, transform, mean, std, test_loader)
    if not config["IMAGE"]["REFERENCE_IMAGE"]:
        test_model_on_snow_scenes("snow_images", best_model, transform, mean, std)
    #show_autoencoder_results(model, test_loader)
    show_label_counts()
    wandb.finish()
