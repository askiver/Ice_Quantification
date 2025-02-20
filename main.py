import logging
import torch
from config import get_config, init_config
from data_preparation import DataPreparation
from model import AutoEncoder, VariationalAutoEncoder, SnowRanker, Vision_Transformer
from trainer import Trainer
import wandb
from utils import show_autoencoder_results, evaluate_model_accuracy, visualize_predictions, evaluate_and_sort_results


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level (e.g., INFO, DEBUG)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Format for the log messages
        handlers=[
            logging.StreamHandler()  # This ensures output goes to the console (stdout)
        ]
    )

def init_wandb(model):
    config = get_config()
    wandb.login("never", config["WANDB"]["API_KEY"])
    wandb.init(project=config["WANDB"]["PROJECT"], entity=config["WANDB"]["USERNAME"])

    wandb.config.update(config)

    wandb.watch(model)


if __name__ == "__main__":
    setup_logger()
    init_config()
    run_table = wandb.Table(columns=["Parameter", "Value"])
    config = get_config()
    # Decide the device dynamically
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Update the configuration with the runtime-decided device
    config["TRAINING"]["DEVICE"] = device

    model = SnowRanker().to(device)
    #Vision_Transformer().to(device)
    if isinstance(model, Vision_Transformer):
        config["IMAGE"]["HEIGHT"] = 224
        config["IMAGE"]["WIDTH"] = 224
        config["IMAGE"]["CENTER_CROP"] = True
        config["IMAGE"]["NORMALIZE"] = True

    run_table.add_data("Device", device)
    # Add model name to the table
    run_table.add_data("Model Name", model.name)

    init_wandb(model)
    data_preparation = DataPreparation()
    train_loader, val_loader, test_loader, transform = data_preparation.create_dataloaders()
    trainer = Trainer(model)
    best_model = trainer.train(train_loader, val_loader, save_model=True)
    # load model
    #model.load_state_dict(torch.load("models/model.pth", weights_only=True))
    # log test accuracy
    run_table.add_data("Test Accuracy", str(evaluate_model_accuracy(best_model, test_loader)))
    wandb.log({"Test Accuracy Table": run_table})

    visualize_predictions(best_model, test_loader)
    evaluate_and_sort_results(best_model, transform, test_loader)
    #show_autoencoder_results(model, test_loader)
