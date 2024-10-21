import torch
from data_preparation import DataPreparation
from model import AutoEncoder, VariationalAutoEncoder
from trainer import Trainer
from config import CONFIG
from utils import show_autoencoder_results

# Decide the device dynamically
device = "cuda" if torch.cuda.is_available() else "cpu"

# Update the configuration with the runtime-decided device
CONFIG['TRAINING']['DEVICE'] = device

if __name__ == "__main__":
    model = VariationalAutoEncoder().to(device)
    data_preparation = DataPreparation()
    train_loader, val_loader, test_loader = data_preparation.create_dataloaders()
    trainer = Trainer(model)
    trainer.train(train_loader, val_loader)
    show_autoencoder_results(model, test_loader)

