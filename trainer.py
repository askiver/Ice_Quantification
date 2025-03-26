# Class that trains the model
import logging
import torch
from torch import nn
import wandb
from config import get_config
from torch.optim.lr_scheduler import OneCycleLR


class Trainer:
    def __init__(self, model: torch.nn.Module) -> None:
        self.scheduler = None
        config = get_config()
        self.model = model
        self.device = config["TRAINING"]["DEVICE"]
        self.epochs = config["TRAINING"]["EPOCHS"]
        self.criterion = model.get_correct_loss(config["TRAINING"]["LOSS"])
        self.logger = logging.getLogger(__name__)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["TRAINING"]["LEARNING_RATE"],
            weight_decay=config["TRAINING"]["WEIGHT_DECAY"],
        )     
        self.train_method = self.pair_learning if config["TRAINING"]["LOSS"] == "PairWise" else self.list_learning
        self.lowest_val_loss = float("inf")
        self.best_model = None
        self.early_stopping = config["TRAINING"]["EARLY_STOPPING"]

    def train(
        self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, save_model=False
    ) -> nn.Module:
        config = get_config()

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config["TRAINING"]["LEARNING_RATE"],
            steps_per_epoch=len(train_loader),
            epochs=config["TRAINING"]["EPOCHS"],
        )
        
        train_length = len(train_loader)
        val_length = len(val_loader)
        epochs_since_improvement = 0
        # loss_history = []
        for epoch in range(self.epochs):

            self.model.train()
            epoch_loss = self.train_method(train_loader, train=True)
            self.logger.info("Epoch: %d, Training Loss: %.6f", epoch, epoch_loss.to("cpu").item() / train_length)


            self.model.eval()
            with torch.no_grad():
                epoch_val_loss = self.train_method(val_loader, train=False)
            self.logger.info("Epoch: %d, Validation Loss: %.6f", epoch, epoch_val_loss.to("cpu").item() / val_length)


            if epoch_val_loss < self.lowest_val_loss:
                self.lowest_val_loss = epoch_val_loss
                self.best_model = self.model.state_dict()
                epochs_since_improvement = 0

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": epoch_loss.to("cpu").item() / train_length,
                    "val_loss": epoch_val_loss.to("cpu").item() / val_length,
                    "learning_rate": self.scheduler.get_last_lr()[0],
                })

            if epochs_since_improvement > self.early_stopping:
                self.logger.info("Early stopping")
                break

            epochs_since_improvement += 1

        if save_model:
            torch.save(self.best_model, "models/model.pth")

        self.model.load_state_dict(self.best_model)
        self.logger.info("Training completed")
        return self.model


    def list_learning(self, data_loader, train=True):
        epoch_loss = torch.tensor(0.0, device=self.device)
        for images, true_ranks, _ in data_loader:
            self.optimizer.zero_grad()
            # Remove batch dimension for images and ranks
            images = images.squeeze(0)
            true_ranks = true_ranks.squeeze(0)

            images, true_ranks = images.to(self.device), true_ranks.to(self.device)
            images_output = self.model(images)
            # squeeze images
            images_output = images_output.squeeze(-1)
            loss = self.criterion(images_output, true_ranks)
            epoch_loss += loss.item()

            if train:
                loss.backward()
                self.optimizer.step()
                # Update learning rate
                self.scheduler.step()

        return epoch_loss


    def pair_learning(self, data_loader, train=True):
        epoch_loss = torch.tensor(0.0, device=self.device)
        for lower_img_batch, higher_img_batch, _, _, rank_difference in data_loader:
            self.optimizer.zero_grad()
            lower_img_data, higher_img_data, rank_difference = (
                lower_img_batch.to(self.device),
                higher_img_batch.to(self.device),
                rank_difference.to(self.device),
            )

            lower_img_output, higher_img_output = (
                self.model(lower_img_data),
                self.model(higher_img_data),
            )

            loss = self.criterion(lower_img_output, higher_img_output, rank_difference)
            epoch_loss += loss.item()

            if train:
                loss.backward()
                self.optimizer.step()
                # Update learning rate
                self.scheduler.step()

        return epoch_loss