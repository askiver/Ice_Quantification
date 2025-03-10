# Class that trains the model
import logging
import torch
from torch import nn
import wandb
from config import get_config


class Trainer:
    def __init__(self, model: torch.nn.Module) -> None:
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
        self.lowest_val_loss = float("inf")
        self.best_model = None
        self.early_stopping = config["TRAINING"]["EARLY_STOPPING"]

    def train(
        self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, save_model=False
    ) -> nn.Module:
        train_length = len(train_loader)
        val_length = len(val_loader)
        epochs_since_improvement = 0
        # loss_history = []
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = torch.tensor(0.0, device=self.device)
            for images, true_ranks, _ in train_loader:
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
                loss.backward()
                self.optimizer.step()

            """
            for lower_img_batch, higher_img_batch, _, _, rank_difference in train_loader:
                lower_img_data, higher_img_data, rank_difference = (
                    lower_img_batch.to(self.device),
                    higher_img_batch.to(self.device),
                    rank_difference.to(self.device),
                )

                lower_img_output, higher_img_output = (
                    self.model(lower_img_data),
                    self.model(higher_img_data),
                )
                """

            # loss_history.append(epoch_loss)
            self.logger.info("Epoch: %d, Training Loss: %.6f", epoch, epoch_loss.to("cpu").item() / train_length)

            self.model.eval()

            epoch_val_loss = torch.tensor(0.0, device=self.device)
            """
            for lower_img_batch_val, higher_img_batch_val, _, _, rank_difference_val in val_loader:

                lower_img_data_val, higher_img_data_val =lower_img_batch_val.to(self.device),higher_img_batch_val.to(self.device)
                rank_difference_val = rank_difference_val.to(self.device)

                lower_img_output_val, higher_img_output_val = self.model(lower_img_data_val), self.model(higher_img_data_val)
            """
            for images_val, true_ranks_val, _ in val_loader:
                # Remove batch dimension for images and ranks
                true_ranks_val = true_ranks_val.squeeze(0)
                images_val = images_val.squeeze(0)
                images_val, true_ranks_val = images_val.to(self.device), true_ranks_val.to(self.device)

                images_val_output = self.model(images_val)

                # squeeze images
                images_val_output = images_val_output.squeeze(-1)

                loss = self.criterion(images_val_output, true_ranks_val)
                epoch_val_loss += loss.item()
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
                }
            )
            if epochs_since_improvement > self.early_stopping:
                self.logger.info("Early stopping")
                break

            epochs_since_improvement += 1

        if save_model:
            torch.save(self.best_model, "models/model.pth")

        self.model.load_state_dict(self.best_model)
        self.logger.info("Training completed")
        return self.model
