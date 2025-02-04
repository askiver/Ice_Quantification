# Class that trains the model

import logging

import torch
import wandb

from config import CONFIG


class Trainer:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.device = CONFIG["TRAINING"]["DEVICE"]
        self.criterion = model.loss
        self.logger = logging.getLogger(__name__)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=CONFIG["TRAINING"]["LEARNING_RATE"],
            weight_decay=CONFIG["TRAINING"]["WEIGHT_DECAY"],
        )

    def train(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, save_model=False) -> None:
        train_length = len(train_loader)
        val_length = len(val_loader)
        # loss_history = []
        for epoch in range(CONFIG["TRAINING"]["EPOCHS"]):
            self.model.train()
            epoch_loss = torch.tensor(0.0, device=self.device)
            for lower_img_batch, higher_img_batch, _, _ in train_loader:
                lower_img_data, higher_img_data = lower_img_batch.to(self.device), higher_img_batch.to(self.device)

                lower_img_output, higher_img_output = self.model(lower_img_data), self.model(higher_img_data)
                loss = self.criterion(lower_img_output, higher_img_output)
                epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # loss_history.append(epoch_loss)
            self.logger.info("Epoch: %d, Loss: %.6f", epoch, epoch_loss.to("cpu").item() / train_length)

            self.model.eval()

            epoch_val_loss = torch.tensor(0.0, device=self.device)
            for lower_img_batch_val, higher_img_batch_val, _, _ in val_loader:
                lower_img_data_val, higher_img_data_val = lower_img_batch_val.to(self.device), higher_img_batch_val.to(self.device)
                lower_img_output_val, higher_img_output_val = self.model(lower_img_data_val), self.model(higher_img_data_val)
                loss = self.criterion(lower_img_output_val, higher_img_output_val)
                epoch_val_loss += loss.item()
            print(f"Epoch: {epoch}, Loss: {epoch_val_loss.to('cpu').item()/val_length}")

            wandb.log({
                "epoch": epoch,
                "train_loss": epoch_loss.to("cpu").item() / train_length,
                "val_loss": epoch_val_loss.to("cpu").item() / val_length,
            })

        if save_model:
            torch.save(self.model.state_dict(), "models/model.pth")

