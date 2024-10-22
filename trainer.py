# Class that trains the model

import logging

import torch

from config import CONFIG


class Trainer:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.device = CONFIG["TRAINING"]["DEVICE"]
        self.criterion = model.loss
        self.logger = logging.getLogger(__name__)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=CONFIG["TRAINING"]["LEARNING_RATE"],
            weight_decay=CONFIG["TRAINING"]["WEIGHT_DECAY"],
        )

    def train(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader) -> None:
        train_length = len(train_loader)
        val_length = len(val_loader)
        # loss_history = []
        for epoch in range(CONFIG["TRAINING"]["EPOCHS"]):
            self.model.train()
            epoch_loss = torch.tensor(0.0, device=self.device)
            for train_data_batch, _ in train_loader:
                train_data = train_data_batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(train_data)
                loss = self.criterion(*output, train_data)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            # loss_history.append(epoch_loss)
            self.logger.info("Epoch: %d, Loss: %.6f", epoch, epoch_loss.to("cpu").item() / train_length)

            self.model.eval()
            """
            epoch_val_loss = torch.tensor(0.0, device=self.device)
            for val_data, _ in val_loader:
                val_data = val_data.to(self.device)
                output = self.model(val_data)
                loss = self.criterion(output, val_data)
                epoch_val_loss += loss.item()
            print(f"Epoch: {epoch}, Loss: {epoch_val_loss.to('cpu').item()/val_length}")
            """
