# Class that trains the model

import torch
from main import CONFIG


class Trainer:
    def __init(self, model):
        self.model = model
        self.device = CONFIG['TRAINING']['DEVICE']
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=CONFIG['TRAINING']['LEARNING_RATE'],
                                          weight_decay=CONFIG['TRAINING']['WEIGHT_DECAY'])

    def train(self, train_loader, val_loader):
        for epoch in range(CONFIG['TRAINING']['EPOCHS']):
            self.model.train()
            for train_data, target in train_loader:
                train_data, target = train_data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(train_data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            for val_data, target in val_loader:
                val_data, target = val_data.to(self.device), target.to(self.device)
                output = self.model(val_data)
                loss = self.criterion(output, target)
                print(f"Epoch: {epoch}, Loss: {loss.item()}")

