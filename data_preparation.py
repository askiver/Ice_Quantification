import torch
from torchvision import datasets, transforms
from config import CONFIG
from torch.utils.data import DataLoader, random_split, Subset


class DataPreparation:
    def __init__(self):
        self.batch_size = CONFIG['TRAINING']['BATCH_SIZE']
        self.root_path = './data/MNIST'
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5,), (0.5,))  # Normalize with mean and std for grayscale images
        ])

    def create_dataloaders(self):
        train_data = datasets.MNIST(self.root_path, train=True, download=True, transform=self.transform)
        test_data = datasets.MNIST(self.root_path, train=False, download=True, transform=self.transform)

        # Get all the indices for the data samples that are not '6'
        indices = [i for i, target in enumerate(train_data.targets) if target != 6]

        # Create a subset of the data with the indices
        train_data = Subset(train_data, indices)

        # Separate train into train and validation
        train_portion = CONFIG['TRAINING']['TRAIN_PORTION']
        train_size = int(len(train_data) * train_portion)
        val_size = len(train_data) - train_size

        # Split the dataset
        train_data, val_data = random_split(train_data, [train_size, val_size])

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
