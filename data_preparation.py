import random

import torch
from torch.utils.data import DataLoader, Subset, random_split, Dataset
from torchvision import datasets, transforms
from label_ordering import load_progress_ordered
from PIL import Image
from utils import create_image_splits, retrieve_snowless_images, add_reference_image
from config import CONFIG

class PairWiseImageDataset(Dataset):

    def __init__(self, ordered_images_subset, max_rank_index, snowless_images=None):

        image_height = CONFIG["TRAINING"]["IMAGE_DIMENSIONS"]["HEIGHT"]
        image_width = CONFIG["TRAINING"]["IMAGE_DIMENSIONS"]["WIDTH"]
        add_reference = CONFIG["TRAINING"]["REFERENCE_IMAGE"]

        # Generate ordered image pairs
        # Currently do not support ties

        self.transform = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
        ])

        self.pairs = []
        # Due to ties, images are ordered in groups
        for idx, (lower_group, low_rank_index) in enumerate(ordered_images_subset):
            for higher_group, high_rank_index in ordered_images_subset[idx + 1:]:

                for lower_img_path in lower_group:
                    lower_img = self.transform(Image.open(lower_img_path).convert("RGB"))
                    if add_reference:
                        lower_img = add_reference_image(lower_img, self.transform, snowless_images)

                    for higher_img_path in higher_group:
                        higher_img = self.transform(Image.open(higher_img_path).convert("RGB"))
                        if add_reference:
                            higher_img = add_reference_image(higher_img, self.transform, snowless_images)

                        rank_difference = (high_rank_index - low_rank_index) / max_rank_index
                        # Leftmost always lower
                        self.pairs.append((lower_img, higher_img, str(lower_img_path), str(higher_img_path), rank_difference))


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        lower_img_path, higher_img_path = self.pairs[idx]

        # Load and transform images on-the-fly
        lower_img = self.transform(Image.open(lower_img_path).convert("RGB"))
        higher_img = self.transform(Image.open(higher_img_path).convert("RGB"))

        return lower_img, higher_img, lower_img_path, higher_img_path
        """
        return self.pairs[idx]


class DataPreparation:
    def __init__(self) -> None:
        self.batch_size = CONFIG["TRAINING"]["BATCH_SIZE"]
        self.train_portion = CONFIG["TRAINING"]["TRAIN_PORTION"]
        self.val_portion = CONFIG["TRAINING"]["VAL_PORTION"]
        #self.root_path = "./data/MNIST"

    def create_dataloaders(self) -> (DataLoader, DataLoader, DataLoader):

        ordered_images = load_progress_ordered()

        max_rank_index = len(ordered_images)
        print(f"Max rank index: {max_rank_index}")

        # Separate train, validation and test
        train_groups, val_groups, test_groups = create_image_splits(ordered_images, self.train_portion, self.val_portion)

        # load snowless images
        snowless_images = retrieve_snowless_images(41, "03")



        train_data = PairWiseImageDataset(train_groups, max_rank_index, snowless_images)
        val_data = PairWiseImageDataset(val_groups, max_rank_index, snowless_images)
        test_data = PairWiseImageDataset(test_groups, max_rank_index, snowless_images)

        """
        # Separate train into train and validation
        train_portion = CONFIG["TRAINING"]["TRAIN_PORTION"]
        train_size = int(len(train_data) * train_portion)
        val_size = len(train_data) - train_size

        # Split the dataset
        train_data, val_data = random_split(train_data, [train_size, val_size])
        """

        print(f"Train data size: {len(train_data)}")
        print(f"Validation data size: {len(val_data)}")
        print(f"Test data size: {len(test_data)}")

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
