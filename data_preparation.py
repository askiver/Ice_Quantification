from torch.utils.data import DataLoader, Subset, random_split, Dataset
from torchvision import datasets, transforms
from label_ordering import load_progress_ordered
from PIL import Image

from config import CONFIG

class PairWiseImageDataset(Dataset):

    def __init__(self, train=True, train_portion=0.8):

        ordered_images = load_progress_ordered()
        image_height = CONFIG["TRAINING"]["IMAGE_DIMENSIONS"]["HEIGHT"]
        image_width = CONFIG["TRAINING"]["IMAGE_DIMENSIONS"]["WIDTH"]

        # Generate ordered image pairs
        # Currently do not support ties

        transform = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
        ])

        self.pairs = []

        for idx, higher_group in enumerate(ordered_images):
            for lower_group in ordered_images[idx + 1:]:
                for higher_img_path in higher_group:
                    higher_img_tensor = transform(Image.open(higher_img_path).convert("RGB"))
                    for lower_img_path in lower_group:
                        lower_img_tensor = transform(Image.open(lower_img_path).convert("RGB"))
                        # Leftmost always higher
                        self.pairs.append((higher_img_tensor, lower_img_tensor, higher_img_path, lower_img_path))

        if train:
            self.pairs = self.pairs[:int(len(self.pairs)*train_portion)]
        else:
            self.pairs = self.pairs[int(len(self.pairs)*train_portion):]


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class DataPreparation:
    def __init__(self) -> None:
        self.batch_size = CONFIG["TRAINING"]["BATCH_SIZE"]
        #self.root_path = "./data/MNIST"

    def create_dataloaders(self) -> (DataLoader, DataLoader, DataLoader):

        # TODO: Add disjoint image sets, not only disjoint pair sets
        train_data = PairWiseImageDataset(train=True, train_portion=0.8)
        test_data = PairWiseImageDataset(train=False, train_portion=0.8)


        # Separate train into train and validation
        train_portion = CONFIG["TRAINING"]["TRAIN_PORTION"]
        train_size = int(len(train_data) * train_portion)
        val_size = len(train_data) - train_size

        # Split the dataset
        train_data, val_data = random_split(train_data, [train_size, val_size])

        print(f"Train data size: {len(train_data)}")
        print(f"Validation data size: {len(val_data)}")
        print(f"Test data size: {len(test_data)}")

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
