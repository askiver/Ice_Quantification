from torch.utils.data import DataLoader, Subset, random_split, Dataset
from torchvision import datasets, transforms
from label_ordering import load_progress_ordered
from PIL import Image
from utils import create_image_splits
from config import CONFIG

class PairWiseImageDataset(Dataset):

    def __init__(self, ordered_images_subset):

        image_height = CONFIG["TRAINING"]["IMAGE_DIMENSIONS"]["HEIGHT"]
        image_width = CONFIG["TRAINING"]["IMAGE_DIMENSIONS"]["WIDTH"]

        # Generate ordered image pairs
        # Currently do not support ties

        self.transform = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
        ])

        self.pairs = []
        # Due to ties, images are ordered in groups
        for idx, lower_group in enumerate(ordered_images_subset):
            for higher_group in ordered_images_subset[idx + 1:]:
                for lower_img_path in lower_group:
                    lower_img = self.transform(Image.open(lower_img_path).convert("RGB"))
                    for higher_img_path in higher_group:
                        higher_img = self.transform(Image.open(higher_img_path).convert("RGB"))
                        # Leftmost always lower
                        self.pairs.append((lower_img, higher_img, lower_img_path, higher_img_path))


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

        # TODO: Add disjoint image sets, not only disjoint pair sets
        ordered_images = load_progress_ordered()

        # Separate train, validation and test
        train_groups, val_groups, test_groups = create_image_splits(ordered_images, self.train_portion, self.val_portion)



        train_data = PairWiseImageDataset(train_groups)
        val_data = PairWiseImageDataset(val_groups)
        test_data = PairWiseImageDataset(test_groups)

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
