from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from config import get_config
from label_ordering import load_progress_ordered
from utils import add_reference_image, create_image_splits, retrieve_snowless_images


class PairWiseImageDataset(Dataset):
    def __init__(self, ordered_images_subset, snowless_images=None):
        config = get_config()

        image_height = config["IMAGE"]["HEIGHT"]
        image_width = config["IMAGE"]["WIDTH"]
        add_reference = config["IMAGE"]["REFERENCE_IMAGE"]
        center_crop = config["IMAGE"]["CENTER_CROP"]
        normalize = config["IMAGE"]["NORMALIZE"]
        shortest_side = 1080

        # Generate ordered image pairs

        self.transform = transforms.Compose(
            [
                *([transforms.CenterCrop(shortest_side)] if center_crop else []),
                transforms.Resize((image_height, image_width)),
                transforms.ToTensor(),
                *([transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)] if normalize else []),
            ]
        )

        self.pairs = []
        max_rank_index = len(ordered_images_subset) - 1
        # Due to ties, images are ordered in groups
        for low_idx, lower_image_path in enumerate(ordered_images_subset):
            lower_img = self.transform(Image.open(lower_image_path).convert("RGB"))
            if add_reference:
                lower_img = add_reference_image(lower_img, self.transform, snowless_images)

            for high_idx, higher_image_path in enumerate(ordered_images_subset[low_idx + 1 :]):
                higher_img = self.transform(Image.open(higher_image_path).convert("RGB"))
                if add_reference:
                    higher_img = add_reference_image(higher_img, self.transform, snowless_images)
                rank_difference = high_idx / max_rank_index
                # Leftmost always lower
                self.pairs.append(
                    (lower_img, higher_img, str(lower_image_path), str(higher_image_path), rank_difference)
                )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """lower_img_path, higher_img_path = self.pairs[idx]

        # Load and transform images on-the-fly
        lower_img = self.transform(Image.open(lower_img_path).convert("RGB"))
        higher_img = self.transform(Image.open(higher_img_path).convert("RGB"))

        return lower_img, higher_img, lower_img_path, higher_img_path
        """
        return self.pairs[idx]


class DataPreparation:
    def __init__(self) -> None:
        config = get_config()
        self.batch_size = config["TRAINING"]["BATCH_SIZE"]
        self.train_portion = config["TRAINING"]["TRAIN_PORTION"]
        self.val_portion = config["TRAINING"]["VAL_PORTION"]

    def create_dataloaders(self) -> (DataLoader, DataLoader, DataLoader):
        train_datasets = []
        val_datasets = []
        test_datasets = []

        for wind_turbine in ["07", "21", "41"]:
            for angle in ["01", "02", "03"]:
                ordered_images = load_progress_ordered(f"WT_{wind_turbine}_SVIV{angle}")

                # Separate train, validation and test
                train_groups, val_groups, test_groups = create_image_splits(
                    ordered_images, self.train_portion, self.val_portion
                )

                # load snowless images
                snowless_images = retrieve_snowless_images(wind_turbine, angle)

                train_data = PairWiseImageDataset(train_groups, snowless_images)
                val_data = PairWiseImageDataset(val_groups, snowless_images)
                test_data = PairWiseImageDataset(test_groups, snowless_images)

                train_datasets.append(train_data)
                val_datasets.append(val_data)
                test_datasets.append(test_data)

        # Concatenate all datasets
        transform = train_datasets[0].transform
        train_data = ConcatDataset(train_datasets)
        val_data = ConcatDataset(val_datasets)
        test_data = ConcatDataset(test_datasets)

        print(f"Train data size: {len(train_data)}")
        print(f"Validation data size: {len(val_data)}")
        print(f"Test data size: {len(test_data)}")

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, drop_last=False)

        return train_loader, val_loader, test_loader, transform
