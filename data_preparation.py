import functools
import random
import glob
import os

import cv2
import torch
from pathlib import Path
from cv2 import imread
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageOps
import albumentations as A
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import functional as F

from config import get_config, init_config
from label_images import load_progress
from label_ordering import load_progress_ordered
from utils import add_reference_image, create_image_splits, retrieve_snowless_images

class SquareCenterCrop:
    def __call__(self, img):
        # Get the image size: (width, height)
        width, height = img.size
        # Determine the smallest side to use as crop size
        crop_size = min(width, height)
        # Apply center crop with computed square dimensions
        return F.center_crop(img, [crop_size, crop_size])

class LetterboxPad:
    def __init__(self, size, fill=(0, 0, 0)):
        """
        size: desired output square size (e.g., 224)
        fill: padding color, default black
        """
        self.size = size
        self.fill = fill

    def __call__(self, img):
        # Use ImageOps.pad to resize the image, preserving aspect ratio,
        # and add padding (if needed) to make it exactly (size, size)
        return ImageOps.pad(img, (self.size, self.size), color=self.fill, centering=(0.5, 0.5))

class ListImageDataset(Dataset):
    def __init__(self, ordered_images_subset):
        config = get_config()

        self.transform = define_transform()

        images = []
        image_ranks = []
        image_paths = []
        for idx, image_path in enumerate(reversed(ordered_images_subset)):
            #img = self.transform(Image.open(image_path).convert("RGB"))
            images.append(img)
            image_ranks.append(idx)
            image_paths.append(image_path)

        # Stack images into a single tensor
        images_tensor = torch.stack(images, dim=0)

        # Convert ranks to tensor
        image_ranks = torch.tensor(image_ranks, dtype=torch.float32)

        self.data = [(images_tensor, image_ranks, image_paths)]

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]


def define_transform(train=True):
    config = get_config()

    center_crop = config["IMAGE"]["CENTER_CROP"]
    image_height = config["IMAGE"]["HEIGHT"]
    image_width = config["IMAGE"]["WIDTH"]
    horizontal_flip = config["IMAGE"]["HORIZONTAL_FLIP"]
    brightness = config["IMAGE"]["BRIGHTNESS"]
    contrast = config["IMAGE"]["CONTRAST"]
    saturation = config["IMAGE"]["SATURATION"]
    hue = config["IMAGE"]["HUE"]
    normalize = config["IMAGE"]["NORMALIZE"]

    """
    transform = transforms.Compose(
        [
            #*([SquareCenterCrop()] if center_crop else []),
            #transforms.Resize((image_height, image_width)),
            *([transforms.RandomHorizontalFlip(p=horizontal_flip)] if train else []),
            *([transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)] if train else []),
            LetterboxPad(size=224, fill=(0, 0, 0)),
            transforms.ToTensor(),
            *([transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)] if normalize else []),
        ]
    )
    """
    transform = A.Compose([
        A.LongestMaxSize(max_size=max(image_height, image_width)),
        *([A.HorizontalFlip(p=horizontal_flip)] if train else []),
        *([A.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)] if train else []),
        A.PadIfNeeded(min_height=image_height, min_width=image_width, border_mode=cv2.BORDER_CONSTANT, fill=(0, 0, 0)),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

    return transform

class PairWiseImageDataset(Dataset):
    def __init__(self, ordered_images_subset, snowless_images=None, train=True):
        config = get_config()
        add_reference = config["IMAGE"]["REFERENCE_IMAGE"]
        dev_run = config["TRAINING"]["QUICK_DEV_RUN"]

        # Generate ordered image pairs

        self.transform = define_transform(train)

        self.pairs = []

        max_rank_index = len(ordered_images_subset) - 1

        for low_idx, lower_image_path in enumerate(ordered_images_subset):
            for high_idx, higher_image_path in enumerate(ordered_images_subset[low_idx + 1 :]):
                rank_difference = high_idx / max_rank_index
                # Leftmost always lower
                self.pairs.append((lower_image_path, higher_image_path, rank_difference))

        # Only use a subset of the dataset for quick dev run
        if dev_run:
            self.pairs = random.sample(self.pairs, min(100, len(self.pairs)))

        """
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
        """

    def __len__(self):
        return len(self.pairs)

    @functools.cache
    def load_image(self, image_path):
        # Load the image from the path
        img = imread(image_path)
        # Convert to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


    def __getitem__(self, idx):
        # Retrieve the file paths and rank difference for the given index
        lower_image_path, higher_image_path, rank_difference = self.pairs[idx]

        # Open images using PIL and convert to RGB
        lower_img = self.load_image(lower_image_path)
        higher_img = self.load_image(higher_image_path)

        # Apply the transformation pipeline
        lower_aug = self.transform(image=lower_img)
        higher_aug = self.transform(image=higher_img)

        # Return the transformed images and the rank difference.
        # Optionally, you can also return the file paths if needed.
        return lower_aug["image"], higher_aug["image"], lower_image_path, higher_image_path, rank_difference


class DataPreparation:
    def __init__(self) -> None:
        config = get_config()
        self.batch_size = config["TRAINING"]["BATCH_SIZE"]
        self.train_portion = config["TRAINING"]["TRAIN_PORTION"]
        self.val_portion = config["TRAINING"]["VAL_PORTION"]

    def create_dataloaders(self) -> (DataLoader, DataLoader, DataLoader):
        config = get_config()

        train_datasets = []
        val_datasets = []
        test_datasets = []

        dataset_class = ListImageDataset if config["TRAINING"]["LOSS"] != "PairWise" else PairWiseImageDataset

        for wind_turbine in ["07", "21", "41"]:
            for angle in ["01", "02", "03"]:
                ordered_images = load_progress_ordered(f"WT_{wind_turbine}_SVIV{angle}")

                # Separate train, validation and test
                train_groups, val_groups, test_groups = create_image_splits(
                    ordered_images, self.train_portion, self.val_portion
                )

                # load snowless images
                snowless_images = retrieve_snowless_images(wind_turbine, angle) if config["IMAGE"]["REFERENCE_IMAGE"] else None

                train_data = dataset_class(train_groups, snowless_images, train=True)
                val_data = dataset_class(val_groups, snowless_images, train=False)
                test_data = PairWiseImageDataset(test_groups, snowless_images, train=False)

                train_datasets.append(train_data)
                val_datasets.append(val_data)
                test_datasets.append(test_data)

        # Concatenate all datasets
        transform = test_datasets[0].transform
        train_data = ConcatDataset(train_datasets)
        val_data = ConcatDataset(val_datasets)
        test_data = ConcatDataset(test_datasets)

        print(f"Train data size: {len(train_data)}")
        print(f"Validation data size: {len(val_data)}")
        print(f"Test data size: {len(test_data)}")

        train_val_batch_size = 1 if not config["TRAINING"]["LOSS"] == "PairWise" else self.batch_size

        common_args = {
            "drop_last": False,
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
        }

        train_loader = DataLoader(train_data, batch_size=train_val_batch_size, shuffle=True, **common_args)
        val_loader = DataLoader(val_data, batch_size=train_val_batch_size, shuffle=False, **common_args)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, **common_args)

        return train_loader, val_loader, test_loader, transform



if __name__ == "__main__":
    init_config()
    # create transform, load some random images and see how they look after the transform
    config = get_config()
    transform = define_transform(train=True)
    all_paths = list(load_progress())
    random.shuffle(all_paths)

    # clear directory
    [f.unlink() for f in Path("transformed_images").glob("*") if f.is_file()]

    for i in range(100):
        img_path = all_paths[i]
        img = imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = transform(image=img)
        img = augmented["image"]
        # Convert the tensor back to a PIL image
        img = transforms.ToPILImage()(img)
        # Store the transformed image in a new folder
        img.save(f"transformed_images/{i}.jpg")

