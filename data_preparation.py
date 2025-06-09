import functools
import random
from pathlib import Path

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from cv2 import imread
from PIL import Image, ImageOps
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import functional as F

from anonymize_turbine_number import replace_turbine_number
from config import get_config, init_config
from label_images import load_progress
from label_ordering import load_progress_hogaliden, load_progress_ordered
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
        """size: desired output square size (e.g., 224)
        fill: padding color, default black
        """
        self.size = size
        self.fill = fill

    def __call__(self, img):
        # Use ImageOps.pad to resize the image, preserving aspect ratio,
        # and add padding (if needed) to make it exactly (size, size)
        return ImageOps.pad(img, (self.size, self.size), color=self.fill, centering=(0.5, 0.5))

class ListImageDataset(Dataset):
    def __init__(self, ordered_images_subset, snowless_images=None, train=True):
        config = get_config()

        self.transform = define_transform()

        self.image_ranks = []
        self.image_paths = []
        for idx, image_path in enumerate(reversed(ordered_images_subset)):

            self.image_ranks.append(idx)
            self.image_paths.append(image_path)

        # Convert ranks to tensor
        self.image_ranks = torch.tensor(self.image_ranks, dtype=torch.float32)

    def __len__(self):
        return 1


    def __getitem__(self, idx):
        # Augment all images in the dataset
        images = [self.transform(image=self.load_image(img))["image"] for img in self.image_paths]
        # Convert the list of images to a tensor
        images_tensor = torch.stack(images)

        return images_tensor, self.image_ranks, self.image_paths

    def load_image(self, image_path):
        # Load the image from the path
        img = imread(image_path)
        # Convert to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


def define_transform(train=True):
    config = get_config()

    image_height = config["IMAGE"]["HEIGHT"]
    image_width = config["IMAGE"]["WIDTH"]
    horizontal_flip = config["IMAGE"]["HORIZONTAL_FLIP"]
    brightness = config["IMAGE"]["BRIGHTNESS"]
    contrast = config["IMAGE"]["CONTRAST"]
    saturation = config["IMAGE"]["SATURATION"]
    hue = config["IMAGE"]["HUE"]
    normalize = config["IMAGE"]["NORMALIZE"]


    transform = A.Compose([
        A.LongestMaxSize(max_size=max(image_height, image_width)),
        *([A.HorizontalFlip(p=horizontal_flip)] if train else []),
        *([A.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)] if train else []),
        A.PadIfNeeded(min_height=image_height, min_width=image_width, border_mode=cv2.BORDER_CONSTANT, fill=(0, 0, 0)),
        *([A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))] if normalize else []),
        ToTensorV2(),
    ])

    return transform

class PairWiseImageDataset(Dataset):
    def __init__(self, ordered_images_subset, snowless_images=None, train=True):
        config = get_config()
        dev_run = config["TRAINING"]["QUICK_DEV_RUN"]

        # Generate ordered image pairs

        self.transform = define_transform(train)
        self.test_transform = define_transform(train=False)

        self.pairs = []
        self.snowless_images = snowless_images

        if not train and snowless_images:
            # Only use a single snowless image for testing and evaluating
            self.snowless_images = [random.choice(self.snowless_images)]

        max_rank_index = len(ordered_images_subset) - 1

        for low_idx, lower_image_path in enumerate(ordered_images_subset):
            for high_idx, higher_image_path in enumerate(ordered_images_subset[low_idx + 1 :]):
                rank_difference = high_idx / max_rank_index
                # Leftmost always lower
                self.pairs.append((lower_image_path, higher_image_path, rank_difference))

        # Only use a subset of the dataset for quick dev run
        if dev_run:
            self.pairs = random.sample(self.pairs, min(100, len(self.pairs)))

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

        # Add reference images if specified in the config
        if self.snowless_images is not None:
            # Choose a random snowless image from the list
            #random_snowless_image = self.load_image(random.choice(self.snowless_images))
            random_snowless_image = random.choice(self.snowless_images)
            random_snowless_image = imread(random_snowless_image)
            # Convert to RGB format
            random_snowless_image = cv2.cvtColor(random_snowless_image, cv2.COLOR_BGR2RGB)


            # combine the snowless image with the transformed images
            lower_image = add_reference_image(lower_aug["image"], random_snowless_image, self.test_transform)
            higher_image = add_reference_image(higher_aug["image"], random_snowless_image, self.test_transform)

        else:
            # If no snowless images are provided, just use the transformed images
            lower_image = lower_aug["image"]
            higher_image = higher_aug["image"]

        # Return the transformed images and the rank difference.
        # Optionally, you can also return the file paths if needed.
        return lower_image, higher_image, lower_image_path, higher_image_path, rank_difference


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
        val_pair_datasets = []
        test_datasets = []

        dataset_class = ListImageDataset if config["TRAINING"]["LOSS"] != "PairWise" else PairWiseImageDataset

        for wind_turbine in ["A", "B", "C"]:
            # Change wind turbine to proper number
            turbine_number = replace_turbine_number(wind_turbine)
            for angle in ["01", "02", "03"]:
                ordered_images = load_progress_ordered(f"WT_{turbine_number}_SVIV{angle}")


                if config["DATASET"]["SPLIT"] == "angle":
                    if (wind_turbine == "C" and angle == "02") or (wind_turbine == "B" and angle == "02"):
                        val_data = dataset_class(ordered_images, snowless_images=None, train=False)
                        val_datasets.append(val_data)
                    elif (wind_turbine == "B" and angle == "01") or (wind_turbine == "B" and angle == "03"):
                        test_data = dataset_class(ordered_images, snowless_images=None, train=False)
                        test_datasets.append(test_data)
                    else:
                        train_data = dataset_class(ordered_images, snowless_images=None, train=True)
                        train_datasets.append(train_data)


                elif config["DATASET"]["SPLIT"] == "turbine":
                # Separate train, validation and test based on wind turbine
                    if wind_turbine == "C":
                        train_data = dataset_class(ordered_images, snowless_images=None, train=True)
                        train_datasets.append(train_data)

                    elif wind_turbine == "B":
                        val_data = dataset_class(ordered_images, snowless_images=None, train=False)
                        val_datasets.append(val_data)

                    else:
                        test_data = PairWiseImageDataset(ordered_images, snowless_images=None, train=False)
                        test_datasets.append(test_data)

                else:
                    # Separate train, validation and test
                    train_groups, val_groups, test_groups = create_image_splits(
                        ordered_images, self.train_portion, self.val_portion
                    )

                    # load snowless images
                    snowless_images = retrieve_snowless_images(turbine_number, angle) if config["IMAGE"]["REFERENCE_IMAGE"] else None

                    train_data = dataset_class(train_groups, snowless_images=snowless_images, train=True)
                    val_data = dataset_class(val_groups, snowless_images=snowless_images, train=False)
                    val_data_pair = PairWiseImageDataset(val_groups, snowless_images=snowless_images, train=False)
                    test_data = PairWiseImageDataset(test_groups, snowless_images=snowless_images, train=False)

                    train_datasets.append(train_data)
                    val_datasets.append(val_data)
                    val_pair_datasets.append(val_data_pair)
                    test_datasets.append(test_data)


        # Concatenate all datasets
        transform = test_datasets[0].transform
        train_data = ConcatDataset(train_datasets)
        val_data = ConcatDataset(val_datasets)
        test_data = ConcatDataset(test_datasets)
        if config["TRAINING"]["LOSS"] != "PairWise":
            val_data_pair = ConcatDataset(val_pair_datasets)
        else:
            val_data_pair = None

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

        if config["TRAINING"]["LOSS"] != "PairWise":
            val_loader_pair = DataLoader(val_data_pair, batch_size=self.batch_size, shuffle=False, **common_args)
        else:
            val_loader_pair = None


        return train_loader, val_loader, test_loader, transform, val_loader_pair

    def create_hogaliden_dataloader(self):
        ordered_images = load_progress_hogaliden()

        hogaliden_snow = ["hogaliden/FredO-Cam01_2025-02-12T123000_ZUTC-left.jpg"] if get_config()["IMAGE"]["REFERENCE_IMAGE"] else None

        hogaliden_dataset = PairWiseImageDataset(ordered_images, snowless_images=hogaliden_snow, train=False)
        hogaliden_loader = DataLoader(hogaliden_dataset, batch_size=self.batch_size, shuffle=False)

        return hogaliden_loader


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

    # Create a plot showing the before and after of an image
    img_path = all_paths[0]
    img = imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    augmented = transform(image=img)
    img = augmented["image"]
    # Convert the tensor back to a PIL image
    img = transforms.ToPILImage()(img)
    # Store the transformed image in a new folder
    img.save("transformed_images/0.jpg")
    # Create a plot showing the before and after of an image
    import matplotlib.pyplot as plt
    import numpy as np
    img_before = Image.open(img_path)
    img_after = Image.open("transformed_images/0.jpg")
    img_before = np.array(img_before)
    img_after = np.array(img_after)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img_before)
    ax[0].set_title("Before")
    ax[0].axis("off")
    ax[1].imshow(img_after)
    ax[1].set_title("After")
    ax[1].axis("off")
    plt.show()
    # Create a plot showing the before and after of an image
    # Save the plot
    plt.savefig("transformed_images/plot.jpg")

    # save the two images
    img_before = Image.fromarray(img_before)
    img_after = Image.fromarray(img_after)
    img_before.save("transformed_images/before.jpg")
    img_after.save("transformed_images/after.jpg")


