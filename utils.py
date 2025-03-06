import csv
import os
import random
import re
import shutil
from pathlib import Path

import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import torch
import torch.nn.functional as f
from black.trans import defaultdict
from PIL import Image

import wandb
from config import get_config
from label_ordering import load_progress_ordered


def show_autoencoder_results(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_images: int = 100,
    images_per_row: int = 10,
    save_dir: str = "plots",
) -> None:
    model.eval()  # Set the model to evaluation mode
    device = next(model.parameters()).device  # Get the device model is on
    all_originals = []
    all_reconstructions = []
    class_errors = {i: [] for i in range(10)}  # Assuming 10 classes (0-9 for MNIST-like datasets)

    with torch.no_grad():  # Disable gradient calculation for evaluation
        # Collect images up to num_images
        images_collected = 0
        for batch in test_loader:
            test_images, labels = batch  # Get the images and labels
            test_images = test_images.to(device)  # Move images to the same device as the model

            # Pass the test images through the autoencoder
            reconstructed_images = model(test_images)

            # Move the data back to CPU for visualization and error calculation
            test_images = test_images.cpu()
            reconstructed_images = reconstructed_images[0].cpu()

            # Calculate reconstruction error (MSE)
            for i in range(test_images.size(0)):
                original = test_images[i]
                reconstructed = reconstructed_images[i]
                mse_error = f.binary_cross_entropy(reconstructed, original).item()  # MSE error for this image
                class_errors[labels[i].item()].append(mse_error)  # Store error for the corresponding class

            # Append images to the lists for visualization
            all_originals.append(test_images)
            all_reconstructions.append(reconstructed_images)

            images_collected += test_images.size(0)
            if images_collected >= num_images:
                break  # Stop collecting after reaching num_images

    # Concatenate all images
    all_originals = torch.cat(all_originals, dim=0)[:num_images]
    all_reconstructions = torch.cat(all_reconstructions, dim=0)[:num_images]

    # Calculate average reconstruction error for each class
    average_class_errors = {
        cls: (sum(errors) / len(errors)) if len(errors) > 0 else 0 for cls, errors in class_errors.items()
    }

    # Create a bar chart for average reconstruction error per class
    classes = list(average_class_errors.keys())
    avg_errors = list(average_class_errors.values())

    plt.figure(figsize=(10, 6))
    plt.bar(classes, avg_errors, color="skyblue")
    plt.xlabel("Class")
    plt.ylabel("Average Reconstruction Error (MSE)")
    plt.title("Average Reconstruction Error by Class")
    plt.xticks(classes)  # Display class labels on the x-axis
    plt.tight_layout()

    # Ensure the save directory exists
    Path.mkdir(Path(save_dir), parents=True, exist_ok=True)
    plt.savefig(Path(save_dir) / "autoencoder_class_errors.png")
    plt.show()

    # Visualize original and reconstructed images (same as before)
    num_rows = (num_images + images_per_row - 1) // images_per_row  # Ceiling division
    fig, axes = plt.subplots(nrows=num_rows * 2, ncols=images_per_row, figsize=(images_per_row * 2, num_rows * 4))

    for idx in range(num_images):
        row = idx // images_per_row
        col = idx % images_per_row

        # Original image
        axes[row * 2, col].imshow(all_originals[idx].squeeze(), cmap="gray")
        axes[row * 2, col].set_title(f"Original {idx}")
        axes[row * 2, col].axis("off")

        # Reconstructed image
        axes[row * 2 + 1, col].imshow(all_reconstructions[idx].squeeze(), cmap="gray")
        axes[row * 2 + 1, col].set_title(f"Reconstructed {idx}")
        axes[row * 2 + 1, col].axis("off")

    plt.tight_layout()

    # Save the results
    plt.savefig(Path(save_dir) / "autoencoder_results.png")
    plt.show()


def evaluate_model_accuracy(model, dataloader):
    config = get_config()
    device = config["TRAINING"]["DEVICE"]
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No need for gradient tracking
        for lower_img, higher_img, _, _, _ in dataloader:
            # Move images to the correct device
            higher_img = higher_img.to(device)
            lower_img = lower_img.to(device)

            # Forward pass: Get model predictions
            higher_scores = model(higher_img).squeeze()  # Scores for "higher" images
            lower_scores = model(lower_img).squeeze()  # Scores for "lower" images

            # Compare predictions
            correct += (higher_scores > lower_scores).sum().item()
            total += higher_img.size(0)  # Batch size

    accuracy = (correct / total) * 100  # Convert to percentage
    print(accuracy)
    return accuracy


def visualize_predictions(model, dataloader, num_samples=5):
    model.eval()
    config = get_config()
    device = config["TRAINING"]["DEVICE"]
    output_folder = "output/examples.png"
    transform = transforms.ToPILImage()  # Convert tensors to images

    all_samples = []  # Store all image pairs for random selection

    with torch.no_grad():
        for lower_img, higher_img, _, _, _ in dataloader:
            higher_img = higher_img.to(device)
            lower_img = lower_img.to(device)

            # Get model predictions
            higher_scores = model(higher_img).cpu().numpy()
            lower_scores = model(lower_img).cpu().numpy()

            # Store image pairs along with their scores
            for i in range(higher_img.shape[0]):
                all_samples.append((higher_img[i].cpu(), lower_img[i].cpu(), higher_scores[i][0], lower_scores[i][0]))

    # Randomly sample `num_samples` pairs
    selected_samples = random.sample(all_samples, min(num_samples, len(all_samples)))

    # Plot selected pairs
    fig, axes = plt.subplots(len(selected_samples), 2, figsize=(8, 4 * len(selected_samples)))

    fig.suptitle("Model Predictions \nLeft has more snow", fontsize=16)

    if len(selected_samples) == 1:
        axes = [axes]  # Ensure axes is iterable when only one sample is chosen

    for i, (img_a, img_b, score_a, score_b) in enumerate(selected_samples):
        # Revert normalization
        if config["IMAGE"]["NORMALIZE"]:
            img_a = img_a * torch.tensor([0.5] * 3).view(3, 1, 1) + torch.tensor([0.5] * 3).view(3, 1, 1)
            img_b = img_b * torch.tensor([0.5] * 3).view(3, 1, 1) + torch.tensor([0.5] * 3).view(3, 1, 1)

        img_a = transform(
            img_a[:3, :, :]
        )  # Convert to PIL image, Only retrieve the first 3 channels if reference image has been added
        img_b = transform(img_b[:3, :, :])

        correct = score_a > score_b  # Check correctness

        # Display higher-ranked image
        axes[i][0].imshow(img_a)
        axes[i][0].axis("off")
        axes[i][0].set_title(f"Score: {score_a:.2f}", color="green" if correct else "red")

        # Display lower-ranked image
        axes[i][1].imshow(img_b)
        axes[i][1].axis("off")
        axes[i][1].set_title(f"Score: {score_b:.2f}", color="green" if correct else "red")

    plt.tight_layout()

    plt.savefig(output_folder)

    wandb.log({"examples": [wandb.Image(output_folder)]})


def evaluate_and_sort_results(model, transform, test_loader=None):
    if not test_loader:
        image_paths = load_progress_ordered()
        # flatten list
        image_paths = [item for sublist in image_paths for item in sublist]

    else:
        flattened_set = set()
        for _, _, lower_img_paths, _, _ in test_loader:
            for i in range(len(lower_img_paths)):
                flattened_set.add(lower_img_paths[i])
        image_paths = list(flattened_set)
    output_csv = "output/results.csv"
    sorted_folder = "output/sorted_images"
    # Ensure evaluation mode
    model.eval()
    config = get_config()
    device = config["TRAINING"]["DEVICE"]
    reference_image = config["IMAGE"]["REFERENCE_IMAGE"]

    # Define transformations (ensure consistency with training)

    image_scores = []

    with torch.no_grad():
        for image_path in image_paths:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image)

            if reference_image:
                image_tensor = add_reference_image(image_tensor, transform)

            # add batch dimension and move image to device
            image_tensor = image_tensor.unsqueeze(0).to(device)

            # Get model prediction
            score = model(image_tensor).cpu().item()  # Convert tensor output to scalar

            # Store result
            image_scores.append((image_path, score))

    # Sort images by predicted score (highest first)
    image_scores.sort(key=lambda x: x[1], reverse=True)

    # Save results to CSV
    df = pd.DataFrame(image_scores, columns=["Image Path", "Predicted Score"])
    df.to_csv(output_csv, index=False)
    print(f"Saved scores to {output_csv}")

    # Ensure the sorted images folder is clean (delete old images)
    if os.path.exists(sorted_folder):
        for file in os.listdir(sorted_folder):
            file_path = os.path.join(sorted_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Remove old images
    else:
        os.makedirs(sorted_folder)  # Create folder if it doesn't exist

    # Normalize scores using Z-score normalization

    mean = torch.mean(torch.tensor([score for _, score in image_scores]))
    std = torch.std(torch.tensor([score for _, score in image_scores]))

    model_normalized_scores = defaultdict(list)

    # load own scores
    df = pd.read_csv("image_labels/labeled_data.csv", index_col=0)

    image_list = []

    # Copy images into sorted folder with ranking in filename
    for rank, (image_path, score) in enumerate(image_scores, start=1):
        image_name = os.path.basename(image_path)

        normalized_score = (score - mean) / std

        turbine_angle = image_name[-16:-4]

        model_normalized_scores[turbine_angle].append(
            (normalized_score, int(df.loc[image_path.replace("/", "\\")].iloc[0]))
        )

        new_image_name = f"{rank:03d}_{score:.2f}_{image_name}"
        new_image_path = os.path.join(sorted_folder, new_image_name)

        shutil.copy(image_path, new_image_path)  # Copy image to new location

        # Add image to list for visualization
        image = Image.open(image_path)
        image_list.append(
            wandb.Image(image, caption=f"Rank: {rank}, Score: {score:.2f}, Normalized Score: {normalized_score:.3f}")
        )

    # log images to wandb
    wandb.log({"sorted_images": image_list})

    # Compare scores for different angles
    compare_scores(model_normalized_scores)

    # Compare scores for different human scores
    compare_quantities(model_normalized_scores)


def compare_scores(score_dict):
    angles = sorted(score_dict.keys())
    means = []
    stds = []
    num_images = []
    for angle, score_pairs in score_dict.items():
        scores = [score for score, _ in score_pairs]
        means.append(torch.mean(torch.tensor(scores)).item())
        stds.append(torch.std(torch.tensor(scores)).item())
        num_images.append(len(scores))

    # Create figure with two subplots
    fig, ax1 = plt.subplots(figsize=(14, 6))  # Adjust figure size

    # Bar plot for number of images
    ax1.bar(angles, num_images, alpha=0.6, color="blue", label="Number of Images")

    # Secondary y-axis for mean and standard deviation
    ax2 = ax1.twinx()
    ax2.errorbar(angles, means, yerr=stds, fmt="o", color="red", label="Mean ± Std Dev")

    # Labels and title
    ax1.set_xlabel("Angle")
    ax1.set_ylabel("Number of Images", color="blue")
    ax2.set_ylabel("Mean Score", color="red")
    plt.title("Analysis of Image Angles")

    # Rotate x-axis labels properly using plt.setp()
    plt.xticks(rotation=30, ha="right")
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")  # Explicit rotation

    # Legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Save plot to file and W&B
    plt.savefig("output/angle_analysis.png")
    wandb.log({"angle_analysis": wandb.Image("output/angle_analysis.png")})


def compare_quantities(score_dict):
    human_scores = defaultdict(list)

    for key, value in score_dict.items():
        for model_score, human_score in value:
            human_scores[human_score].append(model_score)

    labels_list = sorted(human_scores.keys())
    means = []
    stds = []
    num_images = []

    for label in labels_list:
        scores_tensor = torch.tensor(human_scores[label], dtype=torch.float)
        num_images.append(len(scores_tensor))
        means.append(scores_tensor.mean().item())
        stds.append(scores_tensor.std().item())

    # 3. Create the figure & axes
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot the bar chart (left y-axis)
    ax1.bar(labels_list, num_images, alpha=0.6, color="blue", label="Number of Images")

    # Create a secondary y-axis for mean ± std dev
    ax2 = ax1.twinx()
    ax2.errorbar(labels_list, means, yerr=stds, fmt="o", color="red", label="Mean ± Std Dev")

    # 4. Labeling & Titles
    ax1.set_xlabel("Human Label")  # It's not an angle now, but the label category
    ax1.set_ylabel("Number of Images", color="blue")
    ax2.set_ylabel("Mean Model Score", color="red")
    plt.title("Analysis of Human Label vs. Model Score")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=30, ha="right")

    # Combine legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # 5. Save the plot & log to W&B
    plt.tight_layout()
    plt.savefig("output/metric_analysis.png")
    wandb.log({"metric_analysis": wandb.Image("output/metric_analysis.png")})


def create_image_splits(ordered_images, train_ratio, val_ratio):
    if train_ratio + val_ratio >= 1:
        raise ValueError("Train and validation ratios must sum to less than 1.")

    indices = list(range(len(ordered_images)))

    # Shuffle indices randomly
    random.shuffle(indices)

    # Compute split sizes
    train_size = int(len(indices) * train_ratio)
    val_size = int(len(indices) * val_ratio)

    # Perform random splits
    train_indices = sorted(indices[:train_size])  # Sort to preserve relative order
    val_indices = sorted(indices[train_size : train_size + val_size])
    test_indices = sorted(indices[train_size + val_size :])

    # Use indices to extract groups while maintaining original ordering
    train_groups = [ordered_images[i] for i in train_indices]
    val_groups = [ordered_images[i] for i in val_indices]
    test_groups = [ordered_images[i] for i in test_indices]

    return train_groups, val_groups, test_groups


def retrieve_snowless_images(windmill, angle):
    snowless_images = []
    with open("image_labels/labeled_data.csv") as file:
        reader = csv.reader(file)
        for row in reader:
            img_path, value = row[0], row[1]

            windmill_str = f"WT_{windmill}_SVIV{angle}"

            if windmill_str in img_path and value == "0":
                snowless_images.append(Path(img_path.replace("\\", "/")).resolve())

    return snowless_images


def add_reference_image(original_image:torch.tensor, transform, snowless_images=None):
    if not snowless_images:
        snowless_images = retrieve_snowless_images(41, "03")
    # Choose random image as reference image
    reference_image = random.choice(snowless_images)
    reference_image = transform(Image.open(reference_image).convert("RGB"))

    # combine images
    return torch.cat([original_image, reference_image], 0)

def show_label_counts():
    # Load labeled data
    labeled_images = pd.read_csv("image_labels/labeled_data.csv")

    labeled_images["angle"] = labeled_images["image_path"].apply(extract_angle)
    angles = sorted(labeled_images["angle"].unique())

    counts = labeled_images.groupby(["angle", "label"]).size().unstack(fill_value=0)


    labels = counts.columns.tolist()  # e.g. ["catA", "catB", "catC", "catD"]
    x = np.arange(len(angles))  # positions for angles
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot each category
    for i, lbl in enumerate(labels):
        # bar positions offset by i * width
        bar_positions = x + i * width
        bar_values = counts[lbl]

        # Create bars
        bars = ax.bar(bar_positions, bar_values, width, label=lbl)

        # Annotate each bar with its value
        for pos, val in zip(bar_positions, bar_values):
            ax.text(pos, val + 0.05, str(val), ha="center", va="bottom", fontsize=8)

    # Configure x-axis
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(angles, rotation=45, ha="right")

    ax.set_xlabel("Angle")
    ax.set_ylabel("Count of Images")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Label Counts per Angle (with annotation)")
    ax.legend()

    plt.tight_layout()

    # Save plot to file and W&B
    plt.savefig("output/label_counts.png")
    wandb.log({"label_counts": wandb.Image("output/label_counts.png")})


def extract_angle(path):
    match = re.search(r"(WT_\d+_SVIV\d+)", path)
    return match.group(1) if match else "Unknown"


if __name__ == "__main__":
    pass
