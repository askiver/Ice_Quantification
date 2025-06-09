import csv
import os
import random
import re
import shutil
from pathlib import Path, PurePosixPath

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pillow_heif
import torch
import torch.nn.functional as f
from black.trans import defaultdict
from matplotlib.ticker import MaxNLocator
from PIL import Image
from scipy.stats import kendalltau
from torchvision import transforms

import wandb
from anonymize_turbine_number import replace_turbine_number
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


def evaluate_model_accuracy(model, dataloader, device):
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
            img_a = img_a[:3, :, :] * torch.tensor([0.5] * 3).view(3, 1, 1) + torch.tensor([0.5] * 3).view(3, 1, 1)
            img_b = img_b[:3, :, :] * torch.tensor([0.5] * 3).view(3, 1, 1) + torch.tensor([0.5] * 3).view(3, 1, 1)

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


def evaluate_and_sort_results(model, transform, mean, std, test_loader=None):
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

    # Find all unique angles in the image paths
    unique_angles = set()
    for path in image_paths:
        unique_angles.add(extract_angle(path))

    no_snow_image_paths = []
    for angle in unique_angles:
        m = re.search(r"WT_(\d+)_SVIV(\d+)", angle)
        turbine_number, angle = m.groups()
        no_snow_image_paths.extend(retrieve_snowless_images(turbine_number, angle, 100))

    # Add snowless images to the image paths
    image_paths.extend(no_snow_image_paths)

    image_scores = torch.empty((len(image_paths)),device=device)

    snowless_dict = {}

    for windturbine in ["A", "B", "C"]:
        for angle in ["01", "02", "03"]:
            turbine_number = replace_turbine_number(windturbine)
            snowless_image = retrieve_snowless_images(turbine_number, angle, num_images=1)[0]
            snowless_dict[f"WT_{turbine_number}_SVIV{angle}"] = snowless_image

    with torch.no_grad():
        for idx, image_path in enumerate(image_paths):
            # Load and preprocess image
            image = cv2.imread(image_path)
            # convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_aug = transform(image=image)
            image_tensor = image_aug["image"]

            if reference_image:
                # Retrieve correct angle
                snowless_path = snowless_dict.get(extract_angle(image_path))
                img = cv2.imread(snowless_path)
                # Convert to RGB format
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                image_tensor = add_reference_image(image_tensor, img, transform)

            # add batch dimension and move image to device
            image_tensor = image_tensor.unsqueeze(0).to(device)

            # Get model prediction
            score = model(image_tensor).item() # Convert tensor output to scalar

            # Store result
            image_scores[idx] = score

    # combine image paths and scores
    image_scores = list(zip(image_paths, image_scores.cpu().numpy(), strict=False))

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

    model_normalized_scores = defaultdict(list)

    # load own scores
    df = pd.read_csv("image_labels/labeled_data.csv", index_col=0)

    image_list = []

    # Copy images into sorted folder with ranking in filename
    for rank, (image_path, score) in enumerate(image_scores, start=1):
        image_name = os.path.basename(image_path)

        normalized_score = (score - mean) / std

        turbine_angle = image_name[-16:-4]

        value = df.loc[image_path].iloc[0]
        try:
            value = int(value)
        except ValueError:
            print(image_path)
            continue

        model_normalized_scores[turbine_angle].append(
            (normalized_score, value)
        )

        new_image_name = f"{rank:03d}_{score:.2f}_{image_name}"
        new_image_path = os.path.join(sorted_folder, new_image_name)

        shutil.copy(image_path, new_image_path)  # Copy image to new location

        # Add image to list for visualization
        image = Image.open(image_path)
        image_list.append(
            wandb.Image(image, caption=f"Rank: {rank}, Score: {score:.2f}, Normalized Score: {normalized_score:.3f}")
        )

    # Sample some images for visualization
    image_list = random.sample(image_list, min(108, len(image_list)))

    # log images to wandb
    wandb.log({"sorted_images": image_list})

    # Compare scores for different angles
    compare_scores(model_normalized_scores)

    # Compare scores for different human scores
    compare_quantities(model_normalized_scores)


def compare_scores(score_dict):
    raw_angles = sorted(score_dict.keys())
    means = []
    stds = []
    num_images = []
    for a in raw_angles:
        pairs = score_dict[a]
        scores = [s for s, _ in pairs]
        t = torch.tensor(scores, dtype=torch.float)
        means.append(t.mean().item())
        stds.append(t.std().item())
        num_images.append(len(scores))

    # Replace turbine numbers with anonymized labels
    angles = [replace_turbine_number(angle) for angle in raw_angles]

    angles = [simplify_angle_label(a) for a in angles]

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
    plt.xticks(rotation=0, ha="center")
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha="center")  # Explicit rotation

    # Legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Save plot to file and W&B
    plt.savefig("output/angle_analysis.png")
    wandb.log({"angle_analysis": wandb.Image("output/angle_analysis.png")})

    # Save means and stds using weights and biases
    # create a table
    table = wandb.Table(columns=["Angle", "Mean Score", "Std Dev", "Number of Images"])
    for angle, mean, std, num in zip(angles, means, stds, num_images, strict=False):
        table.add_data(angle, mean, std, num)
    wandb.log({"angle_analysis_table": table})



def compare_quantities(score_dict):
    human_scores = defaultdict(list)

    for key, value in score_dict.items():
        for model_score, human_score in value:
            human_scores[human_score].append(model_score)

    labels_list = sorted(human_scores.keys())
    means = []
    stds = []
    num_images = []

    cat_names = ["None", "Low", "Medium", "High"]
    x = np.arange(len(cat_names))

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

    # Set x-ticks to the categorical names
    ax1.set_xticks(x)
    ax1.set_xticklabels(cat_names, rotation=0, ha="center")

    # 4. Labeling & Titles
    ax1.set_xlabel("Human Label")  # It's not an angle now, but the label category
    ax1.set_ylabel("Number of Images", color="blue")
    ax2.set_ylabel("Mean Model Score", color="red")
    plt.title("Analysis of Human Label vs. Model Score")

    # Combine legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # 5. Save the plot & log to W&B
    plt.tight_layout()
    plt.savefig("output/metric_analysis.png")
    wandb.log({"metric_analysis": wandb.Image("output/metric_analysis.png")})

    # Save means and stds using weights and biases
    # create a table
    table = wandb.Table(columns=["Human Label", "Mean Score", "Std Dev", "Number of Images"])
    for label, mean, std, num in zip(labels_list, means, stds, num_images, strict=False):
        table.add_data(label, mean, std, num)
    wandb.log({"metric_analysis_table": table})


def create_image_splits(ordered_images, train_ratio, val_ratio):
    if train_ratio + val_ratio >= 1:
        raise ValueError("Train and validation ratios must sum to less than 1.")

    rng = random.Random(0)

    indices = list(range(len(ordered_images)))

    # Shuffle indices randomly
    rng.shuffle(indices)

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


def retrieve_snowless_images(windturbine, angle, num_images=None):
    # Load CSV into DataFrame with columns [img_path, value]
    df = pd.read_csv("image_labels/labeled_data.csv")

    windmill_str = f"WT_{windturbine}_SVIV{angle}"

    # Filter rows where 'img_path' contains windmill_str and 'value' == "0"
    df_filtered = df[df["image_path"].str.contains(windmill_str) & (df["label"] == "0")]
    snowless_images = []
    for raw_path in df_filtered["image_path"]:
        snowless_images.append(raw_path)

    if num_images:
        # sample randomly
        snowless_images = random.sample(snowless_images, num_images)
    return snowless_images


def add_reference_image(original_image:torch.tensor, snowless_image, transform):

    # Apply the transformation pipeline
    snowless_aug = transform(image=snowless_image)

    # combine images
    return torch.cat([original_image, snowless_aug["image"]], 0)

def show_label_counts():
    # Load labeled data
    labeled_images = pd.read_csv("image_labels/labeled_data.csv")

    labeled_images["angle"] = labeled_images["image_path"].apply(extract_angle)
    angles = sorted(labeled_images["angle"].unique())

    counts = labeled_images.groupby(["angle", "label"]).size().unstack(fill_value=0)

    cats = ["None", "Low", "Medium", "High", "Unknown"]

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
        bars = ax.bar(bar_positions, bar_values, width, label=cats[i])

        # Annotate each bar with its value
        #for pos, val in zip(bar_positions, bar_values):
            #ax.text(pos, val + 0.05, str(val), ha="center", va="bottom", fontsize=8)

    # Change angle labels to anonymization
    angles = [replace_turbine_number(angle) for angle in angles]

    angles = [simplify_angle_label(a) for a in angles]

    # Configure x-axis
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(angles)

    ax.set_xlabel("Angle")
    ax.set_ylabel("Count of Images")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Label Counts per Angle")
    ax.legend()

    plt.tight_layout()

    # Save plot to file and W&B
    plt.savefig("output/label_counts.png")
    #wandb.log({"label_counts": wandb.Image("output/label_counts.png")})


def extract_angle(path):
    match = re.search(r"(WT_\d+_SVIV\d+)", path)
    return match.group(1) if match else "Unknown"

def kendall_tau(model,test_loader, device, pair=True):
    model.eval()
    if pair:
        concordant = torch.zeros((), dtype=torch.long, device=device)
        total = torch.zeros((), dtype=torch.long, device=device)
        with torch.no_grad():
            for batch in test_loader:
                # Unpack the batch: we only need the first two items (left_img, right_img)
                left_img, right_img, *rest = batch

                # Move to device
                left_img = left_img.to(device)
                right_img = right_img.to(device)

                # Forward pass
                left_scores = model(left_img).squeeze()  # shape: (batch_size,)
                right_scores = model(right_img).squeeze()  # shape: (batch_size,)

                # Count how many pairs the model got "correct"

                # Update totals
                concordant += (left_scores < right_scores).sum()
                total += left_scores.numel()

        c = concordant.item()
        n = total.item()
        # Kendall's tau = (C − D) / N = (C − (N−C)) / N = (2C/N) − 1
        return (2*c/n) - 1
    all_model_scores = []
    all_human_ranks = []
    with torch.no_grad():
        for images, * _ in test_loader:
            # ---- unpack your batch here ----
            # If loader returns (images, labels, ...) but labels are just placeholders,
            # do something like:
            # images, *rest = batch
            images = images.squeeze(0)  # Remove batch dimension if necessary

            # move to device and forward
            images = images.to(device)
            outputs = model(images)  # shape: (B,1) or (B,)
            scores = outputs.squeeze().cpu().tolist()

            # human ranks = [0, 1, 2, ..., B-1] since the batch is ordered
            ranks = list(range(len(scores)))

            all_model_scores.extend(scores)
            all_human_ranks.extend(ranks)

    # compute Kendall’s tau
    tau, p_value = kendalltau(all_human_ranks, all_model_scores)
    return tau


def test_model_on_snow_scenes(snow_folder, model, transform, mean, std):
    config = get_config()
    device = config["TRAINING"]["DEVICE"]

    # Set model to eval mode, move to correct device
    model.eval()
    model.to(device)

    # Gather image paths
    image_paths = []
    for root, _, files in os.walk(snow_folder):
        for file in files:
            image_paths.append(os.path.join(root, file))

    results = []

    # Register the HEIF plugin with Pillow
    pillow_heif.register_heif_opener()

    with torch.no_grad():
        for img_path in image_paths:
            # Convert using Pillow to support HEIC and HEIF
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
            # Apply transforms
            tensor_aug = transform(image=img)
            tensor_img = tensor_aug["image"].unsqueeze(0).to(device)

            # Forward pass
            output = model(tensor_img)
            # If your model returns a single float per image, we can do:
            snow_val = output.squeeze().item()
            # e.g., if output.shape == (1,1), then output.squeeze().item() is a float

            results.append((img_path, snow_val))

        # Sort by predicted snow amount (descending: most snow first)
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

        # Build a list of wandb.Image objects with captions
        image_list = []
        for path, score in sorted_results:
            # caption includes the file name + the predicted score
            normalized_score = (score - mean) / std
            pil_img = Image.open(path).convert("RGB")
            caption_text = f"{Path(path).name} | Score={score:.2f}, Normalized Score={normalized_score:.2f}"
            image_list.append(wandb.Image(pil_img, caption=caption_text))

        # Log the list of images to W&B
        wandb.log({"scene_images": image_list})

        print(f"Analysis complete. {len(sorted_results)} images processed.")



def calculate_mean_and_std(model, dataloader, transform, device):

    # retrieve all unique images from the set
    flattened_set = set()
    for _, _, lower_img_paths, _, _ in dataloader:
        for i in range(len(lower_img_paths)):
            flattened_set.add(lower_img_paths[i])

    snowless_dict = {}

    for windturbine in ["A", "B", "C"]:
        for angle in ["01", "02", "03"]:
            turbine_number = replace_turbine_number(windturbine)
            snowless_image = retrieve_snowless_images(turbine_number, angle, num_images=1)[0]
            snowless_dict[f"WT_{turbine_number}_SVIV{angle}"] = snowless_image

    # Convert the set to a list
    image_paths = list(flattened_set)
    all_outputs = torch.empty((len(image_paths)), device=device)
    for idx, image_path in enumerate(image_paths):
        # Load and preprocess image
        image = cv2.imread(image_path)
        # convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_aug = transform(image=image)
        image_tensor = image_aug["image"]

        # Add reference image if specified
        if get_config()["IMAGE"]["REFERENCE_IMAGE"]:
            # Retrieve correct angle
            snowless_path = snowless_dict.get(extract_angle(image_path))
            img = cv2.imread(snowless_path)
            # Convert to RGB format
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            image_tensor = add_reference_image(image_tensor, img, transform)

        # Add batch dimension and move image to device
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Get model prediction
        output = model(image_tensor)
        all_outputs[idx] = output.item()

    # Calculate mean and std
    mean = torch.mean(all_outputs)
    std = torch.std(all_outputs)
    return mean, std


def simplify_angle_label(angle: str) -> str:
    """Turn "WT_A_SVIV01" into "WT-A-01", etc.
    """
    m = re.match(r"^(WT)_([A-Z])_SVIV0*(\d+)$", angle)
    if m:
        wt, turbine, num = m.groups()
        return f"{wt}-{turbine}-{num}"
    # fallback: replace underscores with dashes
    return angle.replace("_","-")


if __name__ == "__main__":
    show_label_counts()
