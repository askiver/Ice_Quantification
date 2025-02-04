
from pathlib import Path
from config import CONFIG
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as f
import torchvision.transforms as transforms
import random
import pandas as pd
import os
import shutil
from PIL import Image
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
    device = CONFIG["TRAINING"]["DEVICE"]
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No need for gradient tracking
        for higher_img, lower_img, _, _ in dataloader:
            # Move images to the correct device
            higher_img = higher_img.to(device)
            lower_img = lower_img.to(device)

            # Forward pass: Get model predictions
            higher_scores = model(higher_img)  # Scores for "higher" images
            lower_scores = model(lower_img)  # Scores for "lower" images

            # Compare predictions
            correct += torch.sum(higher_scores > lower_scores).item()
            total += higher_img.size(0)  # Batch size

    accuracy = (correct / total) * 100  # Convert to percentage
    print(accuracy)
    return accuracy


def visualize_predictions(model, dataloader, num_samples=5):
    model.eval()
    device = CONFIG["TRAINING"]["DEVICE"]
    transform = transforms.ToPILImage()  # Convert tensors to images

    all_samples = []  # Store all image pairs for random selection

    with torch.no_grad():
        for lower_img, higher_img, _, _ in dataloader:
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

    if len(selected_samples) == 1:
        axes = [axes]  # Ensure axes is iterable when only one sample is chosen

    for i, (img_a, img_b, score_a, score_b) in enumerate(selected_samples):
        img_a = transform(img_a)  # Convert to PIL image
        img_b = transform(img_b)

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
    plt.show()


def evaluate_and_sort_results(model):
    image_paths = load_progress_ordered()
    # flatten list
    image_paths = [item for sublist in image_paths for item in sublist]
    output_csv = "output/results.csv"
    sorted_folder = "output/sorted_images"
    # Ensure evaluation mode
    model.eval()
    device = CONFIG["TRAINING"]["DEVICE"]

    # Define transformations (ensure consistency with training)
    transform = transforms.Compose([
        transforms.Resize((CONFIG["TRAINING"]["IMAGE_DIMENSIONS"]["HEIGHT"],
                           CONFIG["TRAINING"]["IMAGE_DIMENSIONS"]["WIDTH"])),
        transforms.ToTensor(),
    ])

    image_scores = []

    with torch.no_grad():
        for image_path in image_paths:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

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

    # Copy images into sorted folder with ranking in filename
    for rank, (image_path, score) in enumerate(image_scores, start=1):
        image_name = os.path.basename(image_path)
        new_image_name = f"{rank:03d}_{score:.2f}_{image_name}"
        new_image_path = os.path.join(sorted_folder, new_image_name)

        shutil.copy(image_path, new_image_path)  # Copy image to new location
