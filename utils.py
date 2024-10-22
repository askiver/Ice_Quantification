
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as f


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
