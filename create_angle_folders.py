import json
import os
import shutil
from pathlib import Path


def create_angle_overview(json_file: str, output_dir: str = "angle_overview"):
    """Reads angle-to-image-paths mapping from a JSON file and creates
    one folder per angle, copying images in the specified order.

    Args:
        json_file: Path to the JSON file with structure { angle: [list_of_image_paths], ... }
        output_dir: Directory where angle folders will be created.

    """
    # 1. Load the JSON data
    with open(json_file) as f:
        angle_data = json.load(f)

    # 2. Make the main output directory (e.g. "angle_overview")
    os.makedirs(output_dir, exist_ok=True)

    # 3. For each angle in the JSON, create (or overwrite) a subfolder
    for angle, image_paths in angle_data.items():
        # The subfolder name for this angle
        angle_folder = Path(output_dir) / angle

        # If the folder already exists, remove it
        if angle_folder.exists():
            shutil.rmtree(angle_folder)
        angle_folder.mkdir(parents=True)

        # 4. Copy images into the angle folder, preserving the order
        for idx, img_path in enumerate(image_paths):
            # Convert to Path object, handle slashes
            src_path = Path(img_path)

            # Name the copied file with an index prefix to reflect order
            # e.g. "000_imgA.jpg", "001_imgB.jpg", etc.
            dst_filename = f"{idx:03d}_{src_path.name}"
            dst_path = angle_folder / dst_filename

            # Copy the image (metadata included)
            shutil.copy2(src_path, dst_path)

    print(f"Folders created in '{output_dir}' successfully!")

# Example usage:
if __name__ == "__main__":
    # Suppose your JSON is "angle_orderings.json"
    create_angle_overview("image_labels/hogaliden.json", output_dir="angle_overview")
