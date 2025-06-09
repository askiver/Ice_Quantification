import pandas as pd

from label_ordering import load_progress_ordered

if __name__ == "__main__":
    # Need to verify that the labels are correct
    # Load the labeled images
    labeled_images = pd.read_csv("image_labels/labeled_data.csv", header=0)

    # Load the ordered images
    ordered_images = load_progress_ordered()

    # retrieve all image_paths from ordered_images
    ordered_image_paths = []
    for key in ordered_images:
        ordered_image_paths.extend(ordered_images[key])

    for image_path in ordered_image_paths:
        # Check if the image_path is in the labeled images and if the value is not "unknown" or 0
        row = labeled_images.loc[labeled_images["image_path"] == image_path]
        if row.empty:
            raise ValueError(f"No row found for image path: {image_path}")
        snow_value = row["label"].iloc[0]
        try:
            snow_value = int(snow_value)
        except ValueError:
            raise ValueError(f"Could not convert snow_value to int for image path: {image_path}")
        if snow_value == 0:
            raise ValueError(f"Image path {image_path} has snow_value 0, but is in ordered images")

    # Also need to verify that all labeled images are in the ordered images
    # Get all image paths that contain snow
    df_filtered = labeled_images
    df_filtered["snow_value_numeric"] = pd.to_numeric(df_filtered["label"], errors="coerce")
    df_filtered = df_filtered[df_filtered["snow_value_numeric"].astype(float) > 0]
    df_filtered = df_filtered["image_path"].tolist()

    # Check if all labeled images are in the ordered images
    for image_path in df_filtered:
        if image_path not in ordered_image_paths:
            raise ValueError(f"Labeled image path {image_path} is not in ordered images")

