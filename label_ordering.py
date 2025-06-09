import json
import random
import sys
from pathlib import Path

import cv2
import pandas as pd

from label_images import resize_image

# Define paths
IMAGES_FOLDER = "images"  # Folder containing the images
OUTPUT_CSV = "image_labels/order_labels.csv"
# SELECTED_ANGLE = "WT_41_SVIV03"
ANGLES = ["01", "02", "03"]
WIND_TURBINES = ["07", "21", "41"]

# Select fitting key inputs
KEY_TO_COMMAND = {
    "a": ">",
    "d": "<",
    "t": "=",
    "q": "quit",
}


# Only select images that contain snow
def filter_images(ordered_dict):
    # load csv into dataframe
    df = pd.read_csv("image_labels/labeled_data.csv")

    # Filter rows where 'label' is not 'unknown' and is > 0
    # Convert label to int before comparison
    df = df[df["label"] != "unknown"]
    df = df[df["label"].astype(int) > 0]

    # find all image paths that are in the ordered dict
    ordered_paths = set()
    for angle in ordered_dict:
        for image_path in ordered_dict[angle]:
            ordered_paths.add(image_path)

    # Remove images that are already ordered
    df = df[~df["image_path"].isin(ordered_paths)]

    # Convert image_path column to Path and then to posix, storing as a set
    labelled_images = set(df["image_path"].apply(lambda x: Path(x).as_posix()))

    return labelled_images


# Load progress from CSV
def load_progress_ordered(chosen_angle=None):
    """ordered_images = []
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                if row:
                    current_row = [Path(item.replace("\\", "/")).resolve() for item in row]
                    ordered_images.append(current_row)
    """
    with open("image_labels/order_labels.json") as f:
        rankings = json.load(f)

    for turbine in WIND_TURBINES:
        for angle in ANGLES:
            turbine_angle = f"WT_{turbine}_SVIV{angle}"
            if turbine_angle not in rankings:
                rankings[turbine_angle] = []

    if chosen_angle:
        return rankings[chosen_angle]

    return rankings

def load_progress_hogaliden():
    with open("image_labels/hogaliden.json") as f:
        rankings = json.load(f)

    return rankings["hogaliden"]


# Save progress to CSV
def save_progress(ordered_images):
    """With open(OUTPUT_CSV, "w", newline="") as file:
    writer = csv.writer(file)
    # Convert tuples to strings for saving
    for item in ordered_images:
        writer.writerow(item)
    """
    with open("image_labels/order_labels.json", "w") as f:
        json.dump(ordered_images, f, indent=4)


# Display images and get user input
def compare_images(image_a, image_b):
    img_a = cv2.imread(image_a)
    img_b = cv2.imread(image_b)

    img_a_resized = resize_image(img_a, 1200, 1200)
    img_b_resized = resize_image(img_b, 1200, 1200)

    # Create a single window with two images
    combined_image = cv2.hconcat([img_a_resized, img_b_resized])
    cv2.imshow("Compare Images: Left (A) vs Right (B)", combined_image)

    print(f"Compare:\n  A: {image_a}\n  B: {image_b}")

    while True:
        key = cv2.waitKey(0)
        key_char = chr(key & 0xFF)

        if key_char in KEY_TO_COMMAND:
            return KEY_TO_COMMAND[key_char]
        print(f"Invalid input. Please press one of {KEY_TO_COMMAND.keys()}.")


# Insert a new image into the ordered list
def insert_image(new_image, ordered_images):
    if not ordered_images:
        ordered_images.append(new_image)
        return

    # Binary search for the appropriate position
    low, high = 0, len(ordered_images)
    while low < high:
        mid = (low + high) // 2
        current = ordered_images[mid]

        relation = compare_images(new_image, ordered_images[mid])

        match relation:
            case ">":
                print(f"{new_image} has more snow")
                low = mid + 1

            case "<":
                print(f"{current} has more snow")
                high = mid

            case "=":
                print("There is a tie")
                # Randomly choose one of the two
                # Ends the comparison immediately
                if random.choice([True, False]):
                    low, high = mid, mid
                else:
                    low, high = mid + 1, mid + 1

            case "quit":
                print("Exiting program")
                sys.exit(0)

    # Insert the new image in the correct position
    ordered_images.insert(low, new_image)


def retrieve_angle(image_path):
    return image_path[-16:-4]


# Main function
def label_orderings():
    # Load already ordered images
    ordered_dict = load_progress_ordered()

    # Get all images from the folder
    # images = get_all_image_paths(IMAGES_FOLDER)
    images = filter_images(ordered_dict)

    for idx, new_image in enumerate(images):
        # Retrieve angle from image path
        relevant_angle = retrieve_angle(new_image)

        # retrieve relevant list
        ordered_images = ordered_dict[relevant_angle]
        if new_image in ordered_images:
            continue  # Skip already ordered images

        print(f"Images left to label: {len(images) - idx}")

        # Compare the new image with the ordered list
        insert_image(new_image, ordered_images)

        # Save progress after each insertion
        save_progress(ordered_dict)


if __name__ == "__main__":
    label_orderings()
