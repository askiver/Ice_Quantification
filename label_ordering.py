import os
import cv2
import csv
from label_images import get_all_image_paths, resize_image
import sys

# Define paths
IMAGES_FOLDER = "images"  # Folder containing the images
OUTPUT_CSV = "image_labels/order_labels.csv"

# Select fitting key inputs
KEY_TO_COMMAND = {
    "a": ">",
    "d": "<",
    "t": "=",
    "q": "quit",
}

# Load progress from CSV
def load_progress():
    ordered_images = []
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                for item in row:  # Process each item in the row
                    if item.startswith("("):
                        ordered_images.append(eval(item))  # Convert back to tuple
                    else:
                        ordered_images.append(item)

    return ordered_images

# Save progress to CSV
def save_progress(ordered_images):
    with open(OUTPUT_CSV, "w", newline="") as file:
        writer = csv.writer(file)
        # Convert tuples to strings for saving
        for item in ordered_images:
            writer.writerow([str(item)])

# Display images and get user input
def compare_images(image_a, image_b):
    img_a = cv2.imread(image_a)
    img_b = cv2.imread(image_b)

    img_a_resized = resize_image(img_a, 800, 800)
    img_b_resized = resize_image(img_b, 800, 800)

    # Create a single window with two images
    combined_image = cv2.hconcat([img_a_resized, img_b_resized])
    cv2.imshow("Compare Images: Left (A) vs Right (B)", combined_image)

    print(f"Compare:\n  A: {image_a}\n  B: {image_b}")

    while True:
        key = cv2.waitKey(0)
        key_char = chr(key & 0xFF)

        if key_char in KEY_TO_COMMAND.keys():
            return KEY_TO_COMMAND[key_char]
        else:
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

        # Compare the new image with the current position
        if isinstance(current, tuple):  # Handle tied images
            relation = compare_images(new_image, current[0])
        else:
            relation = compare_images(new_image, current)

        match relation:
            case ">":
                print(f"{new_image} has more snow")
                low = mid + 1

            case "<":
                print(f"{current} has more snow")
                high = mid

            case "=":
                print("There is a tie")
                # Merge into the existing tie group
                if isinstance(current, tuple):
                    ordered_images[mid] = current + (new_image,)
                else:
                    ordered_images[mid] = (current, new_image)
                return

            case "quit":
                print("Exiting program")
                sys.exit(0)

    # Insert the new image in the correct position
    ordered_images.insert(low, new_image)

# Main function
def label_orderings():
    # Load already ordered images
    ordered_images = load_progress()

    # Get all images from the folder
    images = get_all_image_paths(IMAGES_FOLDER)

    for idx, new_image in enumerate(images):
        if any(new_image in group for group in ordered_images if isinstance(group, tuple)) or new_image in ordered_images:
            continue  # Skip already ordered images

        print(f"Images left to label: {len(images) - idx}")

        # Compare the new image with the ordered list
        insert_image(new_image, ordered_images)

        # Save progress after each insertion
        save_progress(ordered_images)

    print("Final Ordered List:")
    print(ordered_images)

if __name__ == "__main__":
    label_orderings()
