import os
import csv
import random
import cv2

# Keyboard button to image-category
KEY_TO_CATEGORY = {
    "1": "0",
    "2": "1",
    "3": "2",
    "4": "3",
    "u": "unknown",
    "s": "skip",
    "q": "quit",  # 'q' will quit the program
}

# Paths
IMAGES_FOLDER = "images"  # Update with the folder containing your images
OUTPUT_CSV = "image_labels/labeled_data.csv"

def load_progress():
    """Load progress from the CSV file."""
    labeled_images = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                if row:
                    labeled_images.add(row[0])
    return labeled_images

def get_all_image_paths(folder):
    """Get all image paths from the folder and subfolders."""
    image_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)

def draw_text_on_image(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, color=(255, 255, 255), thickness=1):
    """Draw text on an image."""
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

def label_images():
    """Label images manually."""
    labeled_images = load_progress()
    all_images = get_all_image_paths(IMAGES_FOLDER)

    # Shuffle images
    random.shuffle(all_images)

    with open(OUTPUT_CSV, "a", newline="") as file:
        writer = csv.writer(file)

        for image_path in all_images:
            # Check if the image is already labeled
            relative_image_path = os.path.relpath(image_path, IMAGES_FOLDER)
            if relative_image_path in labeled_images:
                continue  # Skip already labeled images

            # Load and prepare the image
            img = cv2.imread(image_path)
            display_img = img.copy()

            # Display key-to-category mappings on the image
            y_offset = 20
            draw_text_on_image(display_img, "Key Mappings:", (10, y_offset), font_scale=0.6, color=(0, 255, 0))
            for key, category in KEY_TO_CATEGORY.items():
                y_offset += 20
                draw_text_on_image(display_img, f"{key}: {category}", (10, y_offset), font_scale=0.5)

            while True:
                # Show the image with instructions
                cv2.imshow("Labeling", display_img)
                print(f"Image: {relative_image_path}")

                # Wait for a key press
                key = cv2.waitKey(0)

                # Convert key press to string
                key_char = chr(key & 0xFF)

                command = KEY_TO_CATEGORY.get(key_char, "-")

                match command:
                    case "skip":
                        print(f"Skipping {relative_image_path}.")
                        break

                    case "quit":
                        print("Exiting and saving progress...")
                        cv2.destroyAllWindows()
                        return

                    case command if command in KEY_TO_CATEGORY.values():
                        writer.writerow([relative_image_path, command])
                        labeled_images.add(relative_image_path)
                        print(f"Labeled {relative_image_path} as {command}.")
                        file.flush()  # Flush the file buffer to ensure the data is written immediately
                        break

                    case _:
                        print(f"Invalid key '{key_char}'. Please press a valid key.")

            # Close the current image window before moving on
            cv2.destroyWindow("Labeling")

if __name__ == "__main__":
    label_images()