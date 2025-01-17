import os
import csv
import random
import cv2

# Keyboard button to image-category
KEY_TO_CATEGORY = {
    "z": "0",
    "x": "1",
    "c": "2",
    "v": "3",
    "-": "unknown",
    "s": "skip",
    "q": "quit",  # 'q' will quit the program
}

# Paths
IMAGES_FOLDER = "images"  # Update with the folder containing your images
OUTPUT_CSV = "image_labels/labeled_data.csv"

def resize_image(img, max_width, max_height):
    height, width = img.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    return cv2.resize(img, (new_width, new_height))

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

    # Only keep images not yet labeled
    remaining_images = list(set(all_images) - set(labeled_images))

    # Shuffle images
    random.shuffle(remaining_images)

    with open(OUTPUT_CSV, "a", newline="") as file:
        writer = csv.writer(file)

        for idx, image_path in enumerate(remaining_images):

            print(f"Number of images to label:{len(remaining_images) - idx}")

            # Load and prepare the image
            img = cv2.imread(image_path)
            display_img = img.copy()
            display_img = resize_image(display_img, 1400, 1400)

            # Display key-to-category mappings on the image
            y_offset = 20
            draw_text_on_image(display_img, "Key Mappings:", (10, y_offset), font_scale=0.6, color=(0, 255, 0))
            for key, category in KEY_TO_CATEGORY.items():
                y_offset += 20
                draw_text_on_image(display_img, f"{key}: {category}", (10, y_offset), font_scale=0.5, color=(0, 255, 0))

            while True:
                # Show the image with instructions
                cv2.imshow("Labeling", display_img)
                print(f"Image: {image_path}")

                # Wait for a key press
                key = cv2.waitKey(0)

                # Convert key press to string
                key_char = chr(key & 0xFF)

                command = KEY_TO_CATEGORY.get(key_char, "-")

                match command:
                    case "skip":
                        print(f"Skipping {image_path}.")
                        break

                    case "quit":
                        print("Exiting and saving progress...")
                        cv2.destroyAllWindows()
                        return

                    case command if command in KEY_TO_CATEGORY.values():
                        writer.writerow([image_path, command])
                        labeled_images.add(image_path)
                        print(f"Labeled {image_path} as {command}.")
                        file.flush()  # Flush the file buffer to ensure the data is written immediately
                        break

                    case _:
                        print(f"Invalid key '{key_char}'. Please press a valid key.")

            # Close the current image window before moving on
            cv2.destroyWindow("Labeling")

if __name__ == "__main__":
    label_images()