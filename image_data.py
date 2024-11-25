import os
import numpy as np
from PIL import Image

# Define the directory containing the images
IMAGE_DIR = r"C:\Users\Jyoti\OneDrive\Desktop\Coding\SciRe 2024-25 AIM-TO-C-STAHZ\images"
OUTPUT_FILE = "image_data.npy"

def preprocess_images(image_dir, target_size=(128, 128)):
    images = []
    for file_name in os.listdir(image_dir):
        file_path = os.path.join(image_dir, file_name)
        try:
            # Open image and resize
            img = Image.open(file_path).resize(target_size)
            # Convert to numpy array
            img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img_array)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Convert list of images to NumPy array
    images_np = np.array(images)
    return images_np

# Preprocess and save the images
image_data = preprocess_images(IMAGE_DIR)
np.save(OUTPUT_FILE, image_data)
print(f"Saved preprocessed images to {OUTPUT_FILE}")
