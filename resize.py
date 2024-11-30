import os
from PIL import Image
from tqdm import tqdm

# Define directories
# original_train_dir = "train2014/"
original_val_dir = "val2014/"
# resized_train_dir = "resized_train2014/"
resized_val_dir = "resized_val2014/"

# Create directories to store resized images
# os.makedirs(resized_train_dir, exist_ok=True)
os.makedirs(resized_val_dir, exist_ok=True)


def resize_images(input_dir, output_dir, scale=0.9):
    """Resize all images in input_dir by the given scale and save to output_dir."""
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".jpg"):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)

            # Calculate the new dimensions
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)

            # Resize the image and save it
            img_resized = img.resize((new_width, new_height), Image.ANTIALIAS)
            img_resized.save(os.path.join(output_dir, filename))


# Resize train and validation images
# resize_images(original_train_dir, resized_train_dir)
resize_images(original_val_dir, resized_val_dir)
