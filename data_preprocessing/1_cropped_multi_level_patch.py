from PIL import Image
import os
import pandas as pd

"""
Step 2: patch cropping for region/global level patches.
With the method of traverse, crop images into different resolutions, including region-512*512 and global-1024*1024
Input: Dir of WSIs, Save dir for patches, Patch size=512/1024, Save dir for patch information(csv file)
Output: 1.Cropped Images
        2.Patch information for 512/1024, saved as csv file
        Information including: patch_name, WSI_name, true_position, position_index, patch_size

Note that: for spot-level 224*224 patches, each patch is centered with the coordinate in ST data. 
"""


def generate_patches(image_path, output_dir, patch_size, output_csv_dir):
    # Open the PNG image
    img = Image.open(image_path)
    width, height = img.size  # Get the width and height of the image

    # Calculate the patch step size to ensure the last patch aligns exactly with the edge
    step_x = width // patch_size[0]
    step_y = height // patch_size[1]

    patches = []
    image_name = os.path.splitext(os.path.basename(image_path))[0]  # Get the image filename without extension

    save_dir = os.path.join(output_dir, image_name)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print(f" {save_dir}")

    # Traverse the image and crop patches based on the step size
    for i in range(step_x):
        for j in range(step_y):
            # Calculate the top-left coordinates of the patch
            x = i * patch_size[0]
            y = j * patch_size[1]

            # Ensure the patch does not exceed the bottom-right edge of the image
            if x + patch_size[0] > width:
                x = width - patch_size[0]
            if y + patch_size[1] > height:
                y = height - patch_size[1]

            # Crop the patch
            patch = img.crop((x, y, x + patch_size[0], y + patch_size[1]))

            # Save the patch to file
            patch_filename = f"{image_name}_size_{patch_size[0]}_{x}_{y}.png"
            patch.save(os.path.join(save_dir, patch_filename))

            # Record patch information including i and j
            patches.append({
                'patch_filename': patch_filename,
                'image_name': image_name,
                'x': x,
                'y': y,
                'i': i,  # i value in the grid
                'j': j,  # j value in the grid
                'width': patch_size[0],
                'height': patch_size[1]
            })

    # Generate the CSV file path for the image
    csv_file = os.path.join(output_csv_dir, f"{image_name}_patches_{patch_size[0]}.csv")

    # Save all patch information to a separate CSV file
    df = pd.DataFrame(patches)
    df.to_csv(csv_file, index=False)
    print(f"CSV file saved: {csv_file}")


def batch_process_images(image_dir, output_dir_1024, output_dir_512, csv_dir_1024, csv_dir_512):
    # Get paths of all PNG images
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        # Generate patches of size 1024x1024
        generate_patches(image_path, output_dir_1024, patch_size=(1024, 1024), output_csv_dir=csv_dir_1024)

        # Generate patches of size 512x512
        generate_patches(image_path, output_dir_512, patch_size=(512, 512), output_csv_dir=csv_dir_512)


if __name__ == "__main__":
    # Input folder path where WSI images are located
    images = "./ST_Breast/imgs"

    # Output folder paths for patches
    output_dir_1024_root = "./ST_Breast/cropped_imgs/patches_1024"
    output_dir_512_root = "./ST_Breast/cropped_imgs/patches_512"

    # Output folder paths for CSV files (to store patch info for each image)
    csv_dir_1024_root = "./ST_Breast/patches_csv/patches_1024"
    csv_dir_512_root = "./ST_Breast/patches_csv/patches_512"

    for sample in os.listdir(images):
        img = os.path.join(images, sample)
        output_dir_1024 = os.path.join(output_dir_1024_root, sample)
        output_dir_512 = os.path.join(output_dir_512_root, sample)

        csv_dir_1024 = os.path.join(csv_dir_1024_root, sample)
        csv_dir_512 = os.path.join(csv_dir_512_root, sample)

        # Create output folders
        os.makedirs(output_dir_1024, exist_ok=True)
        os.makedirs(output_dir_512, exist_ok=True)

        # Create CSV folders
        os.makedirs(csv_dir_1024, exist_ok=True)
        os.makedirs(csv_dir_512, exist_ok=True)

        # Batch process PNG images: generate patches of 1024x1024 and 512x512, and save to separate CSV files
        batch_process_images(img, output_dir_1024, output_dir_512, csv_dir_1024, csv_dir_512)
