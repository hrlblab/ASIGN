import pandas as pd
import os

"""
Step 4: find the pair of region and global level patches based on their overlaps, and filter the global level patches.
Input: region information, global information
Output: filtered region information, filtered patches
Update: pair relation between 512 and 1024 patches
"""


def is_overlap(patch1, patch2, size1, size2):
    """Check whether two patches have overlapping areas."""
    # Range of patch1
    x1_min = patch1['x']
    x1_max = x1_min + size1
    y1_min = patch1['y']
    y1_max = y1_min + size1

    # Range of patch2
    x2_min = patch2['x']
    x2_max = x2_min + size2
    y2_min = patch2['y']
    y2_max = y2_min + size2

    # Check for overlap
    overlap_x = not (x1_max <= x2_min or x1_min >= x2_max)
    overlap_y = not (y1_max <= y2_min or y1_min >= y2_max)

    return overlap_x and overlap_y


def find_overlapping_patches(patches_512_csv, patches_1024_csv, size_512, size_1024):
    """Find overlaps between 512x512 and 1024x1024 patches."""
    # Read patch info CSV files for 512x512 and 1024x1024 patches
    patches_512_df = pd.read_csv(patches_512_csv)
    patches_1024_df = pd.read_csv(patches_1024_csv)

    # Initialize a dictionary to store 1024x1024 patches and their overlapping 512x512 patches
    overlap_dict = {}

    # Initialize a set to store all 1024x1024 patches that have overlaps
    patches_1024_with_overlap = set()

    # Iterate over each 512x512 patch
    for _, patch_512_row in patches_512_df.iterrows():
        patch_512_name = patch_512_row['patch_filename']

        # Iterate over each 1024x1024 patch
        for _, patch_1024_row in patches_1024_df.iterrows():
            patch_1024_name = patch_1024_row['patch_filename']

            # Check for overlap
            if is_overlap(patch_512_row, patch_1024_row, size_512, size_1024):
                # If overlap exists, record it
                if patch_1024_name in overlap_dict:
                    overlap_dict[patch_1024_name].append(patch_512_name)
                else:
                    overlap_dict[patch_1024_name] = [patch_512_name]

                # Mark this 1024 patch as having an overlap
                patches_1024_with_overlap.add(patch_1024_name)

    # Identify 1024x1024 patches with no overlaps
    all_patches_1024 = set(patches_1024_df['patch_filename'])
    patches_1024_without_overlap = all_patches_1024 - patches_1024_with_overlap

    return overlap_dict, list(patches_1024_without_overlap)


def remove_patches_without_overlap(patches_1024_without_overlap, patch_dir, patches_csv):
    """Delete 1024x1024 patches without overlaps and update the CSV file."""
    for patch_filename in patches_1024_without_overlap:
        patch_path = os.path.join(patch_dir, patch_filename)
        if os.path.exists(patch_path):
            os.remove(patch_path)
            print(f"Deleted: {patch_path}")

    # Remove records from the CSV for patches without 512x512 overlaps
    patches_df = pd.read_csv(patches_csv)
    patches_df_filtered = patches_df[~patches_df['patch_filename'].isin(patches_1024_without_overlap)]

    # Save the updated CSV file
    patches_df_filtered.to_csv(patches_csv, index=False)
    print(f"CSV file updated, removed 1024x1024 patches without overlaps: {patches_csv}")


def update_1024_csv_with_overlap(patches_1024_csv, overlap_dict):
    """Update the 1024x1024 patch CSV file to include overlapping 512x512 patch names."""
    patches_1024_df = pd.read_csv(patches_1024_csv)

    # Add a column to each 1024x1024 patch listing overlapping 512x512 patch names
    patches_1024_df['overlapping_512_patches'] = patches_1024_df['patch_filename'].apply(
        lambda patch_name: ', '.join(overlap_dict.get(patch_name, []))
    )

    # Save the updated CSV file
    patches_1024_df.to_csv(patches_1024_csv, index=False)
    print(f"CSV file updated with overlapping 512x512 patch names: {patches_1024_csv}")


def batch_process_patches(image_dir, patches_512_dir, patches_1024_dir, output_dir_1024, size_512, size_1024):
    """Batch process each image to find overlaps between 512x512 and 1024x1024 patches."""
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

    for image_path in image_paths:
        image_name = os.path.splitext(os.path.basename(image_path))[0]  # Get the image filename (without extension)
        print(f"Processing image: {image_name}")

        # Corresponding CSV file paths for 512x512 and 1024x1024 patches
        patches_512_csv = os.path.join(patches_512_dir, f"{image_name}_patches_512.csv")
        patches_1024_csv = os.path.join(patches_1024_dir, f"{image_name}_patches_1024.csv")

        # Folder path for 1024x1024 patch images
        patch_dir_1024 = os.path.join(output_dir_1024, image_name)

        # Find overlaps between 512x512 and 1024x1024 patches
        overlap_dict, patches_1024_without_overlap = find_overlapping_patches(patches_512_csv, patches_1024_csv,
                                                                              size_512, size_1024)

        # Delete 1024x1024 patches with no 512x512 overlap and update the CSV
        remove_patches_without_overlap(patches_1024_without_overlap, patch_dir_1024, patches_1024_csv)

        # Update CSV file: add overlapping 512x512 patch names to each 1024x1024 patch
        update_1024_csv_with_overlap(patches_1024_csv, overlap_dict)


if __name__ == "__main__":
    # Input paths for images and patch info folders
    root_images = "./ST_Breast/imgs"
    root_patches_512_dir = "./ST_Breast/patches_csv/patches_512"
    root_patches_1024_dir = "./ST_Breast/patches_csv/patches_1024"
    root_output_dir_1024 = "./ST_Breast/cropped_imgs/patches_1024"

    # Patch sizes
    size_512 = 512
    size_1024 = 1024

    for sample in os.listdir(root_images):
        images = os.path.join(root_images, sample)
        patches_512_dir = os.path.join(root_patches_512_dir, sample)
        patches_1024_dir = os.path.join(root_patches_1024_dir, sample)
        output_dir_1024 = os.path.join(root_output_dir_1024, sample)

        # Batch process each image to find overlaps and update CSV files and patch images
        batch_process_patches(images, patches_512_dir, patches_1024_dir, output_dir_1024, size_512, size_1024)
