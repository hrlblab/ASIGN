import pandas as pd
import os


def is_overlap(patch1, patch2, size1, size2):
    """Determine whether two Patches have overlapping areas."""
    # Range of Patch1
    x1_min = patch1['x']
    x1_max = x1_min + size1
    y1_min = patch1['y']
    y1_max = y1_min + size1

    # Range of Patch2
    x2_min = patch2['x']
    x2_max = x2_min + size2
    y2_min = patch2['y']
    y2_max = y2_min + size2

    # Check for overlap
    overlap_x = not (x1_max <= x2_min or x1_min >= x2_max)
    overlap_y = not (y1_max <= y2_min or y1_min >= y2_max)

    return overlap_x and overlap_y


def find_overlapping_patches(patches_512_csv, patches_1024_csv, size_512, size_1024):
    """Find overlap between 512x512 and 1024x1024 Patches."""
    # Read CSV files for 512x512 and 1024x1024 Patch information
    patches_512_df = pd.read_csv(patches_512_csv)
    patches_1024_df = pd.read_csv(patches_1024_csv)

    # Initialize a dictionary to store 1024x1024 Patches and their overlapping 512x512 Patches
    overlap_dict = {}

    # Initialize a set to store all 1024x1024 Patches that have overlap
    patches_1024_with_overlap = set()

    # Iterate over each 512x512 Patch
    for _, patch_512_row in patches_512_df.iterrows():
        patch_512_name = patch_512_row['patch_filename']

        # Iterate over each 1024x1024 Patch
        for _, patch_1024_row in patches_1024_df.iterrows():
            patch_1024_name = patch_1024_row['patch_filename']

            # Check for overlap
            if is_overlap(patch_512_row, patch_1024_row, size_512, size_1024):
                # If there is overlap, record the 512x512 Patch under the corresponding 1024x1024 Patch
                if patch_1024_name in overlap_dict:
                    overlap_dict[patch_1024_name].append(patch_512_name)
                else:
                    overlap_dict[patch_1024_name] = [patch_512_name]

                # Record that the 1024x1024 Patch has overlap
                patches_1024_with_overlap.add(patch_1024_name)

    # Identify 1024x1024 Patches without overlap
    all_patches_1024 = set(patches_1024_df['patch_filename'])
    patches_1024_without_overlap = all_patches_1024 - patches_1024_with_overlap

    return overlap_dict, list(patches_1024_without_overlap)


def remove_patches_without_overlap(patches_1024_without_overlap, patch_dir, patches_csv):
    """Delete 1024x1024 Patch images without overlap and update the CSV file."""
    for patch_filename in patches_1024_without_overlap:
        patch_path = os.path.join(patch_dir, patch_filename)
        if os.path.exists(patch_path):
            os.remove(patch_path)
            print(f"Deleted: {patch_path}")

    # Remove 1024x1024 Patches without overlapping 512x512 Patches from the CSV
    patches_df = pd.read_csv(patches_csv)
    patches_df_filtered = patches_df[~patches_df['patch_filename'].isin(patches_1024_without_overlap)]

    # Save the updated CSV file
    patches_df_filtered.to_csv(patches_csv, index=False)
    print(f"Updated CSV file with removed 1024x1024 Patches without overlap: {patches_csv}")


def update_1024_csv_with_overlap(patches_1024_csv, overlap_dict):
    """Update the CSV file for 1024x1024 Patches to add names of overlapping 512x512 Patches."""
    patches_1024_df = pd.read_csv(patches_1024_csv)

    # Add a new column for each 1024x1024 Patch to record the names of overlapping 512x512 Patches
    patches_1024_df['overlapping_512_patches'] = patches_1024_df['patch_filename'].apply(
        lambda patch_name: ', '.join(overlap_dict.get(patch_name, []))
    )

    # Save the updated CSV file
    patches_1024_df.to_csv(patches_1024_csv, index=False)
    print(f"Updated CSV file with overlapping 512x512 Patch names: {patches_1024_csv}")


def batch_process_patches(image_dir, patches_512_dir, patches_1024_dir, output_dir_1024, size_512, size_1024):
    """Batch process each image to find overlap between 512x512 and 1024x1024 Patches."""
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

    for image_path in image_paths:
        image_name = os.path.splitext(os.path.basename(image_path))[0]  # Get the image file name (without extension)
        print(f"Processing image: {image_name}")

        # Corresponding CSV file paths for 512x512 and 1024x1024 Patches
        patches_512_csv = os.path.join(patches_512_dir, f"{image_name}_patches_512.csv")
        patches_1024_csv = os.path.join(patches_1024_dir, f"{image_name}_patches_1024.csv")

        # Corresponding folder for 1024x1024 Patch images
        patch_dir_1024 = os.path.join(output_dir_1024, image_name)

        # Find overlap between 512x512 and 1024x1024 Patches
        overlap_dict, patches_1024_without_overlap = find_overlapping_patches(patches_512_csv, patches_1024_csv,
                                                                              size_512, size_1024)

        # Remove 1024x1024 Patches without any 512x512 Patch overlap and update the CSV file
        remove_patches_without_overlap(patches_1024_without_overlap, patch_dir_1024, patches_1024_csv)

        # Update the CSV file to add the names of overlapping 512x512 Patches for each 1024x1024 Patch
        update_1024_csv_with_overlap(patches_1024_csv, overlap_dict)


if __name__ == "__main__":
    # Input paths for image and Patch information folders
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

        # Batch process each image to find overlap between 512x512 and 1024x1024 Patches, and update CSV files and images
        batch_process_patches(images, patches_512_dir, patches_1024_dir, output_dir_1024, size_512, size_1024)
