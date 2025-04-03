import pandas as pd
import os

"""
Step 3: find the pair of spot and region level patches, and filter the region level patches.
Input: spot information, region information
Output: filtered region information, filtered patches
Update: pair relation between 224 and 512 patches
"""


def find_patches_containing_points(points_csv, patches_csv, patch_size):
    # Read the CSV file containing point information
    points_df = pd.read_csv(points_csv)
    column_names = ['barcode', 'pixel_x', 'pixel_y',  'i', 'j']
    points_df.columns = column_names

    # Read the CSV file containing patch information
    patches_df = pd.read_csv(patches_csv)

    # Initialize a dictionary to store patches and the points they contain
    patch_points_dict = {patch_row['patch_filename']: [] for _, patch_row in patches_df.iterrows()}

    # Iterate over each point to find the corresponding patch
    for _, point_row in points_df.iterrows():
        point_name = point_row['barcode']
        point_x = point_row['pixel_x']
        point_y = point_row['pixel_y']

        # Find which patch contains the point
        for _, patch_row in patches_df.iterrows():
            patch_name = patch_row['patch_filename']
            patch_x = patch_row['x']
            patch_y = patch_row['y']

            # Check if the point is within the patch
            if (patch_x <= point_x < patch_x + patch_size) and (patch_y <= point_y < patch_y + patch_size):
                # Add the point name to the corresponding patch
                patch_points_dict[patch_name].append(point_name)
                break

    return patch_points_dict


def update_patches_csv_with_points(patches_csv, patch_points_dict):
    # Read the CSV file of 512x512 patches
    patches_df = pd.read_csv(patches_csv)

    # Add a new column to each patch recording the names of the points it contains
    patches_df['points'] = patches_df['patch_filename'].apply(
        lambda patch_name: ', '.join(patch_points_dict[patch_name]))

    # Save the updated CSV file
    patches_df.to_csv(patches_csv, index=False)
    print(f"CSV file updated: {patches_csv}")


def remove_patches_without_points(patches_csv, patch_points_dict, patch_dir):
    # Read the CSV file of 512x512 patches
    patches_df = pd.read_csv(patches_csv)

    # Find patches that do not contain any points
    patches_without_points = [patch for patch, points in patch_points_dict.items() if not points]
    print(patches_without_points)

    # Delete patch images from folder that do not contain any points
    for patch_filename in patches_without_points:
        patch_path = os.path.join(patch_dir, patch_filename)

        if os.path.exists(patch_path):
            os.remove(patch_path)
            print(f"Deleted: {patch_path}")

    # Remove records of patches without points from the CSV
    patches_df_filtered = patches_df[~patches_df['patch_filename'].isin(patches_without_points)]

    # Save the updated CSV file
    patches_df_filtered.to_csv(patches_csv, index=False)
    print(f"CSV file updated, patches without points removed: {patches_csv}")


def batch_process_images_with_points(image_dir, points_dir, patches_dir, patch_size, patch_output_dir):
    # Get paths of all PNG images
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

    for image_path in image_paths:
        image_name = os.path.splitext(os.path.basename(image_path))[0]  # Get image file name (without extension)
        print(f"Processing image: {image_name}")

        # Corresponding CSV file path for points
        points_csv = os.path.join(points_dir, f"{image_name}.csv")

        # Corresponding CSV file path for patches
        patches_csv = os.path.join(patches_dir, f"{image_name}_patches_512.csv")

        # Corresponding patch image folder
        patch_dir = os.path.join(patch_output_dir, image_name)

        # Find patches containing points and their associated points
        patch_points_dict = find_patches_containing_points(points_csv, patches_csv, patch_size)

        # Update CSV file by adding a column for contained point names
        update_patches_csv_with_points(patches_csv, patch_points_dict)

        # Delete patches that do not contain any points and their CSV records
        remove_patches_without_points(patches_csv, patch_points_dict, patch_dir)


if __name__ == "__main__":
    root_images = "./ST_Breast/imgs"

    root_points_dir = "./ST_Breast/location"
    root_patches_dir = "./ST_Breast/patches_csv/patches_512"
    root_patch_output_dir = "./ST_Breast/cropped_imgs/patches_512"

    # Patch size is 512x512
    patch_size = 512

    for sample in os.listdir(root_images):
        images = os.path.join(root_images, sample)
        points_dir = os.path.join(root_points_dir, sample)
        patches_dir = os.path.join(root_patches_dir, sample)
        patch_output_dir = os.path.join(root_patch_output_dir, sample)

        # Batch process PNG images: process corresponding points and update patch information
        batch_process_images_with_points(images, points_dir, patches_dir, patch_size, patch_output_dir)
