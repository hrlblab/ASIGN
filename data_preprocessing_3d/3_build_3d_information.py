import os

import pandas as pd
from collections import defaultdict


def set_region_zero(df, idx, n):
    """
    Set all elements within an n*n region in the DataFrame starting at idx to zero.

    Args:
        df (pd.DataFrame): Input DataFrame.
        idx (tuple): Starting index of the region, in the format (row_index, col_index).
        n (int): Size of the region to be set to zero.

    Returns:
        pd.DataFrame: The modified DataFrame with the specified region set to zero.
    """
    row_start, col_start = idx
    df.iloc[row_start:row_start + n, col_start:col_start + n] = 0
    return df


def process_and_save_csv(file1_path, file2_path, output_path):
    """
    Load two CSV files, add them together, count columns based on their name prefixes,
    set specific regions to zero, and save the result to a new CSV file.

    Args:
        file1_path (str): Path to the first CSV file.
        file2_path (str): Path to the second CSV file.
        output_path (str): Path to the output CSV file.
    """
    # Load the two CSV files
    df1 = pd.read_csv(file1_path, index_col=0)
    df2 = pd.read_csv(file2_path, index_col=0)

    # Add the two DataFrames together
    df = df1 + df2

    print(df)

    # Initialize a dictionary to count columns with the same first two letters
    prefix_count = defaultdict(int)

    # Iterate over column names to extract and count the first two letters
    for col in df.columns:
        prefix = col[:2]  # Extract the first two letters
        prefix_count[prefix] += 1

    # Convert the count result to a list format
    result = list(prefix_count.values())
    print("Prefix count:", prefix_count)
    print("Result list:", result)

    # Initialize position index
    tmp_position_idx = 0
    for idxs in result[1:]:
        df = set_region_zero(df, (tmp_position_idx, tmp_position_idx), idxs)
        print("Setting region zero for index:", idxs)
        print("Updated DataFrame:\n", df)
        tmp_position_idx += idxs

    # Save the result to a new CSV file
    df.to_csv(output_path, index=True)
    print(f"Processed file saved to: {output_path}")


similarity_file_dir = './Human_dorsolateral/similarity'
iou_file_dir = './Human_dorsolateral/spot_iou'
output_dir = './Human_dorsolateral_3D/3D_combined_information'

for file in os.listdir(similarity_file_dir):
    simi_file = os.path.join(similarity_file_dir, file)
    iou_file = os.path.join(iou_file_dir, file)
    output_paths = os.path.join(output_dir, file)
    process_and_save_csv(simi_file, iou_file, output_paths)
