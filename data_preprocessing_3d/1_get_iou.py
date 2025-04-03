from tqdm import tqdm
import glob
import os
import pandas as pd
from shapely.geometry import Polygon
from tqdm import tqdm
from joblib import Parallel, delayed

"""
Step 1: calculate IoU of spots cross layers.
During image registration process, you should get registration_3D_ANTs. Then we use it to calculate the position of 
spots regarding the reference WSI slide.
Output: IoU of spots across layers, saved as csv file.
"""


# Load CSV files and return polygons and circle names
def load_polygon_data(file_path):
    df = pd.read_csv(file_path)
    polygons = []
    names = df['barcode'].tolist()

    for index, row in df.iterrows():
        points = [(row[f'now_x_{i}'], row[f'now_y_{i}']) for i in range(0, 100)]  # Pairs of x and y for each point
        polygons.append(Polygon(points))

    return names, polygons


# Compute the IoU (Intersection over Union) of two polygons
def iou(poly1, poly2):
    inter_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    return inter_area / union_area if union_area > 0 else 0


# Main function: calculate the IoU between all circles from two files and return a DataFrame
def calculate_ious(file1, file2):
    names1, polygons1 = load_polygon_data(file1)
    names2, polygons2 = load_polygon_data(file2)

    iou_matrix = pd.DataFrame(index=names1, columns=names2)  # Create an empty DataFrame

    # Function to compute IoU for one pair of polygons
    def compute_iou_row(poly1, index):
        row = []
        for poly2 in polygons2:
            row.append(iou(poly1, poly2))
        return index, row

    # Parallelize the computation across multiple cores
    results = Parallel(n_jobs=-1)(delayed(compute_iou_row)(poly1, i) for i, poly1 in tqdm(enumerate(polygons1)))

    # Populate the DataFrame with results
    for index, row in results:
        iou_matrix.iloc[index, :] = row

    return iou_matrix


# Save the IoU matrix as a CSV file
def save_iou_matrix_to_csv(file1, file2, output_file):
    iou_matrix = calculate_ious(file1, file2)
    iou_matrix.to_csv(output_file)


def set_region_zero(df, idx, n):
    """
    Set all elements within an n*n region in the DataFrame starting at idx to zero.

    Args:
        df (pd.DataFrame): Input DataFrame.
        idx (tuple): Starting index of the region, in the format (row_index, col_index).
        n (int): Size of the region to be set to zero.

    Returns:
        pd.DataFrame: Modified DataFrame with the specified region set to zero.
    """
    row_start, col_start = idx
    df.iloc[row_start:row_start + n, col_start:col_start + n] = 0
    return df


csv_dir = './Human_dorsolateral/WSI_png_test_contour'
for tmp_dir in os.listdir(csv_dir):
    folder_path = f'./Human_dorsolateral/WSI_png_test_contour/{tmp_dir}/registration_3D_ANTs'
    file_pattern = folder_path + '/*.csv'  # Match all CSV files

    # Use glob to get all CSV file paths
    csv_files = glob.glob(file_pattern)

    # Store the processing results for each file
    all_data = []

    # Iterate over each CSV file
    for file_path in csv_files:
        # Get the file name (without path and extension)
        print(os.path.basename(file_path))
        file_name = os.path.basename(file_path).split('_')[0]

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Modify elements of the first column, assuming the column name is df.columns[0]
        df[df.columns[0]] = file_name + "_" + df[df.columns[0]].astype(str)

        # Add the processed DataFrame to the list
        all_data.append(df)

    # Concatenate all data
    combined_data = pd.concat(all_data, ignore_index=True)

    # Save the concatenated DataFrame to a CSV file
    output_file = f'./Human_dorsolateral/spot_regist/{tmp_dir}.csv'
    combined_data.to_csv(output_file, index=False)

# Save concatenated result and calculate IoU matrix
input_dir = './Human_dorsolateral/spot_regist'
output_dir = './Human_dorsolateral/spot_iou'

for file in tqdm(os.listdir(input_dir)):
    file1 = os.path.join(input_dir, file)
    output_file = os.path.join(output_dir, file)
    save_iou_matrix_to_csv(file1, file1, output_file)
