import os
import shutil
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity  # This may be kept for reference


# Read a CSV file to return the names of polygons and circles
def load_polygon_data(file_path):
    df = pd.read_csv(file_path)
    names = df['barcode'].tolist()
    return names


# Main function: compute cosine similarity between all circles from two files and return a DataFrame
def calculate_cos_with_gpu(file1, file2, npy_feature1, npy_feature2):
    names1 = load_polygon_data(file1)
    names2 = load_polygon_data(file2)

    iou_matrix = pd.DataFrame(index=names1, columns=names2)  # Create an empty DataFrame
    features1 = []
    features2 = []

    for i in names1:
        tmp = npy_feature1[i[:6]].item()  # Extract relevant features from the loaded npy data
        value = tmp.get(i[7:])
        if value is not None:
            features1.append(value)
        else:
            print(f"Feature not found for {i[7:]}")  # Log if the feature is not found

    for i in names2:
        tmp = npy_feature2[i[:6]].item()
        value = tmp.get(i[7:])
        if value is not None:
            features2.append(value)
        else:
            print(f"Feature not found for {i[7:]}")

    # Convert features to PyTorch tensors and move them to the GPU
    features1_tensor = torch.tensor(features1, dtype=torch.float32).cuda()
    features2_tensor = torch.tensor(features2, dtype=torch.float32).cuda()

    # Normalize features for cosine similarity computation
    features1_tensor = torch.nn.functional.normalize(features1_tensor, p=2, dim=1)
    features2_tensor = torch.nn.functional.normalize(features2_tensor, p=2, dim=1)

    for i, poly1 in tqdm(enumerate(features1_tensor)):
        # Compute cosine similarity in a batch-wise manner
        similarities = torch.matmul(poly1.unsqueeze(0), features2_tensor.T).cpu().numpy()
        iou_matrix.iloc[i, :] = similarities.flatten()

    return iou_matrix


def save_iou_matrix_to_csv(file1, file2, npy_feature1, npy_feature2, output_file):
    iou_matrix = calculate_cos_with_gpu(file1, file2, npy_feature1, npy_feature2)
    iou_matrix.to_csv(output_file)


csv_dir = './Human_dorsolateral/spot_regist'
output_dir = './Human_dorsolateral/similarity'

for csvs in os.listdir(csv_dir):
    csv_name = csvs[:-4]
    tmp_csv = os.path.join(csv_dir, csvs)

    npy_dir = f'./Human_dorsolateral/spot_feature/{csv_name}'
    tmp_infor = {}
    for npys in os.listdir(npy_dir):
        tmp_infor[npys[:-4]] = np.load(os.path.join(npy_dir, npys), allow_pickle=True)

    output_file = os.path.join(output_dir, csvs)

    save_iou_matrix_to_csv(tmp_csv, tmp_csv, tmp_infor, tmp_infor, output_file)
