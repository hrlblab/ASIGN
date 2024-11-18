import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from models.graph_construction import calcADJ
import os


def find_closest_column(target_sequence, tensor):
    """
    Find the column in `tensor` that is closest to `target_sequence` and return the content and distance.

    Parameters:
    - target_sequence (torch.Tensor): Target sequence with shape (250,)
    - tensor (torch.Tensor): Tensor to compare against with shape (n, 250)

    Returns:
    - closest_column (torch.Tensor): The column closest to the target sequence
    - closest_distance (float): The distance to the closest column
    - closest_index (int): The index of the closest column
    """
    # Calculate Euclidean distance between each column and the target sequence
    target_sequence = torch.tensor(target_sequence)
    distances = torch.norm(tensor - target_sequence, dim=1)

    # Find the index with the smallest distance and corresponding distance
    closest_index = torch.argmin(distances)

    return closest_index.item()


class ImageGraphDataset(Dataset):
    def __init__(self, data_infor_path, graph_path, transform=None):
        # Load data_list and graph information from .npy file
        self.data_list = np.load(data_infor_path, allow_pickle=True).tolist()
        self.graph = torch.load(graph_path)
        self.transform = transform
        self.name = data_infor_path.split('/')[-2]

        # Load graph layer labels
        self.layer_512 = self.graph['layer_512'].y
        self.layer_1024 = self.graph['layer_1024'].y

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_dir = './Our_data_format/HER2'
        image_path = os.path.join(img_dir, self.data_list[idx]['img_path'])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Retrieve additional data
        label = self.data_list[idx]['label']
        position = self.data_list[idx]['position']
        feature_1024 = self.data_list[idx]['feature_1024']
        feature_512 = self.data_list[idx]['feature_512']

        idx_1024 = find_closest_column(self.data_list[idx]['label_1024'], self.layer_1024)
        idx_512 = find_closest_column(self.data_list[idx]['label_512'], self.layer_512)

        return image, label.astype(np.float32), position, int(idx_1024), int(idx_512), feature_1024.astype(
            np.float32), feature_512.astype(np.float32),


def collate_fn(batch, graph):
    # Extract images, labels, positions, and other features
    images, labels, positions, idx_1024_list, idx_512_list, features_1024, features_512 = zip(*batch)

    images = torch.stack(images)
    labels = torch.tensor(labels)
    positions = torch.tensor(positions)
    n_adj = min(positions.shape[0]-1, 8)

    adj_sub = calcADJ(positions.numpy(), k=n_adj)

    features_1024 = torch.tensor(features_1024)
    features_512 = torch.tensor(features_512)
    idx_1024 = torch.tensor(idx_1024_list)
    idx_512 = torch.tensor(idx_512_list)

    return images, labels, adj_sub, positions, idx_1024, idx_512, features_1024, features_512, graph


# Function: Create a separate DataLoader for each npy file
def create_dataloaders_for_each_file(npy_file_paths, batch_size=32, transform=None):
    dataloaders = {}
    for npy_file in npy_file_paths:
        name = npy_file.split('/')[-1][:-4]

        graph_path = f'./HER2/graph_data/{name}.pt'

        dataset = ImageGraphDataset(npy_file, graph_path, transform=transform)
        graph_data = dataset.graph

        def collate_fn_with_params(batch, graph_datas=graph_data):
            return collate_fn(batch, graph_datas)

        dataloaders[npy_file] = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_with_params)
    return dataloaders
