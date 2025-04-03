import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from models.graph_construction import calcADJ, extract_submatrix_3D, calcADJ_from_distance_matrix_3D, \
    build_adj_matrix_with_similarity_and_known_labels, inter_layer_from_distance_matrix
import os
import pandas as pd
import copy


def retain_random_rows(labels, retain_ratio=0.5):
    row_mask = torch.rand(labels.size(0)) < retain_ratio
    retained_labels = labels * row_mask.unsqueeze(1).to(labels.dtype)

    return retained_labels


def find_closest_column(target_sequence, tensor):
    target_sequence = torch.tensor(target_sequence)
    distances = torch.norm(tensor - target_sequence, dim=1)
    closest_index = torch.argmin(distances)

    return closest_index.item()


class ImageGraphDataset(Dataset):
    def __init__(self, data_infor_path, graph_path, adj_matrix, transform=None):
        self.data_list = np.load(data_infor_path, allow_pickle=True).tolist()
        self.graph = torch.load(graph_path)
        self.transform = transform
        self.name = data_infor_path.split('/')[-1][:-4]

        self.layer_512 = self.graph['layer_512'].y
        self.layer_1024 = self.graph['layer_1024'].y
        self.adj_matrix = pd.read_csv(adj_matrix)

        # Initialize train_data and test_data
        self.train_data, self.test_data = self._split_data_by_level()

    def _split_data_by_level(self):
        train_data = [item for item in self.data_list if item.get('level', 1) == 1]
        test_data = [item for item in self.data_list if item.get('level', 1) >= 1]
        return train_data, test_data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Here provides code to load HER2 and ST data, please modify when you use it.
        """

        # img_dir = '/home/zhuj28/Desktop/CVPR/public_datasets/Our_data_format/HER2'
        # image_path = os.path.join(img_dir, self.data_list[idx]['img_path'])

        img_dir = '/home/zhuj28/Desktop/CVPR/public_datasets/Our_data_format/ST_Breast/cropped_imgs/patch_224'
        image_path = os.path.join(img_dir, self.name, self.data_list[idx]['img_path'].split('/')[-2] + '/' +
                                  self.data_list[idx]['img_path'].split('/')[-1])

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = self.data_list[idx]['label']
        position = self.data_list[idx]['position']
        feature_1024 = self.data_list[idx]['feature_1024']
        feature_512 = self.data_list[idx]['feature_512']

        idx_1024 = find_closest_column(self.data_list[idx]['label_1024'], self.layer_1024)
        idx_512 = find_closest_column(self.data_list[idx]['label_512'], self.layer_512)

        img_name = self.data_list[idx]['img_path'].split('/')

        """ """
        # column_name = f'{img_name[2][:2]}_{img_name[3][:-4]}'

        column_name = f'{img_name[2][:2]}__{img_name[3][:-4]}'

        idx_matrix = self.adj_matrix.columns.get_loc(column_name)
        level = self.data_list[idx]['level']

        return image, label.astype(np.float32), position, int(idx_1024), int(idx_512), feature_1024.astype(
            np.float32), feature_512.astype(np.float32), int(idx_matrix), int(level)

    def get_train_dataset(self):
        return Subset(self, range(len(self.train_data)))


def collate_fn(batch, adj_matrix, graph, is_weighted=False):
    images, labels, positions, idx_1024_list, idx_512_list, features_1024, features_512, idx_matrix, level = zip(*batch)

    images = torch.stack(images)
    labels = torch.tensor(labels)
    positions = torch.tensor(positions)

    k = min(np.array(idx_matrix).shape[0], 12)
    sub_matrix = extract_submatrix_3D(adj_matrix, np.array(idx_matrix))
    sub_inter_matrix = inter_layer_from_distance_matrix(sub_matrix, positions, k=4)

    adj_sub = calcADJ_from_distance_matrix_3D(sub_matrix, k=k) + sub_inter_matrix
    adj_sub[adj_sub > 0] = 1

    features_1024 = torch.tensor(features_1024)
    features_512 = torch.tensor(features_512)
    idx_1024 = torch.tensor(idx_1024_list)
    idx_512 = torch.tensor(idx_512_list)

    known_label = retain_random_rows(labels, retain_ratio=0.25)

    adj_propagate = torch.tensor(0)
    if is_weighted:
        adj_propagate = adj_sub * torch.tensor((sub_matrix+1))

    return images, labels, adj_sub, adj_propagate, positions, idx_1024, idx_512, features_1024, features_512, graph, known_label


def collate_fn_test(batch, adj_matrix, graph, is_weighted=False):
    images, labels, positions, idx_1024_list, idx_512_list, features_1024, features_512, idx_matrix, level = zip(*batch)

    images = torch.stack(images)
    labels = torch.tensor(labels)
    positions = torch.tensor(positions)

    features_1024 = torch.tensor(features_1024)
    features_512 = torch.tensor(features_512)
    idx_1024 = torch.tensor(idx_1024_list)
    idx_512 = torch.tensor(idx_512_list)

    indices = [i for i, x in enumerate(level) if x > 1]
    indices_eval = [i for i, x in enumerate(level) if x == 1]

    k = min(np.array(idx_matrix).shape[0]-1, 12)
    num_random_edges = min(np.array(idx_matrix).shape[0]-1, 4)
    sub_matrix = extract_submatrix_3D(adj_matrix, np.array(idx_matrix))
    sub_inter_matrix = inter_layer_from_distance_matrix(sub_matrix, positions)

    adj_sub = build_adj_matrix_with_similarity_and_known_labels(sub_matrix, indices_eval, k=k, num_random_edges=num_random_edges)+sub_inter_matrix
    adj_sub[adj_sub > 0] = 1

    known_label = copy.deepcopy(labels)
    eval_label = copy.deepcopy(labels)
    known_label[indices] = 0
    eval_label[indices_eval] = 0
    adj_propagate = torch.tensor(0)
    if is_weighted:
        adj_propagate = adj_sub * torch.tensor((sub_matrix+1))

    return images, eval_label, adj_sub, adj_propagate, positions, idx_1024, idx_512, features_1024, features_512, graph, known_label, indices


def collate_fn_result(batch, adj_matrix, graph, is_weighted=False):
    images, labels, positions, idx_1024_list, idx_512_list, features_1024, features_512, idx_matrix, level = zip(*batch)

    images = torch.stack(images)
    labels = torch.tensor(labels)
    positions = torch.tensor(positions)

    features_1024 = torch.tensor(features_1024)
    features_512 = torch.tensor(features_512)
    idx_1024 = torch.tensor(idx_1024_list)
    idx_512 = torch.tensor(idx_512_list)

    indices = [i for i, x in enumerate(level) if x > 1]
    indices_eval = [i for i, x in enumerate(level) if x == 1]

    k = min(np.array(idx_matrix).shape[0]-1, 12)
    num_random_edges = min(np.array(idx_matrix).shape[0]-1, 10)
    sub_matrix = extract_submatrix_3D(adj_matrix, np.array(idx_matrix))
    sub_inter_matrix = inter_layer_from_distance_matrix(sub_matrix, positions, k=4)

    adj_sub = build_adj_matrix_with_similarity_and_known_labels(sub_matrix, indices_eval, k=k, num_random_edges=num_random_edges)+sub_inter_matrix
    adj_sub[adj_sub > 0] = 1

    known_label = copy.deepcopy(labels)
    eval_label = copy.deepcopy(labels)
    known_label[indices] = 0
    eval_label[indices_eval] = 0

    adj_propagate = torch.tensor(0)
    if is_weighted:
        adj_propagate = adj_sub * torch.tensor((sub_matrix+1))
    level = torch.tensor(level)

    return images, eval_label, adj_sub, adj_propagate, positions, idx_1024, idx_512, features_1024, features_512, graph, known_label, indices, level


def create_dataloaders_for_train_file(npy_file_paths, batch_size=32, transform=None):

    dataloaders = {}

    for npy_file in npy_file_paths:
        name = npy_file.split('/')[-1][:-4]
        # graph_path = f"./HER2_3D/3D_graph_data/{name}.pt"
        # adj_matrix_path = f'./HER2_3D/3D_combined_information/{name}.csv'

        graph_path = f'./ST_Breast_3D/3D_graph_data/{name}.pt'
        adj_matrix_path = f'./ST_Breast_3D/3D_combined_information/{name}.csv'

        dataset = ImageGraphDataset(npy_file, graph_path, adj_matrix_path, transform=transform)
        adj_matrix_df = dataset.adj_matrix
        graph_data = dataset.graph

        def collate_fn_with_params(batch, adj_matrix_dfs=adj_matrix_df, graph_datas=graph_data):
            return collate_fn(batch, adj_matrix_dfs, graph_datas, is_weighted=True)

        dataloaders[npy_file] = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=collate_fn_with_params)

    return dataloaders


def create_dataloaders_for_result_file(npy_file_paths, batch_size=32, transform=None):
    dataloaders = {}
    testloader = {}
    for npy_file in npy_file_paths:
        name = npy_file.split('/')[-1][:-4]
        # graph_path = f"./HER2_3D/3D_graph_data/{name}.pt"
        # adj_matrix_path = f'./HER2_3D/3D_combined_information/{name}.csv'

        graph_path = f'./ST_Breast_3D/3D_graph_data/{name}.pt'
        adj_matrix_path = f'./ST_Breast_3D/3D_combined_information/{name}.csv'

        dataset = ImageGraphDataset(npy_file, graph_path, adj_matrix_path, transform=transform)
        adj_matrix_df = dataset.adj_matrix
        graph_data = dataset.graph

        def collate_fn_with_params(batch, adj_matrix_dfs=adj_matrix_df, graph_datas=graph_data):
            return collate_fn(batch, adj_matrix_dfs, graph_datas, is_weighted=False)

        def collate_fn_test_with_params(batch, adj_matrix_dfs=adj_matrix_df, graph_datas=graph_data):
            return collate_fn_result(batch, adj_matrix_dfs, graph_datas, is_weighted=False)

        dataloaders[npy_file] = DataLoader(dataset.get_train_dataset(), batch_size=batch_size, shuffle=True,
                                           collate_fn=collate_fn_with_params)

        testloader[npy_file] = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                          collate_fn=collate_fn_test_with_params)

    return dataloaders, testloader


# def create_dataloaders_for_test_file(npy_file_paths, batch_size=32, transform=None):
#     dataloaders = {}
#     testloader = {}
#     for npy_file in npy_file_paths:
#         name = npy_file.split('/')[-1][:-4]
#         # graph_path = f'./HER2_3D/3D_graph_data/{name}.pt'
#         # adj_matrix_path = f'./HER2_3D/3D_combined_information/{name}.csv'
#
#         graph_path = f'./ST_Breast_3D/3D_graph_data/{name}.pt'
#         adj_matrix_path = f'./ST_Breast_3D/3D_combined_information/{name}.csv'
#
#         dataset = ImageGraphDataset(npy_file, graph_path, adj_matrix_path, transform=transform)
#         adj_matrix_df = dataset.adj_matrix
#         graph_data = dataset.graph
#
#         def collate_fn_with_params(batch, adj_matrix_dfs=adj_matrix_df, graph_datas=graph_data):
#             return collate_fn(batch, adj_matrix_dfs, graph_datas, is_weighted=False)
#
#         def collate_fn_test_with_params(batch, adj_matrix_dfs=adj_matrix_df, graph_datas=graph_data):
#             return collate_fn_test(batch, adj_matrix_dfs, graph_datas, is_weighted=False)
#
#         dataloaders[npy_file] = DataLoader(dataset.get_train_dataset(), batch_size=batch_size, shuffle=True,
#                                            collate_fn=collate_fn_with_params)
#
#         testloader[npy_file] = DataLoader(dataset, batch_size=batch_size, shuffle=True,
#                                           collate_fn=collate_fn_test_with_params)
#
#     return dataloaders, testloader
