import os
import random
import torch
from torch_geometric.data import HeteroData
import shutil

"""
Step 4: Combine previous 2D WSI graph into 3D sample graph
Input: .pt files of each WSI
Output: .pt file for samples
"""


def load_hetero_data(file_path):
    return torch.load(file_path)


file_dir = './Human_dorsolateral_3D/graph_data'
for graph_dir in os.listdir(file_dir):
    file_list = os.listdir(os.path.join(file_dir, graph_dir))

    data_list = [load_hetero_data(os.path.join(file_dir, graph_dir, path)) for path in file_list]

    node_offsets_512 = data_list[0]['layer_512'].x.shape[0]
    node_offsets_1024 = data_list[0]['layer_1024'].x.shape[0]
    merged_data = data_list[0]

    for graph_information in data_list[1:]:
        merged_data['layer_512'].x = torch.cat([merged_data['layer_512'].x, graph_information['layer_512'].x], dim=0)
        merged_data['layer_512'].y = torch.cat([merged_data['layer_512'].y, graph_information['layer_512'].y], dim=0)

        edge_index_512 = graph_information['layer_512'].edge_index + node_offsets_512
        merged_data['layer_512'].edge_index = torch.cat([merged_data['layer_512'].edge_index, edge_index_512], dim=1)

        merged_data['layer_1024'].x = torch.cat([merged_data['layer_1024'].x, graph_information['layer_1024'].x], dim=0)
        merged_data['layer_1024'].y = torch.cat([merged_data['layer_1024'].y, graph_information['layer_1024'].y], dim=0)

        edge_index_1024 = graph_information['layer_1024'].edge_index + node_offsets_1024
        merged_data['layer_1024'].edge_index = torch.cat([merged_data['layer_1024'].edge_index, edge_index_1024], dim=1)

        node_offsets_512 += graph_information['layer_512'].x.shape[0]
        node_offsets_1024 += graph_information['layer_1024'].x.shape[0]

    print(merged_data)
    print(f"./Human_dorsolateral_3D/3D_graph_data/{graph_dir}.pt")
    print('\n')
    torch.save(merged_data, f"./Human_dorsolateral_3D/3D_graph_data/{graph_dir}.pt")




