import os

import pandas as pd
import numpy as np

import torch
import networkx as nx
from scipy.spatial import distance_matrix
from torch_geometric.data import HeteroData


def build_graph_from_dict(node_dict, distance_threshold=2.1):
    node_names = list(node_dict.keys())
    coords = np.array([node_dict[node]['position'] for node in node_names])
    features = np.array([node_dict[node]['feature'] for node in node_names])
    labels = np.array([node_dict[node]['label'] for node in node_names])

    # Compute distance matrix
    dist_matrix = distance_matrix(coords, coords)

    # Build a NetworkX graph
    G = nx.Graph()
    for k, node in enumerate(node_names):
        G.add_node(k, pos=coords[k], feature=features[k], label=labels[k])

    # Add edges based on distance threshold
    for k in range(len(node_names)):
        for j in range(k + 1, len(node_names)):
            if dist_matrix[k, j] < distance_threshold:
                G.add_edge(k, j, weight=2.1 - dist_matrix[k, j])

    # Convert to PyTorch Geometric format
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    edge_weights = torch.tensor([G[k][j]['weight'] for k, j in G.edges], dtype=torch.float)
    node_features = torch.tensor([G.nodes[k]['feature'] for k in G.nodes], dtype=torch.float)
    node_labels = torch.tensor([G.nodes[k]['label'] for k in G.nodes], dtype=torch.float)

    return edge_index, edge_weights, node_features, node_labels, node_names


# 2. Build relationships between layers
def build_cross_layer_edges(upper_layer_names, lower_layer_names, cross_layer_dict):
    cross_layer_edges = []
    for upper_node, lower_nodes in cross_layer_dict.items():
        if upper_node in upper_layer_names:
            upper_index = upper_layer_names.index(upper_node)
            for lower_node in lower_nodes:
                if lower_node in lower_layer_names:
                    lower_index = lower_layer_names.index(lower_node)
                    cross_layer_edges.append((upper_index, lower_index))
    return cross_layer_edges


root_dir = './ST_Breast/patches_csv/patches_224'
for sample_dir in os.listdir(root_dir):
    label_file_dir_224 = f'./ST_Breast/gene_expression/224/{sample_dir}'
    feature_file_dir_224 = f'./ST_Breast/extracted_feature/224/{sample_dir}'
    position_file_dir_224 = f'./ST_Breast/patches_csv/patches_224/{sample_dir}'

    label_file_dir_512 = f'./ST_Breast/gene_expression/512/{sample_dir}'
    feature_file_dir_512 = f'./ST_Breast/extracted_feature/512/{sample_dir}'
    position_file_dir_512 = f'./ST_Breast/patches_csv/patches_512/{sample_dir}'

    label_file_dir_1024 = f'./ST_Breast/gene_expression/1024/{sample_dir}'
    feature_file_dir_1024 = f'./ST_Breast/extracted_feature/1024/{sample_dir}'
    position_file_dir_1024 = f'./ST_Breast/patches_csv/patches_1024/{sample_dir}'

    save_dir = f'./ST_Breast/graph_data/{sample_dir}'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for label_file_224 in os.listdir(label_file_dir_224):
        print(label_file_224)
        layer_dict_224 = {}
        layer_dict_512 = {}
        layer_dict_1024 = {}

        relationship_512_224 = {}
        relationship_1024_512 = {}

        tmp_file = label_file_224[:-4]

        tmp_label_file_224 = pd.read_csv(os.path.join(label_file_dir_224, label_file_224))

        feature_file_224_path = os.path.join(feature_file_dir_224, f'{tmp_file}.npy')
        feature_file_224 = np.load(feature_file_224_path, allow_pickle=True)

        position_file_224_path = os.path.join(position_file_dir_224, f'{tmp_file}.csv')
        position_file_224 = pd.read_csv(position_file_224_path)
        column_names = ['barcode', 'pixel_x', 'pixel_y', 'x', 'y']
        position_file_224.columns = column_names

        feature_file_224 = feature_file_224.item()

        for spot in tmp_label_file_224.columns[1:]:
            label = np.array(tmp_label_file_224[spot])

            x_series = position_file_224.loc[position_file_224.iloc[:, 0] == spot]['x']
            y_series = position_file_224.loc[position_file_224.iloc[:, 0] == spot]['y']

            # Check if Series is empty
            if not x_series.empty and not y_series.empty:
                try:
                    position = [int(x_series.iloc[0]), int(y_series.iloc[0])]
                except (ValueError, TypeError) as e:
                    print(f"Skipping due to conversion error: {e}")
                    # You can choose to log this issue or continue execution
                    continue
            else:
                print(f"Skipping because x or y Series is empty for spot {spot}")
                continue

            feature = feature_file_224[spot]
            layer_dict_224[spot] = {'feature': feature, 'label': label, 'position': position}

        label_file_512 = os.path.join(label_file_dir_512, f'{tmp_file}_patches_512.csv')
        tmp_label_file_512 = pd.read_csv(os.path.join(label_file_dir_512, label_file_512))

        feature_file_512_path = os.path.join(feature_file_dir_512, f'{tmp_file}.npy')
        feature_file_512 = np.load(feature_file_512_path, allow_pickle=True)
        feature_file_512 = feature_file_512.item()

        position_file_512_path = os.path.join(position_file_dir_512, f'{tmp_file}_patches_512.csv')
        position_file_512 = pd.read_csv(position_file_512_path)

        for spot in tmp_label_file_512.columns:
            label = np.array(tmp_label_file_512[spot])
            feature = feature_file_512[spot[:-4]]
            position = [int(position_file_512.loc[position_file_512.iloc[:, 0] == spot]['i']),
                        int(position_file_512.loc[position_file_512.iloc[:, 0] == spot]['j'])]
            layer_dict_512[spot] = {'feature': feature, 'label': label, 'position': position}

        label_file_1024 = os.path.join(label_file_dir_1024, f'{tmp_file}_patches_1024.csv')
        tmp_label_file_1024 = pd.read_csv(os.path.join(label_file_dir_512, label_file_1024))

        feature_file_1024_path = os.path.join(feature_file_dir_1024, f'{tmp_file}.npy')
        feature_file_1024 = np.load(feature_file_1024_path, allow_pickle=True)
        feature_file_1024 = feature_file_1024.item()

        position_file_1024_path = os.path.join(position_file_dir_1024, f'{tmp_file}_patches_1024.csv')
        position_file_1024 = pd.read_csv(position_file_1024_path)

        for spot in tmp_label_file_1024.columns:
            label = np.array(tmp_label_file_1024[spot])
            feature = feature_file_1024[spot[:-4]]
            position = [int(position_file_1024.loc[position_file_1024.iloc[:, 0] == spot]['i']),
                        int(position_file_1024.loc[position_file_1024.iloc[:, 0] == spot]['j'])]
            layer_dict_1024[spot] = {'feature': feature, 'label': label, 'position': position}

        relation_512_csv = position_file_512
        for i in range(len(position_file_512)):
            tmp_img = position_file_512['patch_filename'][i]
            tmp_points = position_file_512['points'][i]
            relationship_512_224[tmp_img] = tmp_points.split(', ')

        relation_1024_csv = position_file_1024
        for i in range(len(position_file_1024)):
            tmp_img = position_file_1024['patch_filename'][i]
            tmp_points = position_file_1024['overlapping_512_patches'][i]
            relationship_1024_512[tmp_img] = tmp_points.split(', ')

        layer1_edge_index, layer1_edge_weights, layer1_features, layer1_labels, layer1_names = build_graph_from_dict(
            layer_dict_224)
        layer2_edge_index, layer2_edge_weights, layer2_features, layer2_labels, layer2_names = build_graph_from_dict(
            layer_dict_512)
        layer3_edge_index, layer3_edge_weights, layer3_features, layer3_labels, layer3_names = build_graph_from_dict(
            layer_dict_1024)

        layer1_to_layer2_edges = build_cross_layer_edges(layer2_names, layer1_names, relationship_512_224)
        layer2_to_layer3_edges = build_cross_layer_edges(layer3_names, layer2_names, relationship_1024_512)

        data = HeteroData()

        # Add graph for each layer
        data['layer_224'].x = layer1_features
        data['layer_224'].edge_index = layer1_edge_index
        data['layer_224'].edge_attr = layer1_edge_weights
        data['layer_224'].y = layer1_labels

        data['layer_512'].x = layer2_features
        data['layer_512'].edge_index = layer2_edge_index
        data['layer_512'].edge_attr = layer2_edge_weights
        data['layer_512'].y = layer2_labels

        data['layer_1024'].x = layer3_features
        data['layer_1024'].edge_index = layer3_edge_index
        data['layer_1024'].edge_attr = layer3_edge_weights
        data['layer_1024'].y = layer3_labels

        # Add inter-layer edges
        data['layer_512', 'to', 'layer_224'].edge_index = torch.tensor(layer1_to_layer2_edges).t().contiguous()
        data['layer_1024', 'to', 'layer_512'].edge_index = torch.tensor(layer2_to_layer3_edges).t().contiguous()

        save_path = os.path.join(save_dir, f'{label_file_224[:-4]}.pt')
        torch.save(data, save_path)
        # Output the data structure
        print(data)
        print(f"HeteroData object has been saved to {save_path}")
