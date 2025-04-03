import random
import torch
import numpy as np
from scipy.spatial import distance_matrix, minkowski_distance, distance


def calcADJ(coord, k=8, distanceType='euclidean', pruneTag='NA'):
    r"""
    Calculate spatial adjacency matrix directly from X/Y coordinates.
    """
    spatialMatrix = coord
    nodes = spatialMatrix.shape[0]
    Adj = torch.zeros((nodes, nodes))
    for i in np.arange(spatialMatrix.shape[0]):
        tmp = spatialMatrix[i, :].reshape(1, -1)
        distMat = distance.cdist(tmp, spatialMatrix, distanceType)
        if k == 0:
            k = spatialMatrix.shape[0] - 1
        res = distMat.argsort()[:k + 1]
        tmpdist = distMat[0, res[0][1:k + 1]]
        boundary = np.mean(tmpdist) + np.std(tmpdist)  # Optional threshold

        for j in np.arange(1, k + 1):
            # No pruning
            if pruneTag == 'NA':
                Adj[i][res[0][j]] = 1.0
            elif pruneTag == 'STD':
                if distMat[0, res[0][j]] <= boundary:
                    Adj[i][res[0][j]] = 1.0
            # Grid pruning: use only neighbors within fixed radius
            elif pruneTag == 'Grid':
                if distMat[0, res[0][j]] <= 2.0:
                    Adj[i][res[0][j]] = 1.0
    return Adj


def edge_index_to_adj_matrix(graph_data):
    """
    Convert edge_index in a graph data object to an adjacency matrix.

    Args:
    - graph_data: Graph data object containing edge_index

    Returns:
    - adj_matrix: The resulting adjacency matrix
    """
    edge_index = graph_data.edge_index  # [2, num_edges]
    num_nodes = graph_data.num_nodes

    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    # Map each edge (i, j) in edge_index to adj_matrix
    adj_matrix[edge_index[0], edge_index[1]] = 1
    adj_matrix[edge_index[1], edge_index[0]] = 1

    return adj_matrix


def inter_layer_from_distance_matrix(dist_matrix, positions, k=8):
    """
    Construct an inter-layer adjacency matrix by using the distance matrix.
    Only connects to same-layer spots (distance = 0).

    Returns:
    - Symmetric binary adjacency matrix.
    """
    n = dist_matrix.shape[0]
    result = np.zeros((n, n), dtype='float32')

    for i in range(n):
        # Find indices with distance 0 (same-layer spots)
        zero_indices = np.where(dist_matrix[i] == 0)[0]
        zero_indices = zero_indices[zero_indices != i]

        if len(zero_indices) == 0:
            continue

        pivot = positions[i, :]
        candidate_coords = positions[zero_indices, :]
        distances = np.linalg.norm(candidate_coords - pivot, axis=1)

        # Select k nearest indices
        k_sel = min(k, len(distances))
        nearest_order = np.argsort(distances)[:k_sel]
        selected_indices = zero_indices[nearest_order]

        result[i, selected_indices] = 1

    symmetric_result = np.maximum(result, result.T)
    return symmetric_result


def extract_submatrix_3D(df, idx):
    # Extract submatrix from given row and column indices
    submatrix = df.iloc[idx - 1, idx].to_numpy()
    return torch.tensor(submatrix.astype(np.float32), dtype=torch.float32)


def calcADJ_from_distance_matrix_3D(dist_matrix, k=16):
    """
    Construct a symmetric adjacency matrix from a symmetric distance matrix,
    by selecting top-k highest values per node.

    Args:
        dist_matrix (torch.Tensor): Symmetric [n x n] distance matrix.
        k (int): Number of neighbors to connect per node.

    Returns:
        torch.Tensor: The resulting adjacency matrix.
    """
    nodes = dist_matrix.shape[0]
    Adj = torch.zeros((nodes, nodes))

    for i in range(nodes):
        top_k_indices = torch.topk(dist_matrix[i], k, largest=True).indices
        for j in top_k_indices:
            Adj[i, j] = 1.0
            Adj[j, i] = 1.0

    return Adj


def build_adj_matrix_with_similarity_and_known_labels(feature_matrix, known_label_indices, k=12, num_random_edges=4):
    """
    Build an adjacency matrix based on fixed number of similarity-based edges,
    and connect known-labeled nodes with random neighbors.

    Args:
        feature_matrix (torch.Tensor): Node similarity matrix [num_nodes x num_nodes]
        known_label_indices (list or tensor): Indices of nodes with known labels
        k (int): Top-k most similar nodes to connect for each node
        num_random_edges (int): Number of random edges to add for each known label node

    Returns:
        torch.Tensor: The resulting adjacency matrix
    """
    num_nodes = feature_matrix.shape[0]
    adj_matrix = torch.zeros((num_nodes, num_nodes))

    # Connect each node to its top-k most similar neighbors
    for i in range(num_nodes):
        similarity_scores = feature_matrix[i].clone()
        similarity_scores[i] = -float('inf')  # Exclude self

        top_k_indices = torch.topk(similarity_scores, k=k).indices
        for j in top_k_indices:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

        known_label_similarity = np.full_like(similarity_scores, -100)
        known_label_similarity[known_label_indices] = similarity_scores[known_label_indices]
        known_label_similarity[i] = -float('inf')
        top_k_indices = torch.topk(similarity_scores, k=num_random_edges).indices
        for j in top_k_indices:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

    # For known-labeled nodes, add random neighbors, increasing randomness for subgraph training
    for node in known_label_indices:
        possible_nodes = list(range(num_nodes))
        possible_nodes.remove(node)
        random_neighbors = random.sample(possible_nodes, num_random_edges)

        for neighbor in random_neighbors:
            adj_matrix[node, neighbor] = 1
            adj_matrix[neighbor, node] = 1

    return adj_matrix
