import random
import torch
import numpy as np
from scipy.spatial import distance_matrix, minkowski_distance, distance


def calcADJ(coord, k=8, distanceType='euclidean', pruneTag='NA'):
    r"""
    Calculate the spatial matrix directly using X/Y coordinates.
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
        boundary = np.mean(tmpdist) + np.std(tmpdist)  # Optional
        for j in np.arange(1, k + 1):
            # No pruning
            if pruneTag == 'NA':
                Adj[i][res[0][j]] = 1.0
            elif pruneTag == 'STD':
                if distMat[0, res[0][j]] <= boundary:
                    Adj[i][res[0][j]] = 1.0
            # Prune: only use the nearest neighbor as an exact grid (6 for cityblock, 8 for Euclidean)
            elif pruneTag == 'Grid':
                if distMat[0, res[0][j]] <= 2.0:
                    Adj[i][res[0][j]] = 1.0
    return Adj


def edge_index_to_adj_matrix(graph_data):
    """
    Convert edge_index in graph data to an adjacency matrix.

    Args:
    - graph_data: Graph data object containing edge_index.

    Returns:
    - adj_matrix: Converted adjacency matrix.
    """
    edge_index = graph_data.edge_index  # [2, num_edges]
    num_nodes = graph_data.num_nodes  # Get the number of nodes

    # Create a zero tensor to store the adjacency matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    # Map each edge (i, j) in edge_index to adj_matrix
    adj_matrix[edge_index[0], edge_index[1]] = 1

    # If it is an undirected graph, add symmetric edges
    adj_matrix[edge_index[1], edge_index[0]] = 1

    return adj_matrix


def extract_submatrix_3D(df, idx):
    submatrix = df.iloc[idx-1, idx].to_numpy()  # Extract the specified row and column submatrix
    return torch.tensor(submatrix.astype(np.float32), dtype=torch.float32)


def calcADJ_from_distance_matrix_3D(dist_matrix, k=16):
    """
    Calculate a symmetric adjacency matrix based on a symmetric distance matrix by selecting the top-k highest values for each node.

    Args:
        dist_matrix (torch.Tensor): Symmetric n*n distance matrix.
        k (int): Number of farthest neighboring nodes to connect for each node.

    Returns:
        torch.Tensor: Constructed symmetric adjacency matrix.
    """
    nodes = dist_matrix.shape[0]
    Adj = torch.zeros((nodes, nodes))  # Initialize the adjacency matrix to 0

    # Iterate over each node
    for i in range(nodes):
        # Select the top k neighbors with the highest distances for node i
        top_k_indices = torch.topk(dist_matrix[i], k, largest=True).indices

        # Set the corresponding positions in the adjacency matrix to 1 and maintain symmetry
        for j in top_k_indices:
            Adj[i, j] = 1.0
            Adj[j, i] = 1.0  # Ensure symmetry

    return Adj


def build_adj_matrix_with_similarity_and_known_labels(feature_matrix, known_label_indices, k=16, num_random_edges=4):
    """
    Build an adjacency matrix based on a fixed number of similarity edges and random connections for known label nodes.

    Args:
        feature_matrix (torch.Tensor): Node similarity matrix with shape (num_nodes, num_nodes).
        known_label_indices (list or tensor): Indices of nodes with known labels.
        k (int): Number of top similar nodes to connect for each node.
        num_random_edges (int): Number of random edges to create for each known label node.

    Returns:
        torch.Tensor: Constructed adjacency matrix.
    """
    num_nodes = feature_matrix.shape[0]

    # Initialize the adjacency matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes))

    # Select and connect the top k most similar nodes based on similarity
    for i in range(num_nodes):
        # Get similarity values for node i (excluding itself)
        similarity_scores = feature_matrix[i].clone()
        similarity_scores[i] = -float('inf')  # Exclude self-connections

        # Select the top k most similar nodes
        top_k_indices = torch.topk(similarity_scores, k=k).indices
        for j in top_k_indices:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1  # Assume an undirected graph

    # Randomly create edges for known label nodes
    for node in known_label_indices:
        # Randomly select num_random_edges different nodes to connect
        possible_nodes = list(range(num_nodes))
        possible_nodes.remove(node)  # Avoid self-connections
        random_neighbors = random.sample(possible_nodes, num_random_edges)

        # Add edges to the adjacency matrix
        for neighbor in random_neighbors:
            adj_matrix[node, neighbor] = 1
            adj_matrix[neighbor, node] = 1  # Assume an undirected graph

    return adj_matrix
