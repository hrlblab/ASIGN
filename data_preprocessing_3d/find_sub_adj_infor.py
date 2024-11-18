import numpy as np
import pandas as pd
import random
import torch
import numpy as np


def extract_submatrix(df, idx):
    submatrix = df.iloc[idx, idx].to_numpy()  # Extract submatrix at specified row and column indices
    return submatrix


def calcADJ_from_distance_matrix(dist_matrix, k=8):
    nodes = dist_matrix.shape[0]
    Adj = torch.zeros((nodes, nodes))  # Initialize adjacency matrix with zeros

    # Iterate over each node
    for i in range(nodes):
        # Select the k nearest neighbors with the highest distance values for node i
        top_k_indices = torch.topk(dist_matrix[i], k, largest=True).indices

        # Set the corresponding positions in the adjacency matrix to 1 and ensure symmetry
        for j in top_k_indices:
            Adj[i, j] = 1.0
            Adj[j, i] = 1.0  # Ensure symmetry

    return Adj
