import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class gs_block_with_attention_fixed_weights_3d(nn.Module):
    def __init__(self, feature_dim, embed_dim, policy='mean', gcn=False, attention=True,
                 use_fixed_weights=False, fixed_weight1=None, fixed_weight2=None):
        super().__init__()
        self.gcn = gcn
        self.policy = policy
        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.attention = attention
        self.use_fixed_weights = use_fixed_weights  # Whether to use fixed weights

        # Learnable weight parameter
        self.weight = nn.Parameter(torch.FloatTensor(
            embed_dim,
            self.feat_dim if self.gcn else 2 * self.feat_dim
        ))
        init.xavier_uniform_(self.weight)

        # If using attention, initialize attention weight
        if self.attention:
            self.att_weight = nn.Parameter(torch.FloatTensor(self.feat_dim, 1))
            init.xavier_uniform_(self.att_weight)

        # Define fixed weight matrices (if enabled)
        if use_fixed_weights:
            self.fixed_weight1 = fixed_weight1 if fixed_weight1 is not None else torch.eye(feature_dim)
            self.fixed_weight2 = fixed_weight2 if fixed_weight2 is not None else torch.eye(feature_dim)
            self.fixed_weight1.requires_grad = False
            self.fixed_weight2.requires_grad = False

    def forward(self, x, Adj, x_label, known_label_mask, Adj_propagate=None, is_weighted=False):
        # Check whether to use fixed weights
        if self.use_fixed_weights:
            # Use fixed weight matrices to transform input features
            x_transformed = x @ self.fixed_weight1 + x @ self.fixed_weight2
        else:
            x_transformed = x  # If not using fixed weights, keep original features

        # Label propagation:
        # Predict unknown labels by diffusing known labels through the graph structure
        # If `is_weighted` is True, use weighted adjacency matrix
        if is_weighted:
            label_propagation = self.propagate_labels(x_label, Adj_propagate, known_label_mask)
        else:
            label_propagation = self.propagate_labels(x_label, Adj, known_label_mask)

        # Aggregate neighbor features
        neigh_feats = self.aggregate(x_transformed, Adj)

        if not self.gcn:
            combined = torch.cat([x, neigh_feats], dim=1)
        else:
            combined = neigh_feats

        combined = F.relu(self.weight.mm(combined.T)).T
        combined = F.normalize(combined, 2, 1)

        # Return both the GCN-processed features and the propagated labels
        return combined, label_propagation

    def propagate_labels(self, x_label, Adj, known_label_mask):
        """
        Label propagation process that only updates unknown nodes.

        Args:
            x_label (torch.Tensor): Input label matrix, shape [num_nodes, num_classes]
            Adj (torch.Tensor): Adjacency matrix, shape [num_nodes, num_nodes]
            known_label_mask (torch.Tensor): Binary mask of known labels, shape [num_nodes],
                                             where known labels = 1, unknown = 0
        Returns:
            torch.Tensor: Label matrix after propagation
        """
        adj = Adj.to(Adj.device)
        row_sum = adj.sum(dim=1, keepdim=True)
        adj_normalized = adj / row_sum.clamp(min=1e-10)  # Avoid division by zero
        x_label_transformed = x_label.to(Adj.device)  # Initial labels

        for _ in range(20):  # Iterate label propagation
            # Propagate labels through neighbors
            propagated_labels = torch.mm(adj_normalized, x_label_transformed)

            # Only update labels for unknown nodes
            x_label_transformed = torch.where(known_label_mask.unsqueeze(1) == 1, x_label, propagated_labels)

        return x_label_transformed

    def aggregate(self, x, Adj):
        adj = Adj.to(Adj.device)
        if not self.gcn:
            n = len(adj)
            adj = adj - torch.eye(n).to(adj.device)

        if self.policy == 'mean' and self.attention:
            # Compute attention scores for node features
            att_scores = torch.matmul(x, self.att_weight).squeeze()  # [N, 1] → [N]
            att_scores = F.softmax(att_scores, dim=0)  # Normalize

            # Compute weighted average of neighbor features
            num_neigh = adj.sum(1, keepdim=True)
            mask = adj.div(num_neigh)  # Normalize adjacency
            weighted_feats = mask * att_scores.unsqueeze(1)  # Apply attention scores
            to_feats = weighted_feats.mm(x)  # Weighted sum of neighbors

        elif self.policy == 'mean':
            num_neigh = adj.sum(1, keepdim=True)
            mask = adj.div(num_neigh)
            to_feats = mask.mm(x)

        elif self.policy == 'max':
            indexs = [i.nonzero() for i in adj == 1]
            to_feats = []
            for feat in [x[i.squeeze()] for i in indexs]:
                if len(feat.size()) == 1:
                    to_feats.append(feat.view(1, -1))
                else:
                    to_feats.append(torch.max(feat, 0)[0].view(1, -1))
            to_feats = torch.cat(to_feats, 0)

        return to_feats
