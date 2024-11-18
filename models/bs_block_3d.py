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
        self.use_fixed_weights = use_fixed_weights  # Control whether to use fixed weights

        # Weight parameters
        self.weight = nn.Parameter(torch.FloatTensor(
            embed_dim,
            self.feat_dim if self.gcn else 2 * self.feat_dim
        ))
        init.xavier_uniform_(self.weight)

        # Add attention parameters if attention is used
        if self.attention:
            self.att_weight = nn.Parameter(torch.FloatTensor(self.feat_dim, 1))
            init.xavier_uniform_(self.att_weight)

        # Introduce fixed weight matrices (only effective when use_fixed_weights is True)
        if use_fixed_weights:
            self.fixed_weight1 = fixed_weight1 if fixed_weight1 is not None else torch.eye(feature_dim)
            self.fixed_weight2 = fixed_weight2 if fixed_weight2 is not None else torch.eye(feature_dim)
            # Ensure fixed weights do not participate in training
            self.fixed_weight1.requires_grad = False
            self.fixed_weight2.requires_grad = False

    def forward(self, x, Adj, x_label, known_label_mask):
        # Check whether to use fixed weights
        if self.use_fixed_weights:
            # Transform features using fixed weight matrices
            x_transformed = x @ self.fixed_weight1 + x @ self.fixed_weight2
        else:
            x_transformed = x  # If not using fixed weights, directly use original input features

        # New: label propagation process
        # Use label propagation weights and adjacency matrix to propagate x_label, obtaining predictions for unknown labels
        label_propagation = self.propagate_labels(x_label, Adj, known_label_mask)

        # Aggregate features
        neigh_feats = self.aggregate(x_transformed, Adj)

        if not self.gcn:
            combined = torch.cat([x, neigh_feats], dim=1)
        else:
            combined = neigh_feats

        combined = F.relu(self.weight.mm(combined.T)).T
        combined = F.normalize(combined, 2, 1)

        # Output: return labels after label propagation and features processed by GCN
        return combined, label_propagation

    def propagate_labels(self, x_label, Adj, known_label_mask):
        """
        Label propagation process that only affects unknown nodes.

        Args:
            x_label (torch.Tensor): Input label matrix with shape [num_nodes, num_classes]
            Adj (torch.Tensor): Adjacency matrix with shape [num_nodes, num_nodes]
            known_label_mask (torch.Tensor): Mask for known labels with shape [num_nodes], where 1 indicates known labels and 0 indicates unknown labels

        Returns:
            torch.Tensor: Label matrix after propagation
        """
        adj = Adj.to(Adj.device)
        row_sum = adj.sum(dim=1, keepdim=True)
        adj_normalized = adj / row_sum.clamp(min=1e-10)  # Avoid division by zero

        x_label_transformed = x_label  # Initial labels

        for _ in range(20):  # Iterate the label propagation process
            # Label propagation
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
            # Calculate attention weights between neighbor features and node features
            att_scores = torch.matmul(x, self.att_weight).squeeze()  # [N, 1] -> [N]
            att_scores = F.softmax(att_scores, dim=0)  # Normalize

            # Calculate weighted neighbor features
            num_neigh = adj.sum(1, keepdim=True)
            mask = adj.div(num_neigh)  # Normalize adjacency matrix
            weighted_feats = mask * att_scores.unsqueeze(1)  # Apply attention weights
            to_feats = weighted_feats.mm(x)  # Sum weighted neighbor features

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
