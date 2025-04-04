import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable


class gs_block_with_attention_fixed_weights(nn.Module):
    def __init__(self, feature_dim, embed_dim, policy='mean', gcn=False, attention=True,
                 use_fixed_weights=False, fixed_weight1=None, fixed_weight2=None):
        super().__init__()
        self.gcn = gcn
        self.policy = policy
        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.attention = attention
        self.use_fixed_weights = use_fixed_weights

        # Learnable weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(
            embed_dim,
            self.feat_dim if self.gcn else 2 * self.feat_dim
        ))
        init.xavier_uniform_(self.weight)

        # Add attention parameters if attention is enabled
        if self.attention:
            self.att_weight = nn.Parameter(torch.FloatTensor(self.feat_dim, 1))
            init.xavier_uniform_(self.att_weight)

        # Fixed weight matrices (if enabled)
        if use_fixed_weights:
            self.fixed_weight1 = fixed_weight1 if fixed_weight1 is not None else torch.eye(feature_dim)
            self.fixed_weight2 = fixed_weight2 if fixed_weight2 is not None else torch.eye(feature_dim)
            self.fixed_weight1.requires_grad = False
            self.fixed_weight2.requires_grad = False

    def forward(self, x, Adj):
        # Check whether to use fixed weights
        if self.use_fixed_weights:
            x_transformed = x @ self.fixed_weight1 + x @ self.fixed_weight2
        else:
            x_transformed = x

        neigh_feats = self.aggregate(x_transformed, Adj)

        if not self.gcn:
            combined = torch.cat([x, neigh_feats], dim=1)
        else:
            combined = neigh_feats

        combined = F.relu(self.weight.mm(combined.T)).T
        combined = F.normalize(combined, 2, 1)
        return combined

    def aggregate(self, x, Adj):
        adj = Adj.to(Adj.device)
        if not self.gcn:
            n = len(adj)
            adj = adj - torch.eye(n).to(adj.device)

        # Apply attention mechanism for weighted neighbor aggregation
        if self.policy == 'mean' and self.attention:
            att_scores = torch.matmul(x, self.att_weight).squeeze()
            att_scores = F.softmax(att_scores, dim=0)

            num_neigh = adj.sum(1, keepdim=True)
            mask = adj.div(num_neigh)
            weighted_feats = mask * att_scores.unsqueeze(1)
            to_feats = weighted_feats.mm(x)

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
