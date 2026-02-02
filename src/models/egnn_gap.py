from __future__ import annotations
import torch
import torch.nn as nn
from egnn_pytorch import EGNN_Network


class EGNNEncoder(nn.Module):
    def __init__(self, num_atom_types: int = 100, emb_dim: int = 128, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        self.emb = nn.Embedding(num_atom_types, emb_dim)
        self.egnn = EGNN_Network(
            dim=emb_dim,
            depth=depth,
            num_nearest_neighbors=0,  # fully-connected
            dropout=dropout,
        )

    def forward(self, z, pos, mask):
        feats = self.emb(z) # (B, N, D)
        feats, _ = self.egnn(feats, pos, mask=mask)

        # masked mean pooling -> (B, D)
        m = mask.unsqueeze(-1).float()
        pooled = (feats * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return pooled


class EGNNGapRegressor(nn.Module):
    def __init__(self, num_atom_types: int = 100, emb_dim: int = 128, depth: int = 4, mlp_hidden: int = 256):
        super().__init__()
        self.encoder = EGNNEncoder(num_atom_types=num_atom_types, emb_dim=emb_dim, depth=depth)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, z, pos, mask):
        h = self.encoder(z, pos, mask)
        return self.head(h).squeeze(-1)
