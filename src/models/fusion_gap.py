from __future__ import annotations
import torch
import torch.nn as nn

from .egnn_gap import EGNNEncoder


class EGNNTDARegressor(nn.Module):
    def __init__(self, tda_dim: int, num_atom_types: int = 100, emb_dim: int = 128, depth: int = 4, mlp_hidden: int = 256):
        super().__init__()
        self.encoder = EGNNEncoder(num_atom_types=num_atom_types, emb_dim=emb_dim, depth=depth)

        self.head = nn.Sequential(
            nn.Linear(emb_dim + tda_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, z, pos, mask, tda_vec):
        h = self.encoder(z, pos, mask) # (B, emb_dim)
        x = torch.cat([h, tda_vec], dim=-1) # (B, emb_dim + tda_dim)
        return self.head(x).squeeze(-1)
