from __future__ import annotations

import torch
import torch.nn as nn

from .egnn_gap import EGNNEncoder

class EGNNTDAFiLMRegressor(nn.Module):
    """
    EGNN + TDA via FiLM (Feature-wise Linear Modulation).
    Idea:
      - EGNN produces a graph-level embedding h ∈ R^{emb_dim}.
      - TDA produces a global topological descriptor t ∈ R^{tda_dim}.
      - FiLM uses t to generate (gamma, beta) and modulates h:
            (gamma, beta) = MLP(t)
            h' = (1 + gamma) ⊙ h + beta
      - Then a small head predicts HOMO–LUMO gap from h'.
    """

    def __init__(
        self,
        tda_dim: int,
        num_atom_types: int = 100,
        emb_dim: int = 128,
        depth: int = 4,
        film_hidden: int = 256,
        head_hidden: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Geometric encoder: E(n)-equivariant EGNN -> invariant graph embedding
        self.encoder = EGNNEncoder(
            num_atom_types=num_atom_types,
            emb_dim=emb_dim,
            depth=depth,
            dropout=dropout,
        )

        # FiLM generator: tda_vec -> (gamma, beta) in R^{emb_dim}
        # We use tanh to keep modulation bounded (stabilizes training)
        self.film = nn.Sequential(
            nn.Linear(tda_dim, film_hidden),
            nn.SiLU(),
            nn.Linear(film_hidden, 2 * emb_dim),
        )

        self.film_act = nn.Tanh()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 3) Regression head: h' -> scalar
        self.head = nn.Sequential(
            nn.Linear(emb_dim, head_hidden),
            nn.SiLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, z, pos, mask, tda_vec):
        """
        Inputs:
          z: (B, N) atomic numbers
          pos: (B, N, 3) coordinates
          mask:(B, N) valid atom mask
          tda_vec:(B, tda_dim) cached TDA features (Betti curves + entropy)
        Returns:
          (B,) predicted HOMO–LUMO gap
        """
        # EGNN embedding (invariant to rotations/translations after pooling)
        h = self.encoder(z, pos, mask) # (B, emb_dim)

        # FiLM parameters from TDA
        gb = self.film(tda_vec) # (B, 2*emb_dim)
        gamma, beta = gb.chunk(2, dim=-1)

        # Bound gamma/beta for stability
        gamma = self.film_act(gamma)
        beta = self.film_act(beta)

        # FiLM modulation
        h_mod = (1.0 + gamma) * h + beta
        h_mod = self.dropout(h_mod)

        # Prediction
        out = self.head(h_mod).squeeze(-1)
        return out