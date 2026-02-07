from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceImage


@dataclass
class TDAConfig:
    cache_dir: str = "artifacts/tda_cache"
    max_homology_dim: int = 1 # H0 and H1
    n_bins: int = 16 # image resolution (16x16 per homology dim)
    n_jobs: int = -1 # parallel


class TDACache:
    """
    Computes persistent homology features for each molecule (point cloud of atom coordinates)
    and stores them as .npy files indexed by the global QM9 index.
    """
    def __init__(self, cfg: TDAConfig):
        self.cfg = cfg
        self.cache_dir = Path(cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.vr = VietorisRipsPersistence(
            homology_dimensions=list(range(cfg.max_homology_dim + 1)),
            metric="euclidean",
            n_jobs=cfg.n_jobs,
        )
        self.pi = PersistenceImage(n_bins=cfg.n_bins)

    def path_for_idx(self, idx: int) -> Path:
        return self.cache_dir / f"{idx:06d}.npy"

    def exists(self, idx: int) -> bool:
        return self.path_for_idx(idx).exists()

    def feature_dim(self) -> int:
        # PI returns (n_samples, n_homology_dims, n_bins, n_bins)
        return (self.cfg.max_homology_dim + 1) * self.cfg.n_bins * self.cfg.n_bins

    def compute_vec(self, coords: np.ndarray) -> np.ndarray:
        """
        coords: (N, 3) numpy array
        returns: (D,) float32 vector
        """
        X = coords[None, :, :]  # (1, N, 3)

        diagrams = self.vr.fit_transform(X) # persistence diagrams
        imgs = self.pi.fit_transform(diagrams) # (1, dims, bins, bins)

        vec = imgs.reshape(1, -1)[0].astype(np.float32)
        return vec

    def build_for_dataset(self, dataset) -> None:
        """
        dataset: torch_geometric.datasets.QM9 (not split)
        We cache by global index so subsets can reuse the same cache.
        """
        for idx in tqdm(range(len(dataset)), desc="Building TDA cache"):
            p = self.path_for_idx(idx)
            if p.exists():
                continue
            data = dataset[idx]
            coords = data.pos.detach().cpu().numpy()
            vec = self.compute_vec(coords)
            np.save(p, vec)
