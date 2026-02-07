from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from gtda.utils.validation import DataDimensionalityWarning

from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import BettiCurve, PersistenceEntropy

import warnings


@dataclass
class TDAConfig:
    cache_dir: str = "artifacts/tda_cache"
    max_homology_dim: int = 1 # H0 and H1
    betti_bins: int = 64 # resolution of Betti curves
    n_jobs: int = -1


class TDACache:
    """
    Persistent homology on atomic coordinate point clouds (Vietoris–Rips),
    vectorized via:
      - Betti curves (per homology dimension)
      - Persistence entropy (per homology dimension)
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

        self.betti = BettiCurve(n_bins=cfg.betti_bins)
        self.entropy = PersistenceEntropy()

    def path_for_idx(self, idx: int) -> Path:
        return self.cache_dir / f"{idx:06d}.npy"

    def feature_dim(self) -> int:
        # BettiCurve gives (n_samples, n_homology_dims, n_bins)
        betti_dim = (self.cfg.max_homology_dim + 1) * self.cfg.betti_bins
        # PersistenceEntropy gives (n_samples, n_homology_dims)
        ent_dim = (self.cfg.max_homology_dim + 1)
        return betti_dim + ent_dim

    def compute_vec(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Expected coords shape (N,3), got {coords.shape}")

        # Center
        coords = coords - coords.mean(axis=0, keepdims=True)

        # Scale to unit diameter for stability across molecules
        diffs = coords[:, None, :] - coords[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=-1))
        diameter = float(dists.max())
        if diameter > 0:
            coords = coords / diameter

        X = coords[None, :, :]  # (1, N, 3)
        
        # Supress warnings for (N,3) to make the output cleaner
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DataDimensionalityWarning)
            diagrams = self.vr.fit_transform(X)  # (1, n_points, 3) with (birth, death, dim)
            betti = self.betti.fit_transform(diagrams) # (1, dims, bins)
            ent = self.entropy.fit_transform(diagrams) # (1, dims)

        vec = np.concatenate([betti.reshape(1, -1), ent.reshape(1, -1)], axis=1)[0].astype(np.float32)

        # sanity
        if np.count_nonzero(vec) == 0:
            raise ValueError("TDA vector is all zeros — check diagrams / pipeline")
        return vec

    def build_for_dataset(self, dataset) -> None:
        for idx in tqdm(range(len(dataset)), desc="Building TDA cache"):
            p = self.path_for_idx(idx)
            if p.exists():
                continue
            coords = dataset[idx].pos.detach().cpu().numpy()
            vec = self.compute_vec(coords)
            np.save(p, vec)
