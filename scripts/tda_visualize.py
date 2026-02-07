from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.data.qm9_data import load_qm9
from src.data.tda_features import TDACache, TDAConfig


def main():
    out_dir = Path(os.getenv("FIGURES_DIR", "figures"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_qm9()
    cfg = TDAConfig(
        cache_dir=os.getenv("TDA_CACHE_DIR", "artifacts/tda_cache"),
        betti_bins=64,
        max_homology_dim=1,
        n_jobs=-1,
    )
    tda = TDACache(cfg)

    idx = 0
    coords = ds[idx].pos.cpu().numpy()
    coords = coords - coords.mean(axis=0, keepdims=True)

    X = coords[None, :, :]
    diagrams = tda.vr.fit_transform(X)
    betti = tda.betti.fit_transform(diagrams)[0]  # (dims, bins)

    # plot point cloud projection + betti curves
    plt.figure()
    plt.scatter(coords[:, 0], coords[:, 1])
    plt.title(f"QM9 molecule #{idx}: point cloud (x-y projection)")
    plt.xlabel("x"); plt.ylabel("y")
    plt.savefig(out_dir / "tda_pointcloud_xy.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    xs = np.arange(cfg.betti_bins)
    plt.plot(xs, betti[0], label="H0 Betti curve")
    if betti.shape[0] > 1:
        plt.plot(xs, betti[1], label="H1 Betti curve")
    plt.title("Betti curves from Vietoris–Rips persistent homology")
    plt.xlabel("filtration bin")
    plt.ylabel("Betti number")
    plt.legend()
    plt.savefig(out_dir / "tda_betti_curves.png", bbox_inches="tight")
    plt.close()

    print("Saved:", out_dir / "tda_pointcloud_xy.png")
    print("Saved:", out_dir / "tda_betti_curves.png")


if __name__ == "__main__":
    main()
