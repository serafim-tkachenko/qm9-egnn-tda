from __future__ import annotations

import os
import numpy as np

from src.data.qm9_data import load_qm9
from src.data.tda_features import TDACache, TDAConfig


def main():
    # Read environment variables
    qm9_root = os.getenv("QM9_ROOT", "data/qm9")
    cache_dir = os.getenv("TDA_CACHE_DIR", "artifacts/tda_cache")

    print("QM9_ROOT:", qm9_root)
    print("TDA_CACHE_DIR:", cache_dir)

    # Load dataset
    ds = load_qm9(qm9_root)

    # Configure TDA
    cfg = TDAConfig(
        cache_dir=cache_dir,
        betti_bins=64,
        max_homology_dim=1,
    )
    tda = TDACache(cfg)

    print("TDA feature dim:", tda.feature_dim())

    # Smoke test on a few molecules - we need that as for some TDA runs I got only zeros
    print("Running TDA smoke test...")
    for i in [0, 1, 2]:
        v = tda.compute_vec(ds[i].pos.cpu().numpy())
        print(
            "smoke",
            i,
            v.shape,
            "mean=", float(v.mean()),
            "std=", float(v.std()),
            "nonzero=", int(np.count_nonzero(v)),
        )

    print("Starting full TDA cache build...")
    tda.build_for_dataset(ds)

    print("Done. Cached to:", cfg.cache_dir)


if __name__ == "__main__":
    main()
