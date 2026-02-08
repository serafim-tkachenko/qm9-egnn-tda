from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.data.qm9_data import load_qm9
from src.data.tda_features import TDACache, TDAConfig


def _load_vec(path: Path) -> np.ndarray:
    v = np.load(path)
    if v.shape != (130,):
        raise ValueError(f"Expected (130,), got {v.shape} for {path}")
    return v


def main():
    out_dir = Path(os.getenv("FIGURES_DIR", "figures"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Same config as cache build
    tda_cfg = TDAConfig(
        cache_dir=os.getenv("TDA_CACHE_DIR", "artifacts/tda_cache"),
        betti_bins=64,
        max_homology_dim=1,
        n_jobs=-1,
    )
    cache = TDACache(tda_cfg)
    cache_dir = Path(tda_cfg.cache_dir)

    files = sorted(cache_dir.glob("*.npy"))
    n_total = len(files)
    print("TDA cache:", cache_dir)
    print("Num cached vectors:", n_total)
    assert n_total > 0, "No cached .npy vectors found"

    # Dataset-level aggregation on a sample
    SAMPLE = min(5000, n_total)
    sample_files = random.sample(files, SAMPLE)

    h0_curves = []
    h1_curves = []
    h1_max = []
    h1_nonzero = 0

    for p in sample_files:
        v = _load_vec(p)
        h0 = v[0:64]
        h1 = v[64:128]
        h0_curves.append(h0)
        h1_curves.append(h1)

        mx = float(h1.max())
        h1_max.append(mx)
        if np.count_nonzero(h1) > 0:
            h1_nonzero += 1

    h0_curves = np.stack(h0_curves)
    h1_curves = np.stack(h1_curves)
    h1_max = np.array(h1_max)

    nonzero_rate = h1_nonzero / SAMPLE
    print(f"H1 nonzero rate (sample={SAMPLE}): {nonzero_rate:.4%}")

    # Save rate text
    (out_dir / "tda_h1_nonzero_rate.txt").write_text(
        f"H1 nonzero rate (sample={SAMPLE}): {nonzero_rate:.4%}\n"
    )

    # 1) Histogram of max(H1)
    plt.figure()
    plt.hist(h1_max, bins=30)
    plt.xlabel("max(H1 Betti curve)")
    plt.ylabel("count (sampled molecules)")
    plt.title("Distribution of loop signal strength (H1)")
    plt.savefig(out_dir / "tda_h1_max_hist.png", bbox_inches="tight")
    plt.close()

    # 2) Mean ± std Betti curves
    xs = np.arange(64)

    h0_mean = h0_curves.mean(axis=0)
    h0_std = h0_curves.std(axis=0)

    h1_mean = h1_curves.mean(axis=0)
    h1_std = h1_curves.std(axis=0)

    plt.figure()
    plt.plot(xs, h0_mean, label="H0 mean")
    plt.fill_between(xs, h0_mean - h0_std, h0_mean + h0_std, alpha=0.2)

    plt.plot(xs, h1_mean, label="H1 mean")
    plt.fill_between(xs, h1_mean - h1_std, h1_mean + h1_std, alpha=0.2)

    plt.xlabel("filtration bin")
    plt.ylabel("Betti number")
    plt.title(f"Betti curves: mean ± std (sample={SAMPLE})")
    plt.legend()
    plt.savefig(out_dir / "tda_betti_mean_std.png", bbox_inches="tight")
    plt.close()

    # Representative molecules: low / mid / high H1
    # choose from the sample for speed
    sorted_idx = np.argsort(h1_max)
    low_i = sorted_idx[0]
    mid_i = sorted_idx[len(sorted_idx) // 2]
    high_i = sorted_idx[-1]

    chosen = [
        ("low H1", sample_files[low_i]),
        ("median H1", sample_files[mid_i]),
        ("high H1", sample_files[high_i]),
    ]

    # Load QM9 for point clouds (only for 3 idxs)
    ds = load_qm9(os.getenv("QM9_ROOT", "data/qm9"))

    def file_to_idx(p: Path) -> int:
        return int(p.stem)

    plt.figure(figsize=(12, 8))

    for k, (name, p) in enumerate(chosen):
        idx = file_to_idx(p)
        v = _load_vec(p)
        h0 = v[:64]
        h1 = v[64:128]

        pos = ds[idx].pos.cpu().numpy()
        pos = pos - pos.mean(axis=0, keepdims=True)

        # point cloud
        ax1 = plt.subplot(3, 2, 2 * k + 1)
        ax1.scatter(pos[:, 0], pos[:, 1])
        ax1.set_title(f"{name} (idx={idx}): point cloud (x-y)")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        # betti curves
        ax2 = plt.subplot(3, 2, 2 * k + 2)
        ax2.plot(xs, h0, label="H0")
        ax2.plot(xs, h1, label="H1")
        ax2.set_title(f"{name} (idx={idx}): Betti curves")
        ax2.set_xlabel("filtration bin")
        ax2.set_ylabel("Betti number")
        ax2.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "tda_examples.png", bbox_inches="tight")
    plt.close()

    print("Saved:")
    print(" ", out_dir / "tda_h1_max_hist.png")
    print(" ", out_dir / "tda_betti_mean_std.png")
    print(" ", out_dir / "tda_examples.png")
    print(" ", out_dir / "tda_h1_nonzero_rate.txt")


if __name__ == "__main__":
    main()
