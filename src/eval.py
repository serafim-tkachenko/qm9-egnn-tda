from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.data.qm9_data import load_qm9, make_splits
from src.data.indexed_dataset import IndexedDataset
from src.data.collate import qm9_dense_collate
from src.data.tda_features import TDACache, TDAConfig
from src.models.egnn_gap import EGNNGapRegressor
from src.models.fusion_gap import EGNNTDARegressor
from src.utils.metrics import mae
from src.utils.seed import set_seed
from src.utils.io import save_json, save_csv, ensure_dir


@dataclass
class CompareEvalConfig:
    seed: int = 42
    root: str = "data/qm9" # overridden by QM9_ROOT env var
    batch_size: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # checkpoints (can be overridden via env vars)
    baseline_ckpt: str = "artifacts/checkpoints/best_egnn.pt"
    fusion_ckpt: str = "artifacts/checkpoints/best_fusion.pt"

    # TDA cache (used only for fusion)
    tda_cache_dir: str = "artifacts/tda_cache"
    tda_bins: int = 16
    tda_max_dim: int = 1

    out_dir: str = "results"
    fig_dir: str = "figures"

    noise_sigmas: Tuple[float, ...] = (0.0, 0.01, 0.05, 0.1)


def _env_override(cfg: CompareEvalConfig) -> CompareEvalConfig:
    # CKPT_DIR provides default locations for both checkpoints
    ckpt_dir = os.getenv("CKPT_DIR")
    if ckpt_dir:
        cfg.baseline_ckpt = str(Path(ckpt_dir) / "best_egnn.pt")
        cfg.fusion_ckpt = str(Path(ckpt_dir) / "best_fusion.pt")

    baseline_ckpt = os.getenv("BASELINE_CKPT")
    if baseline_ckpt:
        cfg.baseline_ckpt = baseline_ckpt

    fusion_ckpt = os.getenv("FUSION_CKPT")
    if fusion_ckpt:
        cfg.fusion_ckpt = fusion_ckpt

    tda_dir = os.getenv("TDA_CACHE_DIR")
    if tda_dir:
        cfg.tda_cache_dir = tda_dir

    out_dir = os.getenv("RESULTS_DIR")
    if out_dir:
        cfg.out_dir = out_dir

    fig_dir = os.getenv("FIGURES_DIR")
    if fig_dir:
        cfg.fig_dir = fig_dir

    return cfg


def load_tda_batch(idxs: torch.LongTensor, cache: TDACache) -> torch.FloatTensor:
    vecs = []
    for idx in idxs.tolist():
        p = cache.path_for_idx(int(idx))
        if not p.exists():
            raise FileNotFoundError(f"Missing TDA cache file: {p}")
        vecs.append(np.load(p))
    arr = np.stack(vecs, axis=0)
    return torch.from_numpy(arr).float()


@torch.no_grad()
def eval_baseline(model: torch.nn.Module, loader: DataLoader, device: str, noise_sigma: float) -> float:
    model.eval()
    mae_sum, n_sum = 0.0, 0
    for batch in loader:
        z = batch.z.to(device)
        pos = batch.pos.to(device)
        mask = batch.mask.to(device)
        y = batch.y.to(device)

        if noise_sigma > 0:
            pos = pos + torch.randn_like(pos) * noise_sigma

        pred = model(z, pos, mask)
        m = float(mae(pred, y).item())
        bs = z.size(0)
        mae_sum += m * bs
        n_sum += bs
    return mae_sum / max(n_sum, 1)


@torch.no_grad()
def eval_fusion(model: torch.nn.Module, loader: DataLoader, device: str, noise_sigma: float, tda_cache: TDACache) -> float:
    model.eval()
    mae_sum, n_sum = 0.0, 0
    for batch in loader:
        z = batch.z.to(device)
        pos = batch.pos.to(device)
        mask = batch.mask.to(device)
        y = batch.y.to(device)

        if noise_sigma > 0:
            pos = pos + torch.randn_like(pos) * noise_sigma

        tda_vec = load_tda_batch(batch.idx, tda_cache).to(device)

        pred = model(z, pos, mask, tda_vec)
        m = float(mae(pred, y).item())
        bs = z.size(0)
        mae_sum += m * bs
        n_sum += bs
    return mae_sum / max(n_sum, 1)


def plot_compare_robustness(rows: List[Dict[str, float]], fig_path: str) -> None:
    # rows contain sigma, baseline_test_mae, fusion_test_mae
    sigmas = [r["noise_sigma"] for r in rows]
    base = [r["baseline_test_mae"] for r in rows]
    fus = [r["fusion_test_mae"] for r in rows]

    plt.figure()
    plt.plot(sigmas, base, marker="o", label="EGNN")
    plt.plot(sigmas, fus, marker="o", label="EGNN+TDA")
    plt.xlabel("Gaussian noise sigma (added to coordinates)")
    plt.ylabel("Test MAE (HOMO-LUMO gap)")
    plt.title("Robustness comparison")
    plt.legend()
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


def run():
    cfg = _env_override(CompareEvalConfig())
    set_seed(cfg.seed)

    ensure_dir(cfg.out_dir)
    ensure_dir(cfg.fig_dir)

    ds = load_qm9(cfg.root)
    split = make_splits(ds, seed=cfg.seed)

    val_ds = IndexedDataset(split.val)
    test_ds = IndexedDataset(split.test)

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=qm9_dense_collate)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=qm9_dense_collate)

    # Baseline
    baseline = EGNNGapRegressor().to(cfg.device)
    if not Path(cfg.baseline_ckpt).exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {cfg.baseline_ckpt}")
    baseline.load_state_dict(torch.load(cfg.baseline_ckpt, map_location=cfg.device))

    baseline_val = eval_baseline(baseline, val_loader, cfg.device, noise_sigma=0.0)
    baseline_test = eval_baseline(baseline, test_loader, cfg.device, noise_sigma=0.0)

    # Fusion
    tda_cfg = TDAConfig(cache_dir=cfg.tda_cache_dir, n_bins=cfg.tda_bins, max_homology_dim=cfg.tda_max_dim, n_jobs=-1)
    tda_cache = TDACache(tda_cfg)
    tda_dim = tda_cache.feature_dim()

    fusion = EGNNTDARegressor(tda_dim=tda_dim).to(cfg.device)
    if not Path(cfg.fusion_ckpt).exists():
        raise FileNotFoundError(f"Fusion checkpoint not found: {cfg.fusion_ckpt}")
    fusion.load_state_dict(torch.load(cfg.fusion_ckpt, map_location=cfg.device))

    fusion_val = eval_fusion(fusion, val_loader, cfg.device, noise_sigma=0.0, tda_cache=tda_cache)
    fusion_test = eval_fusion(fusion, test_loader, cfg.device, noise_sigma=0.0, tda_cache=tda_cache)

    # Robustness curves
    rob_rows = []
    for s in cfg.noise_sigmas:
        s = float(s)
        b = eval_baseline(baseline, test_loader, cfg.device, noise_sigma=s)
        f = eval_fusion(fusion, test_loader, cfg.device, noise_sigma=s, tda_cache=tda_cache)
        rob_rows.append({
            "noise_sigma": s,
            "baseline_test_mae": float(b),
            "fusion_test_mae": float(f),
        })

    # Save comparison artifacts
    metrics = {
        "baseline": {
            "model": "EGNN",
            "ckpt": cfg.baseline_ckpt,
            "val_mae": float(baseline_val),
            "test_mae": float(baseline_test),
        },
        "fusion": {
            "model": "EGNN+TDA",
            "ckpt": cfg.fusion_ckpt,
            "tda_cache_dir": cfg.tda_cache_dir,
            "tda_dim": int(tda_dim),
            "val_mae": float(fusion_val),
            "test_mae": float(fusion_test),
        },
        "noise_sigmas": list(cfg.noise_sigmas),
    }
    save_json(Path(cfg.out_dir) / "compare_metrics.json", metrics)
    save_csv(Path(cfg.out_dir) / "compare_robustness.csv", rob_rows)

    # Table for README
    table_rows = [
        {"model": "EGNN", "val_mae": float(baseline_val), "test_mae": float(baseline_test)},
        {"model": "EGNN+TDA", "val_mae": float(fusion_val), "test_mae": float(fusion_test)},
    ]
    save_csv(Path(cfg.out_dir) / "compare_table.csv", table_rows)

    plot_compare_robustness(rob_rows, str(Path(cfg.fig_dir) / "compare_robustness.png"))

    print("Comparison results:")
    print(f"  EGNN      val MAE={baseline_val:.6f} test MAE={baseline_test:.6f}")
    print(f"  EGNN+TDA  val MAE={fusion_val:.6f} test MAE={fusion_test:.6f}")
    print("Robustness (test):")
    for r in rob_rows:
        print(f"  sigma={r['noise_sigma']:.3f} -> EGNN={r['baseline_test_mae']:.6f}  EGNN+TDA={r['fusion_test_mae']:.6f}")

    print("Saved:")
    print(" ", Path(cfg.out_dir) / "compare_metrics.json")
    print(" ", Path(cfg.out_dir) / "compare_table.csv")
    print(" ", Path(cfg.out_dir) / "compare_robustness.csv")
    print(" ", Path(cfg.fig_dir) / "compare_robustness.png")


if __name__ == "__main__":
    run()
