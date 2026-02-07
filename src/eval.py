from __future__ import annotations

import math
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
from src.models.egnn_gap import EGNNGapRegressor
from src.utils.metrics import mae
from src.utils.seed import set_seed
from src.utils.io import save_json, save_csv, ensure_dir


@dataclass
class EvalConfig:
    seed: int = 42
    root: str = "data/qm9"
    batch_size: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path: str = "artifacts/checkpoints/best_egnn.pt"
    out_dir: str = "results"
    fig_dir: str = "figures"
    noise_sigmas: Tuple[float, ...] = (0.0, 0.01, 0.05, 0.1)


@torch.no_grad()
def eval_loader(model: torch.nn.Module, loader: DataLoader, device: str, noise_sigma: float = 0.0) -> float:
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
        m = mae(pred, y).item()

        bs = z.size(0)
        mae_sum += m * bs
        n_sum += bs

    return mae_sum / max(n_sum, 1)


def plot_robustness(rows: List[Dict[str, float]], fig_path: str) -> None:
    sigmas = [r["noise_sigma"] for r in rows]
    maes = [r["test_mae"] for r in rows]

    plt.figure()
    plt.plot(sigmas, maes, marker="o")
    plt.xlabel("Gaussian noise sigma (added to coordinates)")
    plt.ylabel("Test MAE (HOMO-LUMO gap)")
    plt.title("Robustness of EGNN to coordinate noise")
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


def run():
    cfg = EvalConfig()
    set_seed(cfg.seed)

    ds = load_qm9(cfg.root)
    split = make_splits(ds, seed=cfg.seed)

    val_ds = IndexedDataset(split.val)
    test_ds = IndexedDataset(split.test)

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=qm9_dense_collate)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=qm9_dense_collate)

    model = EGNNGapRegressor().to(cfg.device)

    ckpt = Path(cfg.ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.ckpt_path}")

    state = torch.load(cfg.ckpt_path, map_location=cfg.device)
    model.load_state_dict(state)

    # Clean eval (no noise)
    val_mae = eval_loader(model, val_loader, cfg.device, noise_sigma=0.0)
    test_mae = eval_loader(model, test_loader, cfg.device, noise_sigma=0.0)

    ensure_dir(cfg.out_dir)
    ensure_dir(cfg.fig_dir)

    summary = {
        "model": "EGNN",
        "checkpoint": cfg.ckpt_path,
        "val_mae": float(val_mae),
        "test_mae": float(test_mae),
        "noise_sigmas": list(cfg.noise_sigmas),
    }
    save_json(Path(cfg.out_dir) / "baseline_metrics.json", summary)

    # Robustness curve on test set
    rob_rows = []
    for s in cfg.noise_sigmas:
        tm = eval_loader(model, test_loader, cfg.device, noise_sigma=float(s))
        rob_rows.append({"noise_sigma": float(s), "test_mae": float(tm)})

    save_csv(Path(cfg.out_dir) / "baseline_robustness.csv", rob_rows)
    plot_robustness(rob_rows, str(Path(cfg.fig_dir) / "baseline_robustness.png"))

    print("Baseline EGNN results:")
    print(f"  val MAE : {val_mae:.6f}")
    print(f"  test MAE: {test_mae:.6f}")
    print("Robustness (test):")
    for r in rob_rows:
        print(f"  sigma={r['noise_sigma']:.3f} -> test MAE={r['test_mae']:.6f}")
    print("Saved:")
    print(" ", Path(cfg.out_dir) / "baseline_metrics.json")
    print(" ", Path(cfg.out_dir) / "baseline_robustness.csv")
    print(" ", Path(cfg.fig_dir) / "baseline_robustness.png")


if __name__ == "__main__":
    run()
