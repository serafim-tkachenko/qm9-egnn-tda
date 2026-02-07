from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.qm9_data import load_qm9, make_splits
from src.data.indexed_dataset import IndexedDataset
from src.data.collate import qm9_dense_collate
from src.data.tda_features import TDACache, TDAConfig
from src.models.fusion_gap import EGNNTDAFiLMRegressor
from src.utils.seed import set_seed
from src.utils.metrics import mae
from src.utils.exp_logging import History, save_history_json, save_summary_csv, plot_loss, plot_mae


@dataclass
class TrainFusionConfig:
    seed: int = 42
    root: str = "data/qm9" # overridden by QM9_ROOT env var
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # TDA cache settings
    tda_cache_dir: str = "artifacts/tda_cache"
    tda_bins: int = 64
    tda_max_dim: int = 1

    ckpt_path: str = "artifacts/checkpoints/best_fusion.pt"
    results_dir: str = "results"
    figures_dir: str = "figures"


def _env_override(cfg: TrainFusionConfig) -> TrainFusionConfig:
    tda_dir = os.getenv("TDA_CACHE_DIR")
    if tda_dir:
        cfg.tda_cache_dir = tda_dir

    ckpt_dir = os.getenv("CKPT_DIR")
    if ckpt_dir:
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        cfg.ckpt_path = str(Path(ckpt_dir) / "best_fusion.pt")

    res_dir = os.getenv("RESULTS_DIR")
    if res_dir:
        cfg.results_dir = res_dir

    fig_dir = os.getenv("FIGURES_DIR")
    if fig_dir:
        cfg.figures_dir = fig_dir

    return cfg


def load_tda_batch(idxs: torch.LongTensor, cache: TDACache) -> torch.FloatTensor:
    vecs = []
    for idx in idxs.tolist():
        p = cache.path_for_idx(int(idx))
        if not p.exists():
            raise FileNotFoundError(
                f"Missing TDA cache file for idx={idx}. Expected: {p}\n"
                f"Build cache first"
            )
        vecs.append(np.load(p))
    arr = np.stack(vecs, axis=0)  # (B, D)
    return torch.from_numpy(arr).float()


def run():
    cfg = _env_override(TrainFusionConfig())
    set_seed(cfg.seed)

    ds = load_qm9(cfg.root)
    split = make_splits(ds, seed=cfg.seed)

    train_ds = IndexedDataset(split.train)
    val_ds = IndexedDataset(split.val)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=qm9_dense_collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=qm9_dense_collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    tda_cfg = TDAConfig(
        cache_dir=cfg.tda_cache_dir,
        betti_bins=cfg.tda_bins,
        max_homology_dim=cfg.tda_max_dim,
        n_jobs=-1,
    )
    tda_cache = TDACache(tda_cfg)
    tda_dim = tda_cache.feature_dim()

    model = EGNNTDAFiLMRegressor(tda_dim=tda_dim).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = torch.nn.MSELoss()

    os.makedirs(Path(cfg.ckpt_path).parent, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.figures_dir, exist_ok=True)

    best_val = float("inf")
    history = History(epochs=[], train_loss=[], train_mae=[], val_mae=[])

    for epoch in range(1, cfg.epochs + 1):
        # train
        model.train()
        train_mae_sum, train_loss_sum, train_n = 0.0, 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [train_fusion]")
        for batch in pbar:
            z = batch.z.to(cfg.device)
            pos = batch.pos.to(cfg.device)
            mask = batch.mask.to(cfg.device)
            y = batch.y.to(cfg.device)

            tda_vec = load_tda_batch(batch.idx, tda_cache).to(cfg.device)

            pred = model(z, pos, mask, tda_vec)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = z.size(0)
            train_loss_sum += float(loss.item()) * bs

            with torch.no_grad():
                m = float(mae(pred, y).item())
                
            if epoch == 1 and train_n == 0:
                print("TDA batch:", tda_vec.shape, "mean/std:", float(tda_vec.mean()), float(tda_vec.std()))
                assert tda_vec.std() > 0, "TDA features look degenerate (std=0)"
        
            train_mae_sum += m * bs
            train_n += bs

            pbar.set_postfix(train_mae=m, train_loss=float(loss.item()))

        train_mae_epoch = train_mae_sum / max(train_n, 1)
        train_loss_epoch = train_loss_sum / max(train_n, 1)

        # val
        model.eval()
        val_mae_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                z = batch.z.to(cfg.device)
                pos = batch.pos.to(cfg.device)
                mask = batch.mask.to(cfg.device)
                y = batch.y.to(cfg.device)

                tda_vec = load_tda_batch(batch.idx, tda_cache).to(cfg.device)

                pred = model(z, pos, mask, tda_vec)
                m = float(mae(pred, y).item())

                bs = z.size(0)
                val_mae_sum += m * bs
                val_n += bs

        val_mae_epoch = val_mae_sum / max(val_n, 1)

        history.epochs.append(epoch)
        history.train_loss.append(float(train_loss_epoch))
        history.train_mae.append(float(train_mae_epoch))
        history.val_mae.append(float(val_mae_epoch))

        print(f"Epoch {epoch}: train_loss={train_loss_epoch:.6f} train_MAE={train_mae_epoch:.4f} val_MAE={val_mae_epoch:.4f}")

        if val_mae_epoch < best_val:
            best_val = val_mae_epoch
            torch.save(model.state_dict(), cfg.ckpt_path)
            print("Saved best fusion checkpoint:", cfg.ckpt_path)

        # incremental history
        save_history_json(
            Path(cfg.results_dir) / "fusion_train_history.json",
            history,
            extra={
                "model": "EGNN+TDA (FiLM)",
                "seed": cfg.seed,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "epochs": cfg.epochs,
                "best_val_mae": float(best_val),
                "ckpt_path": cfg.ckpt_path,
                "tda_cache_dir": cfg.tda_cache_dir,
                "tda_bins": cfg.tda_bins,
                "tda_max_dim": cfg.tda_max_dim,
                "tda_dim": int(tda_dim),
                "qm9_root_env": os.getenv("QM9_ROOT", ""),
            },
        )

    plot_loss(history, Path(cfg.figures_dir) / "fusion_train_loss.png", title="EGNN+TDA: Training Loss")
    plot_mae(history, Path(cfg.figures_dir) / "fusion_mae_curves.png", title="EGNN+TDA: MAE Curves")

    save_summary_csv(
        Path(cfg.results_dir) / "fusion_summary.csv",
        [{
            "model": "EGNN+TDA",
            "best_val_mae": float(best_val),
            "final_train_mae": float(history.train_mae[-1]),
            "final_val_mae": float(history.val_mae[-1]),
            "final_train_loss": float(history.train_loss[-1]),
            "ckpt_path": cfg.ckpt_path,
            "tda_dim": int(tda_dim),
            "tda_cache_dir": cfg.tda_cache_dir,
        }],
    )

    print("Fusion training completed -> best val MAE:", best_val)
    print("Saved:")
    print(" ", Path(cfg.results_dir) / "fusion_train_history.json")
    print(" ", Path(cfg.results_dir) / "fusion_summary.csv")
    print(" ", Path(cfg.figures_dir) / "fusion_train_loss.png")
    print(" ", Path(cfg.figures_dir) / "fusion_mae_curves.png")


if __name__ == "__main__":
    run()
