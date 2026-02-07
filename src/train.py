from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.qm9_data import load_qm9, make_splits
from src.data.indexed_dataset import IndexedDataset
from src.data.collate import qm9_dense_collate
from src.models.egnn_gap import EGNNGapRegressor
from src.utils.seed import set_seed
from src.utils.metrics import mae
from src.utils.exp_logging import History, save_history_json, save_summary_csv, plot_loss, plot_mae


@dataclass
class TrainConfig:
    seed: int = 42
    root: str = "data/qm9"
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path: str = "artifacts/checkpoints/best_egnn.pt"
    results_dir: str = "results"
    figures_dir: str = "figures"

def run():
    cfg = TrainConfig()
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

    model = EGNNGapRegressor().to(cfg.device)
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

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [train]")
        for batch in pbar:
            z = batch.z.to(cfg.device)
            pos = batch.pos.to(cfg.device)
            mask = batch.mask.to(cfg.device)
            y = batch.y.to(cfg.device)

            pred = model(z, pos, mask)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = z.size(0)
            train_loss_sum += float(loss.item()) * bs

            with torch.no_grad():
                m = float(mae(pred, y).item())
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

                pred = model(z, pos, mask)
                m = float(mae(pred, y).item())

                bs = z.size(0)
                val_mae_sum += m * bs
                val_n += bs

        val_mae_epoch = val_mae_sum / max(val_n, 1)

        # log
        history.epochs.append(epoch)
        history.train_loss.append(float(train_loss_epoch))
        history.train_mae.append(float(train_mae_epoch))
        history.val_mae.append(float(val_mae_epoch))

        print(f"Epoch {epoch}: train_loss={train_loss_epoch:.6f} train_MAE={train_mae_epoch:.4f} val_MAE={val_mae_epoch:.4f}")

        # checkpoint
        if val_mae_epoch < best_val:
            best_val = val_mae_epoch
            torch.save(model.state_dict(), cfg.ckpt_path)
            print("Saved best checkpoint:", cfg.ckpt_path)

        # save incremental history each epoch
        save_history_json(
            Path(cfg.results_dir) / "baseline_train_history.json",
            history,
            extra={
                "model": "EGNN",
                "seed": cfg.seed,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "epochs": cfg.epochs,
                "best_val_mae": float(best_val),
                "ckpt_path": cfg.ckpt_path,
            },
        )

    # plots
    plot_loss(history, Path(cfg.figures_dir) / "baseline_train_loss.png", title="EGNN Baseline: Training Loss")
    plot_mae(history, Path(cfg.figures_dir) / "baseline_mae_curves.png", title="EGNN Baseline: MAE Curves")

    summary_rows = [{
        "model": "EGNN",
        "best_val_mae": float(best_val),
        "final_train_mae": float(history.train_mae[-1]),
        "final_val_mae": float(history.val_mae[-1]),
        "final_train_loss": float(history.train_loss[-1]),
        "ckpt_path": cfg.ckpt_path,
    }]
    save_summary_csv(Path(cfg.results_dir) / "baseline_summary.csv", summary_rows)

    print("Training completed -> best val MAE:", best_val)
    print("Saved:")
    print(" ", Path(cfg.results_dir) / "baseline_train_history.json")
    print(" ", Path(cfg.results_dir) / "baseline_summary.csv")
    print(" ", Path(cfg.figures_dir) / "baseline_train_loss.png")
    print(" ", Path(cfg.figures_dir) / "baseline_mae_curves.png")


if __name__ == "__main__":
    run()
