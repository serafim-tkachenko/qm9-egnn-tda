from __future__ import annotations
import os
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.qm9_data import load_qm9, make_splits
from src.data.indexed_dataset import IndexedDataset
from src.data.collate import qm9_dense_collate
from src.models.egnn_gap import EGNNGapRegressor
from src.utils.seed import set_seed
from src.utils.metrics import mae



@dataclass
class TrainConfig:
    seed: int = 42
    root: str = "data/qm9"
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def run():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    ds = load_qm9(cfg.root)
    split = make_splits(ds, seed=cfg.seed)

    train_ds = IndexedDataset(split.train)
    val_ds = IndexedDataset(split.val)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=qm9_dense_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=qm9_dense_collate)

    model = EGNNGapRegressor().to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = torch.nn.MSELoss()

    os.makedirs("artifacts/checkpoints", exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_mae_sum, train_n = 0.0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [train]"):
            z = batch.z.to(cfg.device)
            pos = batch.pos.to(cfg.device)
            mask = batch.mask.to(cfg.device)
            y = batch.y.to(cfg.device)

            pred = model(z, pos, mask)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                m = mae(pred, y).item()
            train_mae_sum += m * z.size(0)
            train_n += z.size(0)

        train_mae_epoch = train_mae_sum / max(train_n, 1)

        model.eval()
        val_mae_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                z = batch.z.to(cfg.device)
                pos = batch.pos.to(cfg.device)
                mask = batch.mask.to(cfg.device)
                y = batch.y.to(cfg.device)

                pred = model(z, pos, mask)
                m = mae(pred, y).item()
                val_mae_sum += m * z.size(0)
                val_n += z.size(0)

        val_mae_epoch = val_mae_sum / max(val_n, 1)
        print(f"Epoch {epoch}: train_MAE={train_mae_epoch:.4f} val_MAE={val_mae_epoch:.4f}")

        if val_mae_epoch < best_val:
            best_val = val_mae_epoch
            torch.save(model.state_dict(), "artifacts/checkpoints/best_egnn.pt")
            print("Saved best checkpoint")

    print("Training has been completed -> best val MAE:", best_val)


if __name__ == "__main__":
    run()
