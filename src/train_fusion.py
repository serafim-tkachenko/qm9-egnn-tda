from __future__ import annotations
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.qm9_data import load_qm9, make_splits
from src.data.indexed_dataset import IndexedDataset
from src.data.collate import qm9_dense_collate
from src.data.tda_features import TDACache, TDAConfig
from src.models.fusion_gap import EGNNTDARegressor
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
    tda_cache_dir: str = "artifacts/tda_cache"
    tda_bins: int = 16


def load_tda_batch(idxs: torch.LongTensor, cache: TDACache) -> torch.FloatTensor:
    vecs = []
    for idx in idxs.tolist():
        path = cache.path_for_idx(int(idx))
        if not path.exists():
            raise FileNotFoundError(f"Missing TDA cache file: {path}")
        vecs.append(np.load(path))
    arr = np.stack(vecs, axis=0)  # (B, D)
    return torch.from_numpy(arr).float()


def run():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    ds = load_qm9(cfg.root)
    split = make_splits(ds, seed=cfg.seed)

    train_ds = IndexedDataset(split.train)
    val_ds = IndexedDataset(split.val)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=qm9_dense_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=qm9_dense_collate)

    tda_cfg = TDAConfig(cache_dir=cfg.tda_cache_dir, n_bins=cfg.tda_bins, max_homology_dim=1)
    tda_cache = TDACache(tda_cfg)
    tda_dim = tda_cache.feature_dim()

    model = EGNNTDARegressor(tda_dim=tda_dim).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = torch.nn.MSELoss()

    os.makedirs("artifacts/checkpoints", exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_mae_sum, train_n = 0.0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [train_fusion]"):
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

                tda_vec = load_tda_batch(batch.idx, tda_cache).to(cfg.device)

                pred = model(z, pos, mask, tda_vec)
                m = mae(pred, y).item()
                val_mae_sum += m * z.size(0)
                val_n += z.size(0)

        val_mae_epoch = val_mae_sum / max(val_n, 1)
        print(f"Epoch {epoch}: train_MAE={train_mae_epoch:.4f} val_MAE={val_mae_epoch:.4f}")

        if val_mae_epoch < best_val:
            best_val = val_mae_epoch
            torch.save(model.state_dict(), "artifacts/checkpoints/best_fusion.pt")
            print("Saved best fusion checkpoint")

    print("Training completed -> best val MAE:", best_val)


if __name__ == "__main__":
    run()
