from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import torch
from torch.utils.data import random_split
from torch_geometric.datasets import QM9

# PyG QM9 targets
# https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.datasets.QM9.html
# HOMO index=2, LUMO index=3, GAP index=4
GAP_TARGET_INDEX = 4


@dataclass
class Split:
    train: torch.utils.data.Dataset
    val: torch.utils.data.Dataset
    test: torch.utils.data.Dataset


def load_qm9(root: str = "data/qm9") -> QM9:
    # allow override via env var for Colab/Drive persistence
    root = os.getenv("QM9_ROOT", root)
    Path(root).mkdir(parents=True, exist_ok=True)
    return QM9(root)


def make_splits(ds: QM9, seed: int = 42, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Split:
    n = len(ds)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    train, val, test = random_split(ds, [n_train, n_val, n_test], generator=g)
    return Split(train=train, val=val, test=test)
