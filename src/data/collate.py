from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch
from .qm9_data import GAP_TARGET_INDEX


@dataclass
class BatchDense:
    z: torch.LongTensor  # (B, Nmax)
    pos: torch.FloatTensor # (B, Nmax, 3)
    mask: torch.BoolTensor # (B, Nmax)
    y: torch.FloatTensor # (B,)
    idx: torch.LongTensor # (B,) indexes in the original QM9


def qm9_dense_collate(batch: List[Tuple[int, object]]) -> BatchDense:
    # batch: [(global_idx, Data), ...]
    B = len(batch)
    n_nodes = [int(data.z.size(0)) for _, data in batch]
    Nmax = max(n_nodes)

    z = torch.zeros((B, Nmax), dtype=torch.long)
    pos = torch.zeros((B, Nmax, 3), dtype=torch.float32)
    mask = torch.zeros((B, Nmax), dtype=torch.bool)
    y = torch.zeros((B,), dtype=torch.float32)
    idx = torch.zeros((B,), dtype=torch.long)

    for i, (gidx, data) in enumerate(batch):
        n = int(data.z.size(0))
        z[i, :n] = data.z
        pos[i, :n, :] = data.pos
        mask[i, :n] = True
        y[i] = data.y[GAP_TARGET_INDEX].float()
        idx[i] = gidx

    return BatchDense(z=z, pos=pos, mask=mask, y=y, idx=idx)