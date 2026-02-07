from __future__ import annotations
from typing import Any, Tuple

class IndexedDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i: int) -> Tuple[int, Any]:
        # return (global_idx, Data)
        if hasattr(self.dataset, "indices"):
            global_idx = int(self.dataset.indices[i])
        else:
            global_idx = int(i)
        return global_idx, self.dataset[i]
