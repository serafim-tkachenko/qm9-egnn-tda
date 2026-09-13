"""Compare sigma-zero pilot predictions with the original clean evaluator."""
import argparse
import csv
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from src.data.collate import qm9_dense_collate
from src.data.indexed_dataset import IndexedDataset
from src.data.tda_features import TDACache, TDAConfig
from src.eval import eval_baseline, eval_fusion
from src.eval_paired import load_models, load_processed
from src.validation import file_hash, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", required=True)
    p.add_argument("--original-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--legacy-cache-dir", help="Optional staged copy of legacy descriptors")
    args = p.parse_args()
    start = time.perf_counter()
    out, pilot, root = Path(args.out), Path(args.pilot), Path(args.original_root)
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((pilot / "manifest.json").read_text())
    complete = json.loads((pilot / "complete.json").read_text())
    if file_hash(pilot / "predictions.csv") != complete["predictions_sha256"]:
        raise ValueError("Pilot predictions hash mismatch")
    data_path = root / "data/qm9/processed/data_v3.pt"
    if file_hash(data_path) != manifest["dataset_sha256"]:
        raise ValueError("Dataset differs from pilot")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    ds = load_processed(data_path)
    loader = DataLoader(IndexedDataset(Subset(ds, manifest["subset_ids"])),
                        batch_size=manifest["arguments"]["batch_size"],
                        collate_fn=qm9_dense_collate, shuffle=False)
    models = load_models(manifest["checkpoint_provenance"], args.device)
    cache = TDACache(TDAConfig(cache_dir=str(args.legacy_cache_dir or root / "tda_cache"), n_jobs=1))
    legacy = [eval_baseline(models[0], loader, args.device, 0.),
              eval_fusion(models[1], loader, args.device, 0., cache)]
    with (pilot / "predictions.csv").open() as f:
        records = [r for r in csv.DictReader(f) if float(r["sigma"]) == 0.]
    if not records:
        raise ValueError("No clean pilot predictions")
    current = [sum(float(r[name + "_abs_error"]) for r in records) / len(records)
               for name in ("egnn", "fusion_clean")]
    result = {"n_molecules": len(manifest["subset_ids"]), "units": "eV",
              "legacy_clean_mae": dict(zip(("egnn", "fusion"), legacy)),
              "paired_sigma_zero_mae": dict(zip(("egnn", "fusion"), current)),
              "absolute_differences": [abs(a-b) for a, b in zip(legacy, current)],
              "absolute_tolerance_ev": 2e-6,
              "full_original_test_metric_reproduced": False,
              "seconds": time.perf_counter() - start}
    result["passed"] = max(result["absolute_differences"]) <= 2e-6
    write_json(out, result)
    if not result["passed"]:
        raise ValueError("Clean evaluator discrepancy exceeds tolerance")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
