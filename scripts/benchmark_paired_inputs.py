"""Benchmark disposable optimizer steps on actual archived pilot molecules.

No trained model is saved. Timings describe one fixed batch shape, not a measured
full epoch, cache I/O, convergence or the complete cost of a multi-seed experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch_geometric.data import Data

from scripts.prepare_paired import CHECKPOINTS
from src.data.collate import qm9_dense_collate
from src.models.egnn_gap import EGNNGapRegressor
from src.models.fusion_gap import EGNNTDAFiLMRegressor
from src.validation import checked_state_dict, file_hash, versions, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", required=True)
    p.add_argument("--checkpoints", default="artifacts/checkpoints")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.batch_size < 1 or args.steps < 1 or args.warmup < 0:
        p.error("Invalid benchmark sizes")
    pilot, out = Path(args.pilot), Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    completion = json.loads((pilot / "complete.json").read_text())
    for name in ("inputs", "predictions"):
        file = pilot / (name + (".npz" if name == "inputs" else ".csv"))
        if file_hash(file) != completion[name + "_sha256"]:
            raise ValueError(f"Pilot artifact hash mismatch: {name}")
    with (pilot / "predictions.csv").open(encoding="utf-8") as f:
        records = [r for r in csv.DictReader(f) if float(r["sigma"]) == 0]
    records = list({r["molecule_id"]: r for r in records}.values())[:args.batch_size]
    if len(records) != args.batch_size:
        raise ValueError("Not enough distinct sigma-zero pilot molecules")
    molecules, vectors = [], []
    with np.load(pilot / "inputs.npz", allow_pickle=False) as arrays:
        for row in records:
            idx, key = int(row["molecule_id"]), row["noise_id"]
            y = torch.zeros((1, 19))
            y[0, 4] = float(row["target"])
            molecules.append((idx, Data(z=torch.from_numpy(arrays[f"molecule_{idx}_z"]),
                                        pos=torch.from_numpy(arrays[key + "_coordinates"]), y=y)))
            vectors.append(arrays[key + "_clean_descriptor"])
    batch = qm9_dense_collate(molecules)
    z, pos, mask, y = (getattr(batch, x).to(args.device) for x in ("z", "pos", "mask", "y"))
    tda = torch.from_numpy(np.stack(vectors)).to(args.device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    def sync():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
    results = []
    for name in ("baseline", "fusion"):
        model = EGNNGapRegressor() if name == "baseline" else EGNNTDAFiLMRegressor(130)
        filename, sha = CHECKPOINTS[name]
        checked_state_dict(model, Path(args.checkpoints) / filename, sha)
        model.to(args.device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        times = []
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        for i in range(args.warmup + args.steps):
            sync()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(z, pos, mask) if name == "baseline" else model(z, pos, mask, tda)
            loss = torch.nn.functional.mse_loss(prediction, y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite benchmark loss")
            loss.backward()
            optimizer.step()
            sync()
            if i >= args.warmup:
                times.append(time.perf_counter() - start)
        result = {"model": name, "step_seconds": times,
                  "median_step_seconds": float(np.median(times)),
                  "molecules_per_second": args.batch_size / float(np.median(times)),
                  "disposable_optimizer_steps": args.warmup + args.steps}
        if args.device.startswith("cuda"):
            result["peak_allocated_vram_bytes"] = torch.cuda.max_memory_allocated()
        results.append(result)
        del model, optimizer, prediction, loss
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    write_json(out, {"device": args.device,
                     "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
                     "platform": platform.platform(), "packages": versions(),
                     "batch_size": args.batch_size, "padded_atoms": int(z.shape[1]),
                     "molecule_ids": [int(r["molecule_id"]) for r in records],
                     "precision": "float32-TF32-off", "optimizer": "AdamW-lr0.001",
                     "pilot_inputs_sha256": completion["inputs_sha256"],
                     "scope": "Repeated fixed validation batch; throughput only; no model saved; excludes data/cache I/O and validation",
                     "results": results})
    print(out.read_text())


if __name__ == "__main__":
    main()
