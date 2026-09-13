"""Recover and check legacy artifacts before allowing a paired checkpoint pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from src.eval_paired import frozen_split, load_models, load_processed, predict
from src.data.collate import qm9_dense_collate
from src.data.tda_features import TDACache, TDAConfig
from src.validation import (DESCRIPTOR_VERSION, array_hash, file_hash, versions,
                            write_json)


CHECKPOINTS = {
    "baseline": ("best_egnn.pt", "3678bab48fe8b817e59bfbde08b03c85e67700f098f58dd70e31a04a7ac89138"),
    "fusion": ("best_fusion.pt", "0493bf12c3a9c76e0a2bc17fc47265a0dcd3ea63b0c4382e127e9c3f1077797b"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-root", required=True)
    p.add_argument("--out", required=True, help="New directory for recovery records")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    root, out = Path(args.original_root), Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    data_path = root / "data/qm9/processed/data_v3.pt"
    data_sha = file_hash(data_path)
    ds = load_processed(data_path)
    if len(ds) != 130831:
        raise ValueError("Unexpected original QM9 population")
    transforms = {}
    for name in ("pre_transform.pt", "pre_filter.pt"):
        path = data_path.parent / name
        transforms[name] = torch.load(path, map_location="cpu", weights_only=True)
        if transforms[name] != "None":
            raise ValueError(f"Review original preprocessing before proceeding: {transforms}")
    provenance = {
        "descriptor_version": DESCRIPTOR_VERSION,
        "architecture": "source-defaults-width128-depth4-film256-head256",
        "egnn_pytorch_version": "0.2.8", "dataset_sha256": data_sha,
        "target_index": 4, "target_transform": "none", "preprocessing_records": transforms,
        "historical_configuration": {"split_seed": 42, "training_seed": 42, "epochs": 10,
                                     "batch_size": 64, "lr": .001},
        "metadata_status": "Source/artifact-derived legacy binding; original environment and embedded training metadata unavailable",
    }
    for name, (filename, expected) in CHECKPOINTS.items():
        path = root / "checkpoints" / filename
        actual = file_hash(path)
        if actual != expected:
            raise ValueError(f"Recovered checkpoint differs from inventoried artifact: {filename}")
        provenance[name] = {"path": str(path), "sha256": actual}
    write_json(out / "provenance.json", provenance)
    split = frozen_split(out / "split42.json", data_sha, ds)
    ids = split["val"][:256]
    tda = TDACache(TDAConfig(cache_dir=str(out / "scratch"), n_jobs=1))
    checks, mismatches = [], []
    for idx in ids:
        coords = ds[idx].pos.numpy()
        vector = tda.compute_vec(coords)
        old_path = root / "tda_cache" / f"{idx:06d}.npy"
        old = np.load(old_path, allow_pickle=False)
        if old.shape != (130,) or not np.isfinite(old).all():
            raise ValueError(f"Malformed legacy descriptor: {old_path}")
        delta = float(np.abs(old - vector).max())
        # Betti counts must match exactly; float entropy allows cross-platform rounding.
        compatible = np.array_equal(old[:128], vector[:128]) and np.allclose(old[128:], vector[128:], rtol=1e-6, atol=1e-6)
        if not compatible:
            mismatches.append(idx)
        checks.append({"molecule_id": idx, "legacy_file_sha256": file_hash(old_path),
                       "legacy_descriptor_sha256": array_hash(old),
                       "recomputed_descriptor_sha256": array_hash(vector),
                       "max_abs_difference": delta, "compatible": bool(compatible)})
    write_json(out / "legacy_cache_checks.json", checks)
    if mismatches:
        raise ValueError(f"Legacy descriptor mismatch for {len(mismatches)} of 256 molecules; inspect recovery records before evaluation")
    # Concrete adaptive-grid examples, selected without model outcomes.
    examples = []
    for idx in ids[:3]:
        vector = tda.compute_vec(ds[idx].pos.numpy())
        examples.append({"molecule_id": idx, "n_atoms": len(ds[idx].z),
                         "samplings": {str(k): v.tolist() for k, v in tda.betti.samplings_.items()},
                         "entropy": vector[-2:].tolist()})
    write_json(out / "adaptive_grid_examples.json", examples)
    models = load_models(provenance, args.device)
    invariance = []
    rng = np.random.default_rng(20260913)
    for idx in ids[:8]:
        original = ds[idx]
        clean_v = tda.compute_vec(original.pos.numpy())[None]
        ref = predict(models, qm9_dense_collate([(idx, original)]), clean_v, clean_v, args.device)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        perm = torch.from_numpy(rng.permutation(len(original.z)))
        for name in ("translation", "rotation", "reflection", "permutation", "padding"):
            d = original.clone()
            if name == "translation":
                d.pos = d.pos + torch.tensor([1., -2., 3.])
            elif name == "rotation":
                d.pos = d.pos @ torch.tensor(q, dtype=torch.float32)
            elif name == "reflection":
                d.pos = d.pos * torch.tensor([-1., 1., 1.])
            elif name == "permutation":
                d.pos, d.z = d.pos[perm], d.z[perm]
            vec = tda.compute_vec(d.pos.numpy())[None]
            batch = qm9_dense_collate([(idx, d)])
            if name == "padding":
                batch.pos = torch.nn.functional.pad(batch.pos, (0, 0, 0, 5), value=1234)
                batch.z = torch.nn.functional.pad(batch.z, (0, 5), value=0)
                batch.mask = torch.nn.functional.pad(batch.mask, (0, 5), value=False)
            pred = predict(models, batch, vec, vec, args.device)
            invariance.append({"molecule_id": idx, "transform": name,
                               "descriptor_max_abs_difference": float(np.abs(vec-clean_v).max()),
                               "prediction_max_abs_difference_ev": float(np.abs(pred-ref).max()),
                               "passed": bool(np.allclose(pred, ref, rtol=1e-5, atol=1e-4))})
    write_json(out / "invariance.json", invariance)
    if not all(r["passed"] for r in invariance):
        raise ValueError("Recovered-checkpoint invariance check failed; review before expanding")
    write_json(out / "recovery_complete.json", {
        "data_sha256": data_sha, "n_molecules": len(ds), "packages": versions(),
        "python": platform.python_version(), "platform": platform.platform(),
        "seconds": time.perf_counter() - start,
        "legacy_cache_molecules_checked": len(checks), "invariance_molecules_checked": 8,
        "clean_full_split_metrics_reproduced": False,
    })
    print("Recovery checks passed; provenance and frozen split ready", flush=True)


if __name__ == "__main__":
    main()
