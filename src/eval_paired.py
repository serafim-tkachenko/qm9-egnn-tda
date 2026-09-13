"""Small, retrospective paired evaluation of recovered legacy checkpoints.

Run ``python -m src.eval_paired --help``. Output directories must be new.
No training, target scaling, descriptor redesign or historical-output overwrites.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from src.data.collate import qm9_dense_collate
from src.data.qm9_data import make_splits
from src.models.egnn_gap import EGNNGapRegressor
from src.models.fusion_gap import EGNNTDAFiLMRegressor
from src.validation import (CheckedTDACache, DESCRIPTOR_VERSION, array_hash,
                            checked_state_dict, descriptor_spec, digest_json,
                            file_hash, paired_summary, perturb, versions, write_json)


def load_processed(path):
    """Load already processed PyG QM9 without triggering a raw-data download."""
    ds = InMemoryDataset()
    # data_v3.pt is PyG's (data dictionary, slices, Data class) format.
    with torch.serialization.safe_globals([Data]):
        saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, tuple) or len(saved) != 3 or saved[2] is not Data:
        raise ValueError("Expected PyG data_v3.pt (dict, slices, Data)")
    ds.data, ds.slices = Data.from_dict(saved[0]), saved[1]
    if ds._data.y.shape[1] != 19 or ds._data.pos.shape[1] != 3:
        raise ValueError("Unexpected QM9 target or coordinate shape")
    if not torch.isfinite(ds._data.y).all() or not torch.isfinite(ds._data.pos).all():
        raise ValueError("Nonfinite QM9 values")
    return ds


def frozen_split(path, dataset_sha, ds):
    path = Path(path)
    identity = {"dataset_sha256": dataset_sha, "n_molecules": len(ds), "split_seed": 42,
                "algorithm": "torch-CPU-random_split-80-10-10"}
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise ValueError("Frozen split/dataset identity mismatch")
    else:
        split = make_splits(ds, seed=42)
        manifest = {"identity": identity, **{name: list(getattr(split, name).indices)
                                            for name in ("train", "val", "test")}}
        write_json(path, manifest)
    flat = sum([manifest[name] for name in ("train", "val", "test")], [])
    if sorted(flat) != list(range(len(ds))):
        raise ValueError("Split indices must be a disjoint exhaustive partition")
    # Protect legacy checkpoint membership, not just disjointness.
    expected = make_splits(ds, seed=42)
    for name in ("train", "val", "test"):
        if manifest[name] != list(getattr(expected, name).indices):
            raise ValueError("Split differs from the legacy checkpoint's seed-42 split")
    return manifest


def load_models(provenance, device):
    if provenance["descriptor_version"] != DESCRIPTOR_VERSION:
        raise ValueError("Checkpoint descriptor version mismatch")
    if provenance["architecture"] != "source-defaults-width128-depth4-film256-head256":
        raise ValueError("Unknown checkpoint architecture")
    if provenance["egnn_pytorch_version"] != versions()["egnn-pytorch"]:
        raise ValueError("EGNN implementation version mismatch")
    models = [EGNNGapRegressor(), EGNNTDAFiLMRegressor(tda_dim=130)]
    for model, name in zip(models, ("baseline", "fusion")):
        item = provenance[name]
        checked_state_dict(model, item["path"], item["sha256"])
        model.to(device).eval()
    return models


@torch.inference_mode()
def predict(models, batch, clean_tda, noisy_tda, device):
    z, pos, mask = (getattr(batch, x).to(device) for x in ("z", "pos", "mask"))
    return np.stack([
        models[0](z, pos, mask).cpu().numpy(),
        models[1](z, pos, mask, torch.from_numpy(clean_tda).to(device)).cpu().numpy(),
        models[1](z, pos, mask, torch.from_numpy(noisy_tda).to(device)).cpu().numpy(),
    ], axis=1)


def make_figure(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = {"egnn": "EGNN", "fusion_clean": "Fusion: clean topology",
              "fusion_perturbed": "Fusion: matched-input topology"}
    xs = np.arange(len(summary))
    for j, (name, label) in enumerate(labels.items()):
        ys = np.array([r[name + "_mae"] for r in summary])
        ci = np.array([r[name + "_mae_ci95"] for r in summary]).T
        ax.errorbar(xs + (j - 1) * .12, ys, yerr=[ys - ci[0], ci[1] - ys],
                    fmt="o", capsize=4, label=label)
    ax.set_xticks(xs, [str(r["sigma"]) for r in summary])
    ax.set(xlabel="Coordinate noise sigma (angstrom)", ylabel="MAE (eV)",
           title="Paired checkpoint diagnostic — frozen development molecules")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=.2)
    fig.text(.5, .01, "95% molecule-bootstrap intervals; fixed checkpoints; original clean labels",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="Recovered PyG data_v3.pt")
    p.add_argument("--provenance", required=True, help="Reviewed legacy checkpoint/data manifest")
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--cache", required=True, help="Separate versioned cache directory")
    p.add_argument("--out", required=True, help="New output directory")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--size", type=int, default=256, help="First N split IDs; 0 means whole split")
    p.add_argument("--sigmas", type=float, nargs="+", default=[0., .1])
    p.add_argument("--noise-seeds", type=int, nargs="+", default=[20260913])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compute-environment", choices=("local", "colab"), default="local")
    args = p.parse_args()
    if args.size < 0 or args.batch_size < 1 or len(set(args.sigmas)) != len(args.sigmas):
        p.error("Invalid size, batch size or duplicate sigmas")
    if any(s < 0 or not np.isfinite(s) for s in args.sigmas):
        p.error("Sigmas must be finite and nonnegative")
    if len(set(args.noise_seeds)) != len(args.noise_seeds):
        p.error("Duplicate noise seeds")
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("Choose a new output directory; historical runs are immutable")
    provenance = json.loads(Path(args.provenance).read_text(encoding="utf-8"))
    dataset_sha = file_hash(args.data)
    if dataset_sha != provenance["dataset_sha256"]:
        raise ValueError("Processed dataset hash mismatch")
    models = load_models(provenance, args.device)
    ds = load_processed(args.data)
    splits = frozen_split(args.split_manifest, dataset_sha, ds)
    ids = splits[args.split][:args.size or None]
    if not ids:
        raise ValueError("Empty evaluation subset")
    cache = CheckedTDACache(args.cache, dataset_sha)
    out.mkdir(parents=True)
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
        "purpose": "retrospective-development-diagnostic", "target_index": 4,
        "target_units": "eV", "coordinate_units": "angstrom", "precision": "float32-TF32-off",
        "labels": "original-clean-molecule-gaps", "packages": versions(),
        "platform": platform.platform(), "python": platform.python_version(),
        "dataset_sha256": dataset_sha, "split_manifest_sha256": file_hash(args.split_manifest),
        "subset_ids": ids, "subset_sha256": digest_json(ids), "descriptor": descriptor_spec(),
        "checkpoint_provenance": provenance,
        "checkpoint_metadata_limit": "Legacy weights contain no embedded descriptor/config/split metadata; binding is recovered from source and artifacts, not retroactively proven by hash.",
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(path): file_hash(path) for path in sorted(Path("src").rglob("*.py"))},
        "parameters": {name: sum(v.numel() for v in model.parameters())
                       for name, model in zip(("baseline", "fusion"), models)},
    }
    if args.device.startswith("cuda"):
        manifest["gpu"] = torch.cuda.get_device_name()
        manifest["cuda"] = torch.version.cuda
        torch.cuda.reset_peak_memory_stats()
    write_json(out / "manifest.json", manifest)
    rows, topo_seconds, inference_seconds = [], 0., 0.
    # Store exact ragged inputs without object arrays/pickle.
    input_arrays = {}
    for sigma in args.sigmas:
        for noise_seed in args.noise_seeds:
            for offset in range(0, len(ids), args.batch_size):
                batch_ids = ids[offset:offset + args.batch_size]
                records, cleans, noisy_vecs, molecules = [], [], [], []
                t = time.perf_counter()
                for idx in batch_ids:
                    data = ds[idx]
                    clean = data.pos.numpy().astype(np.float32)
                    noisy, noise_id = perturb(clean, idx, sigma, noise_seed)
                    clean_vec, clean_meta = cache.get(idx, clean)
                    noisy_vec, noisy_meta = cache.get(idx, noisy)
                    if sigma == 0 and not np.array_equal(clean_vec, noisy_vec):
                        raise AssertionError("Sigma-zero descriptor mismatch")
                    data = data.clone()
                    data.pos = torch.from_numpy(noisy)
                    molecules.append((idx, data))
                    cleans.append(clean_vec)
                    noisy_vecs.append(noisy_vec)
                    records.append({
                        "molecule_id": idx, "qm9_name": str(getattr(data, "name", "")),
                        "n_atoms": len(clean), "sigma": sigma, "noise_seed": noise_seed,
                        "noise_id": noise_id, "target": float(data.y.view(-1)[4]),
                        "clean_coordinates_sha256": array_hash(clean),
                        "coordinates_sha256": array_hash(noisy),
                        "atomic_numbers_sha256": array_hash(data.z.numpy()),
                        "clean_descriptor_sha256": clean_meta["vector_sha256"],
                        "perturbed_descriptor_sha256": noisy_meta["vector_sha256"],
                    })
                    input_arrays[f"{noise_id}_coordinates"] = noisy
                    input_arrays[f"{noise_id}_clean_descriptor"] = clean_vec
                    input_arrays[f"{noise_id}_perturbed_descriptor"] = noisy_vec
                    input_arrays[f"molecule_{idx}_z"] = data.z.numpy()
                    input_arrays[f"molecule_{idx}_clean_coordinates"] = clean
                topo_seconds += time.perf_counter() - t
                batch = qm9_dense_collate(molecules)
                t = time.perf_counter()
                predictions = predict(models, batch, np.stack(cleans), np.stack(noisy_vecs), args.device)
                inference_seconds += time.perf_counter() - t
                if sigma == 0 and not np.array_equal(predictions[:, 1], predictions[:, 2]):
                    raise AssertionError("Sigma-zero fusion prediction mismatch")
                for record, pred in zip(records, predictions):
                    for name, value in zip(("egnn", "fusion_clean", "fusion_perturbed"), pred):
                        record[name] = float(value)
                        record[name + "_abs_error"] = abs(float(value) - record["target"])
                    rows.append(record)
                # Persist completed batches so interrupted runs retain predictions.
                first_batch = len(rows) == len(records)
                with (out / "predictions.csv").open("w" if first_batch else "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    if first_batch:
                        writer.writeheader()
                    writer.writerows(rows[-len(records):])
                print(f"sigma={sigma:g} seed={noise_seed} {min(offset + args.batch_size, len(ids))}/{len(ids)}", flush=True)
    np.savez_compressed(out / "inputs.npz", **input_arrays)
    summary = paired_summary(rows)
    write_json(out / "summary.json", {
        "interval": "95% percentile bootstrap over molecules; average noise seeds within molecule; fixed checkpoints",
        "delta_sign": "fusion minus EGNN; negative favors fusion",
        "resamples": 4000, "bootstrap_seed": 20260913, "conditions": summary,
    })
    make_figure(summary, out / "paired_mae.png")
    compute = {"wall_seconds": time.perf_counter() - start,
               "descriptor_and_input_seconds": topo_seconds,
               "three_arm_inference_seconds": inference_seconds,
               "colab_credits_consumed": 0 if args.compute_environment == "local" else None,
               "credit_accounting": "Use Colab account/runtime readings for remote runs; not exposed by PyTorch",
               "training_steps": 0}
    if args.device.startswith("cuda"):
        compute["peak_allocated_vram_bytes"] = torch.cuda.max_memory_allocated()
    write_json(out / "compute.json", compute)
    write_json(out / "complete.json", {"predictions_sha256": file_hash(out / "predictions.csv"),
                                       "inputs_sha256": file_hash(out / "inputs.npz")})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
