"""Frozen-checkpoint sensitivity checks; these are not trained-model baselines."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch_geometric.data import Data

from scripts.prepare_paired import CHECKPOINTS
from src.data.collate import qm9_dense_collate
from src.models.fusion_gap import EGNNTDAFiLMRegressor
from src.validation import checked_state_dict, file_hash, versions, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", required=True)
    p.add_argument("--checkpoints", default="artifacts/checkpoints")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    root, out = Path(args.pilot), Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    complete = json.loads((root / "complete.json").read_text())
    for name, ext in (("inputs", ".npz"), ("predictions", ".csv")):
        if file_hash(root / (name + ext)) != complete[name + "_sha256"]:
            raise ValueError("Pilot artifact checksum mismatch")
    with (root / "predictions.csv").open() as f:
        rows = list(csv.DictReader(f))
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = EGNNTDAFiLMRegressor(130)
    filename, sha = CHECKPOINTS["fusion"]
    checked_state_dict(model, Path(args.checkpoints) / filename, sha)
    model.to(args.device).eval()
    generator = np.random.default_rng(20260913)
    results, all_predictions = [], []
    with np.load(root / "inputs.npz", allow_pickle=False) as arrays, torch.inference_mode():
        conditions = sorted({(float(r["sigma"]), int(r["noise_seed"])) for r in rows})
        for sigma, seed in conditions:
            group = [r for r in rows if float(r["sigma"]) == sigma and int(r["noise_seed"]) == seed]
            clean = np.stack([arrays[r["noise_id"] + "_clean_descriptor"] for r in group])
            noisy = np.stack([arrays[r["noise_id"] + "_perturbed_descriptor"] for r in group])
            permutation = generator.permutation(len(group))
            variants = {"clean": clean, "perturbed": noisy,
                        "shuffled": clean[permutation], "zero": np.zeros_like(clean),
                        "constant_first": np.repeat(clean[:1], len(clean), axis=0)}
            predictions = {name: [] for name in variants}
            pre, post = [], []
            for offset in range(0, len(group), 64):
                selected = group[offset:offset + 64]
                molecules = []
                for r in selected:
                    idx = int(r["molecule_id"])
                    y = torch.zeros((1, 19))
                    y[0, 4] = float(r["target"])
                    molecules.append((idx, Data(z=torch.from_numpy(arrays[f"molecule_{idx}_z"]),
                                               pos=torch.from_numpy(arrays[r["noise_id"] + "_coordinates"]), y=y)))
                batch = qm9_dense_collate(molecules)
                z, pos, mask = (getattr(batch, name).to(args.device) for name in ("z", "pos", "mask"))
                for name, vectors in variants.items():
                    tda = torch.from_numpy(vectors[offset:offset + len(selected)]).to(args.device)
                    predictions[name].extend(model(z, pos, mask, tda).cpu().tolist())
                    if name == "clean":
                        pre.append(model.film(tda).cpu().numpy())
                        post.append(model.film_act(model.film(tda)).cpu().numpy())
            predictions = {k: np.asarray(v) for k, v in predictions.items()}
            target = np.asarray([float(r["target"]) for r in group])
            reference = np.asarray([float(r["fusion_clean"]) for r in group])
            before, after = np.concatenate(pre), np.concatenate(post)
            result = {"sigma": sigma, "noise_seed": seed, "n_molecules": len(group),
                      "changed_descriptor_molecules": int(np.any(clean != noisy, axis=1).sum()),
                      "max_descriptor_change": float(np.max(np.abs(clean-noisy))),
                      "max_reference_prediction_difference_ev": float(np.max(np.abs(predictions["clean"]-reference))),
                      "clean_film_abs_preactivation_quantiles": np.quantile(np.abs(before), [0, .01, .5, .99, 1]).tolist(),
                      "clean_film_fraction_abs_tanh_above_0_9999": float((np.abs(after) > .9999).mean()),
                      "clean_film_max_channel_range_across_molecules": float(np.ptp(after, axis=0).max()),
                      "variants": {}}
            for name, pred in predictions.items():
                delta = np.abs(pred - predictions["clean"])
                result["variants"][name] = {"mae_ev": float(np.mean(np.abs(pred-target))),
                                           "mean_abs_prediction_change_ev": float(delta.mean()),
                                           "max_abs_prediction_change_ev": float(delta.max())}
            for i, row in enumerate(group):
                all_predictions.append({"molecule_id": int(row["molecule_id"]), "sigma": sigma,
                                        "noise_seed": seed, "target": float(row["target"]),
                                        "shuffled_descriptor_source_id": int(group[int(permutation[i])]["molecule_id"]),
                                        **{k: float(v[i]) for k, v in predictions.items()}})
            results.append(result)
    write_json(out, {"purpose": "Post-pilot reliance stress test; no fitting; zero/constant/shuffled inputs are not fair trained baselines",
                     "device": args.device, "precision": "float32-TF32-off", "packages": versions(),
                     "checkpoint_sha256": sha, "pilot_inputs_sha256": complete["inputs_sha256"],
                     "shuffle_seed": 20260913, "conditions": results, "predictions": all_predictions})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
