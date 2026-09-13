"""One-epoch, paired raw/standardized FiLM diagnostic; never selects on test data."""
from __future__ import annotations

import argparse
import copy
import csv
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from src.data.collate import qm9_dense_collate
from src.eval_paired import frozen_split, load_processed
from src.models.fusion_gap import EGNNTDAFiLMRegressor
from src.validation import (CheckedTDACache, array_hash, descriptor_spec, digest_json,
                            file_hash, versions, write_json)


def fit_scaler(training):
    x = np.asarray(training, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Scaler requires finite training rows")
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1
    return mean, scale


def transform(x, mean, scale):
    result = ((x.astype(np.float64) - mean) / scale).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite scaled features")
    return result


def state_hash(model):
    return digest_json({k: array_hash(v.detach().cpu().numpy())
                        for k, v in model.state_dict().items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Output must be a new directory")
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(args.device)
    dataset_sha = file_hash(args.data)
    ds = load_processed(args.data)
    split = frozen_split(args.split, dataset_sha, ds)
    train_ids, val_ids = split["train"][:4096], split["val"][:256]
    order = torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(42)).numpy()
    model_template = EGNNTDAFiLMRegressor(tda_dim=130)
    initial_hash = state_hash(model_template)
    manifest = dict(schema=1, experiment="raw-vs-training-standardized-one-epoch",
                    dataset_sha256=dataset_sha, split_sha256=file_hash(args.split),
                    train_ids=train_ids, val_ids=val_ids, seed=42, epochs=1, batch_size=64,
                    optimizer={"name": "AdamW", "lr": 0.001, "weight_decay": 0.01},
                    loss="MSE in eV squared", scaler_fit="4096 diagnostic training IDs only",
                    order=order.tolist(), initial_state_sha256=initial_hash,
                    descriptor=descriptor_spec(), packages=versions(),
                    sources={str(p).replace("\\", "/"): file_hash(p) for p in
                             [Path(__file__), Path("src/models/fusion_gap.py"),
                              Path("src/models/egnn_gap.py"), Path("src/data/collate.py"),
                              Path("src/validation.py"), Path("src/eval_paired.py")]},
                    device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    tf32=False, saturation_warning_fraction=0.95,
                    saturation_cutoff=0.9999, evaluation="clean validation only; no test access")
    # Freeze protocol and membership before feature extraction or model outcomes.
    write_json(args.out / "manifest.json", manifest)
    cache = CheckedTDACache(args.cache, dataset_sha)
    def features(ids):
        return np.stack([cache.get(i, ds[i].pos.numpy())[0] for i in ids])
    train_x, val_x = features(train_ids), features(val_ids)
    mean, scale = fit_scaler(train_x)
    np.savez_compressed(args.out / "inputs.npz", train_ids=train_ids, val_ids=val_ids,
                        train_tda=train_x, val_tda=val_x, order=order, mean=mean, scale=scale)
    preprocessing_seconds = time.perf_counter() - started
    rows, steps, results = [], [], {}
    def batch(ids):
        b = qm9_dense_collate([(int(i), ds[int(i)]) for i in ids])
        return b.z.to(device), b.pos.to(device), b.mask.to(device), b.y.to(device)
    # Materialize the same batches once, independent of arm/evaluation RNG.
    train_batches = [(order[s:s+64], batch(np.asarray(train_ids)[order[s:s+64]]))
                     for s in range(0, len(train_ids), 64)]
    val_batches = [(slice(s, s+64), batch(val_ids[s:s+64])) for s in range(0, len(val_ids), 64)]
    shuffle = np.random.default_rng(20260913).permutation(len(val_ids))

    for arm in ("raw", "standardized"):
        tx, vx = (train_x, val_x) if arm == "raw" else (transform(train_x, mean, scale), transform(val_x, mean, scale))
        fixed = mean.astype(np.float32) if arm == "raw" else np.zeros(130, dtype=np.float32)
        model = copy.deepcopy(model_template).to(device)
        if state_hash(model) != initial_hash:
            raise ValueError("Arms must start with identical weights")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

        @torch.inference_mode()
        def evaluate(stage):
            model.eval()
            preds, labels, shuffled, constant, pre = [], [], [], [], []
            for sl, (z, pos, mask, y) in val_batches:
                t = torch.from_numpy(vx[sl]).to(device)
                pred = model(z, pos, mask, t)
                preds.extend(pred.cpu().tolist())
                labels.extend(y.cpu().tolist())
                shuffled.extend(model(z, pos, mask, torch.from_numpy(vx[shuffle][sl]).to(device)).cpu().tolist())
                constant.extend(model(z, pos, mask, torch.from_numpy(np.tile(fixed, (len(y), 1))).to(device)).cpu().tolist())
                pre.append(model.film(t).cpu().numpy())
            p, y, s, c = map(np.asarray, (preds, labels, shuffled, constant))
            a = np.concatenate(pre)
            if not all(np.isfinite(x).all() for x in (p, y, s, c, a)):
                raise ValueError("Nonfinite evaluation")
            # Same scalar property under rigid motion, atom permutation and padding.
            sl, (z, pos, mask, _) = val_batches[0]
            t = torch.from_numpy(vx[sl]).to(device)
            ref = model(z, pos, mask, t)
            transformed = model(z.flip(1), (pos.flip(1)[..., [1, 2, 0]] + 2), mask.flip(1), t)
            padded = model(torch.nn.functional.pad(z, (0, 3)),
                           torch.nn.functional.pad(pos, (0, 0, 0, 3)),
                           torch.nn.functional.pad(mask, (0, 3)), t)
            symmetry_error = max((ref-transformed).abs().max().item(), (ref-padded).abs().max().item())
            if symmetry_error > 2e-5:
                raise ValueError(f"Symmetry/padding check failed: {symmetry_error}")
            for i, yy, pp, ss, cc in zip(val_ids, y, p, s, c):
                rows.append(dict(arm=arm, stage=stage, molecule_id=i, target=yy,
                                 prediction=pp, shuffled_prediction=ss, constant_prediction=cc))
            return dict(mae_eV=float(np.abs(p-y).mean()),
                        saturation_fraction=float((np.abs(np.tanh(a)) > .9999).mean()),
                        abs_pretanh_quantiles=np.quantile(np.abs(a), [0, .5, .95, 1]).tolist(),
                        mean_channel_range=float(np.ptp(np.tanh(a), axis=0).mean()),
                        shuffled_mean_abs_prediction_change_eV=float(np.abs(p-s).mean()),
                        shuffled_max_abs_prediction_change_eV=float(np.abs(p-s).max()),
                        constant_mean_abs_prediction_change_eV=float(np.abs(p-c).mean()),
                        symmetry_padding_max_abs_error_eV=symmetry_error)

        result = {"initial": evaluate("initial"), "feature_min": float(tx.min()), "feature_max": float(tx.max())}
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        training_start = time.perf_counter()
        for step, (indices, (z, pos, mask, y)) in enumerate(train_batches):
            optimizer.zero_grad(set_to_none=True)
            pred = model(z, pos, mask, torch.from_numpy(tx[indices]).to(device))
            loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError(f"Nonfinite training: {arm} step {step}")
            norm = sum(p.grad.detach().square().sum() for p in model.film.parameters()).sqrt().item()
            steps.append(dict(arm=arm, step=step, mse_eV2=loss.item(), film_gradient_norm=norm))
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        result["training_seconds"] = time.perf_counter() - training_start
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        result["after_epoch"] = evaluate("after_epoch")
        checkpoint = args.out / f"{arm}.pt"
        torch.save({"state_dict": model.state_dict(), "arm": arm,
                    "mean": torch.from_numpy(mean), "scale": torch.from_numpy(scale),
                    "inputs_sha256": file_hash(args.out / "inputs.npz"),
                    "manifest_sha256": file_hash(args.out / "manifest.json")}, checkpoint)
        result["checkpoint_sha256"] = file_hash(checkpoint)
        result["initial_state_sha256"] = initial_hash
        result["min_film_gradient_norm"] = min(r["film_gradient_norm"] for r in steps if r["arm"] == arm)
        results[arm] = result
        print(arm, result, flush=True)
        del model, optimizer
    for name, values in (("predictions.csv", rows), ("steps.csv", steps)):
        with (args.out / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    results["preprocessing_seconds"] = preprocessing_seconds
    results["wall_seconds"] = time.perf_counter() - started
    results["gate"] = "pass" if results["standardized"]["after_epoch"]["saturation_fraction"] <= .95 and results["standardized"]["after_epoch"]["shuffled_mean_abs_prediction_change_eV"] > 1e-5 else "investigate"
    write_json(args.out / "summary.json", results)
    write_json(args.out / "complete.json", {p.name: file_hash(p) for p in args.out.iterdir() if p.is_file()})
    print("COMPLETE", results["gate"], results["wall_seconds"], flush=True)


if __name__ == "__main__":
    main()
