"""Frozen four-arm, three-seed QM9 replication with resumable epoch boundaries."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from scripts.diagnose_conditioning import fit_scaler, transform, state_hash
from src.eval_paired import load_processed, frozen_split
from src.models.egnn_gap import EGNNGapRegressor
from src.models.fusion_gap import EGNNTDAFiLMRegressor
from src.data.tda_features import TDACache, TDAConfig
from src.validation import array_hash, descriptor_spec, file_hash, versions, write_json

ARMS = ("egnn", "tda", "geometry", "constant")


def geometry(z, pos):
    """Counts H/C/N/O/F and population statistics of unordered distances in Å."""
    z, pos = np.asarray(z), np.asarray(pos, dtype=np.float64)
    d = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)[np.triu_indices(len(z), 1)]
    stats = [d.mean(), d.std(), d.min(), d.max(), np.sqrt(np.mean(d*d))] if len(d) else [0]*5
    return np.asarray([np.count_nonzero(z == a) for a in (1, 6, 7, 8, 9)] + stats, dtype=np.float32)


def topology_worker(pos):
    # No per-molecule files: a compact aggregate avoids exhausting local disk.
    computer = TDACache(TDAConfig(cache_dir="outputs/replication-worker-unused", n_jobs=1))
    return computer.compute_vec(pos)


def models_for_seed(seed):
    torch.manual_seed(seed)
    baseline = EGNNGapRegressor()
    models = {"egnn": baseline}
    for arm in ARMS[1:]:
        torch.manual_seed(seed + 1000)
        model = EGNNTDAFiLMRegressor(tda_dim=10 if arm == "geometry" else 130,
                                    film_hidden=371 if arm == "geometry" else 256)
        model.encoder.load_state_dict(baseline.encoder.state_dict())
        model.head.load_state_dict(baseline.head.state_dict())
        models[arm] = model
    expected = {"egnn": 895189, "tda": 994517, "geometry": 994502, "constant": 994517}
    for arm, model in models.items():
        if sum(p.numel() for p in model.parameters()) != expected[arm]:
            raise ValueError("Unexpected parameter count")
    return models


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_json(tmp, value)
    tmp.replace(path)


def atomic_torch(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise ValueError("GPU required for this replication budget")
    args.out.mkdir(parents=True, exist_ok=True)
    dataset_sha = file_hash(args.data)
    ds = load_processed(args.data)
    split = frozen_split(args.split, dataset_sha, ds)
    manifest = dict(schema=1, experiment="controlled-replication-v1", dataset_sha256=dataset_sha,
                    split_sha256=file_hash(args.split), seeds=[42,43,44], epochs=10, batch_size=64,
                    arms=list(ARMS), optimizer={"name":"AdamW","lr":.001,"weight_decay":.01},
                    loss="MSE_eV2", selection="lowest full-validation MAE; earliest on ties",
                    descriptor=descriptor_spec(), scaler="full training split only, population std, floor 1e-6 to 1",
                    constant="full training mean TDA; zero after training standardization",
                    geometry="counts H,C,N,O,F; unordered pair distance mean,population std,min,max,RMS in angstrom",
                    initialization="shared encoder/head per seed; identical TDA/constant FiLM initialization",
                    order="torch CPU randperm, seed 100000*model_seed+epoch (1-based)",
                    noise_sigmas=[0,.01,.05,.1], noise_seeds=[20260913,20260914,20260915],
                    noise_test_ids=split["test"][:1024], clean_test_ids=split["test"],
                    noise_membership="first 1024 frozen test IDs; bounded before outcomes",
                    practical_criterion="TDA mean MAE at sigma .10 at least 1% lower than EACH of EGNN, geometry and constant; lower in all 3 training seeds; paired molecule interval reported separately",
                    claim_limit="three seeds and ten epochs exploratory; historical test is retrospective",
                    epoch_extension="none in this run; report all ten epochs",
                    packages=versions(), gpu=torch.cuda.get_device_name(0), tf32=False,
                    sources={str(f):file_hash(f) for f in map(Path,["scripts/run_controlled_replication.py", "scripts/diagnose_conditioning.py", "src/models/egnn_gap.py", "src/models/fusion_gap.py", "src/data/tda_features.py", "src/validation.py"])})
    mp = args.out / "manifest.json"
    if mp.exists():
        if json.loads(mp.read_text()) != manifest:
            raise ValueError("Resume configuration/source mismatch")
    else:
        write_json(mp, manifest)
    prep_start = time.perf_counter()
    fp = args.out / "features.npz"
    if not fp.exists():
        ids = split["train"] + split["val"]
        coords = [ds[i].pos.numpy() for i in ids]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            values = []
            for j, value in enumerate(pool.map(topology_worker, coords, chunksize=128)):
                values.append(value)
                if (j+1) % 10000 == 0:
                    print("FEATURES", j+1, "/", len(ids), flush=True)
        tda = np.stack(values)
        geom = np.stack([geometry(ds[i].z.numpy(), pos) for i,pos in zip(ids, coords)])
        tm, ts = fit_scaler(tda[:len(split["train"])])
        gm, gs = fit_scaler(geom[:len(split["train"])])
        tmp = args.out / "features.tmp.npz"
        np.savez_compressed(tmp, ids=ids, tda=tda, geometry=geom, tda_mean=tm, tda_scale=ts,
                            geometry_mean=gm, geometry_scale=gs)
        tmp.replace(fp)
        write_json(args.out / "preparation.json", {"seconds":time.perf_counter()-prep_start,
                   "features_sha256":file_hash(fp), "manifest_sha256":file_hash(mp)})
    prep = json.loads((args.out / "preparation.json").read_text())
    if file_hash(fp) != prep["features_sha256"] or file_hash(mp) != prep["manifest_sha256"]:
        raise ValueError("Preparation binding mismatch")
    with np.load(fp) as f:
        ids = f["ids"]
        if ids.tolist() != split["train"] + split["val"]:
            raise ValueError("Feature membership/order mismatch")
        features = {"tda":torch.from_numpy(transform(f["tda"],f["tda_mean"],f["tda_scale"])).to(device),
                    "geometry":torch.from_numpy(transform(f["geometry"],f["geometry_mean"],f["geometry_scale"])).to(device)}
    # All coordinates fit in GPU memory; trim padding to each batch's largest molecule.
    sizes = np.array([len(ds[int(i)].z) for i in ids])
    z = np.zeros((len(ids), int(sizes.max())), dtype=np.int64)
    pos = np.zeros((*z.shape, 3), dtype=np.float32)
    y = np.empty(len(ids), dtype=np.float32)
    for j,i in enumerate(ids):
        d = ds[int(i)]
        z[j,:sizes[j]], pos[j,:sizes[j]], y[j] = d.z.numpy(), d.pos.numpy(), d.y.view(-1)[4].item()
    z, pos, y = (torch.from_numpy(a).to(device) for a in (z,pos,y))
    mask = z != 0
    ntrain = len(split["train"])
    val_order = np.arange(ntrain, len(ids))
    fixed = torch.zeros((64,130), device=device)
    def predict(model, arm, indices):
        width = int(sizes[indices].max())
        idx = torch.as_tensor(indices, device=device)
        b = (z[idx,:width], pos[idx,:width], mask[idx,:width])
        return model(*b) if arm == "egnn" else model(*b, fixed[:len(indices)] if arm == "constant" else features[arm][idx])
    @torch.inference_mode()
    def validate(model, arm):
        model.eval()
        error = 0.
        for start in range(0,len(val_order),64):
            indices=val_order[start:start+64]
            pred=predict(model,arm,indices)
            if not torch.isfinite(pred).all():
                raise ValueError("Nonfinite validation predictions")
            error += (pred-y[indices]).abs().sum().item()
        sat = None
        if arm != "egnn":
            t = torch.zeros((256,130),device=device) if arm == "constant" else features[arm][ntrain:ntrain+256]
            sat = (model.film(t).tanh().abs() > .9999).float().mean().item()
        return error/len(val_order), sat
    for seed in manifest["seeds"]:
        templates = models_for_seed(seed)
        for arm in ARMS:
            run = args.out / f"{arm}-{seed}"
            run.mkdir(exist_ok=True)
            if (run / "complete.json").exists():
                done=json.loads((run/"complete.json").read_text())
                if file_hash(run/"best.pt") != done["best_sha256"]:
                    raise ValueError("Completed checkpoint corrupted")
                continue
            model = copy.deepcopy(templates[arm]).to(device)
            initial_hash = state_hash(model)
            optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            history, best, epoch0 = [], float("inf"), 0
            if (run / "resume.pt").exists():
                ck = torch.load(run/"resume.pt",map_location=device,weights_only=True)
                if ck["manifest_sha256"] != file_hash(mp):
                    raise ValueError("Checkpoint manifest mismatch")
                model.load_state_dict(ck["model"])
                optimizer.load_state_dict(ck["optimizer"])
                history, best, epoch0 = ck["history"], ck["best"], ck["epoch"]
            for epoch in range(epoch0+1,11):
                order=torch.randperm(ntrain,generator=torch.Generator().manual_seed(100000*seed+epoch)).numpy()
                torch.cuda.synchronize()
                started=time.perf_counter()
                model.train()
                loss_sum, min_grad, max_grad = 0., float("inf"), 0.
                for start in range(0,ntrain,64):
                    indices=order[start:start+64]
                    optimizer.zero_grad(set_to_none=True)
                    pred=predict(model,arm,indices)
                    loss=torch.nn.functional.mse_loss(pred,y[indices])
                    loss.backward()
                    norms=torch.stack([p.grad.square().sum() for p in model.parameters() if p.grad is not None])
                    if not torch.isfinite(loss) or not torch.isfinite(norms).all():
                        raise ValueError(f"Nonfinite training {arm}/{seed}/{epoch}/{start}")
                    if arm != "egnn":
                        grad=torch.stack([p.grad.square().sum() for p in model.film.parameters()]).sum().sqrt().item()
                        min_grad, max_grad = min(min_grad,grad), max(max_grad,grad)
                    optimizer.step()
                    loss_sum += loss.item()*len(indices)
                torch.cuda.synchronize()
                train_seconds=time.perf_counter()-started
                val_mae,sat=validate(model,arm)
                total_seconds=time.perf_counter()-started
                record=dict(epoch=epoch,train_mse_eV2=loss_sum/ntrain,val_mae_eV=val_mae,
                            saturation_fraction=sat,film_gradient_min=None if arm=="egnn" else min_grad,
                            film_gradient_max=None if arm=="egnn" else max_grad,
                            training_seconds=train_seconds,train_and_validation_seconds=total_seconds,
                            order_sha256=array_hash(order))
                history.append(record)
                if val_mae < best:
                    best=val_mae
                    atomic_torch(run/"best.pt",dict(model=model.state_dict(),seed=seed,arm=arm,epoch=epoch,
                                 val_mae_eV=best,manifest_sha256=file_hash(mp),features_sha256=file_hash(fp)))
                atomic_torch(run/"resume.pt",dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                             history=history,best=best,epoch=epoch,manifest_sha256=file_hash(mp)))
                atomic_json(run/"history.json",history)
                print("EPOCH",arm,seed,record,flush=True)
            write_json(run/"complete.json",dict(best_sha256=file_hash(run/"best.pt"),
                       history_sha256=file_hash(run/"history.json"),initial_state_sha256=initial_hash,
                       best_val_mae_eV=best,epochs=10,manifest_sha256=file_hash(mp)))
            del model, optimizer
        del templates
    write_json(args.out/"training_complete.json",{f"{a}-{s}":json.loads((args.out/f"{a}-{s}"/"complete.json").read_text()) for s in manifest["seeds"] for a in ARMS})
    print("TRAINING COMPLETE",flush=True)


if __name__ == "__main__":
    main()
