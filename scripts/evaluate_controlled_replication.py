"""Evaluate frozen replication checkpoints on shared clean/noisy inputs."""
from __future__ import annotations
import argparse
import csv
import gzip
import io
import json
import os
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from scripts.run_controlled_replication import ARMS, geometry, models_for_seed, topology_worker
from scripts.diagnose_conditioning import transform
from src.eval_paired import load_processed, frozen_split
from src.data.collate import qm9_dense_collate
from src.validation import file_hash, perturb, write_json


def clustered_interval(differences):
    """One value per molecule, already averaged over noise and model seeds."""
    x=np.asarray(differences,dtype=np.float64)
    rng=np.random.default_rng(20260913)
    means=[]
    for _ in range(20):
        means.extend(x[rng.integers(0,len(x),size=(200,len(x)))].mean(axis=1))
    return np.quantile(means,[.025,.975]).tolist()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--split",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--workers",type=int,default=4)
    args=p.parse_args()
    if args.out.exists():
        raise ValueError("Evaluation output must be new")
    args.out.mkdir(parents=True)
    started=time.perf_counter()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((args.run/"manifest.json").read_text())
    complete=json.loads((args.run/"training_complete.json").read_text())
    if file_hash(args.data)!=manifest["dataset_sha256"]:
        raise ValueError("Dataset identity mismatch")
    ds=load_processed(args.data)
    split=frozen_split(args.split,manifest["dataset_sha256"],ds)
    if split["test"]!=manifest["clean_test_ids"] or split["test"][:1024]!=manifest["noise_test_ids"]:
        raise ValueError("Evaluation membership mismatch")
    with np.load(args.run/"features.npz") as f:
        scalers={k:(f[k+"_mean"],f[k+"_scale"]) for k in ("tda","geometry")}
    features_hash=file_hash(args.run/"features.npz")
    train_manifest_hash=file_hash(args.run/"manifest.json")
    models={}
    for seed in manifest["seeds"]:
        for arm,model in models_for_seed(seed).items():
            name=f"{arm}-{seed}"
            path=args.run/name/"best.pt"
            if file_hash(path)!=complete[name]["best_sha256"]:
                raise ValueError("Checkpoint hash mismatch")
            ck=torch.load(path,map_location="cpu",weights_only=True)
            if ck["manifest_sha256"]!=train_manifest_hash or ck["features_sha256"]!=features_hash:
                raise ValueError("Checkpoint preprocessing mismatch")
            if ck["arm"]!=arm or ck["seed"]!=seed:
                raise ValueError("Checkpoint identity mismatch")
            model.load_state_dict(ck["model"],strict=True)
            models[(seed,arm)]=model.cuda().eval()
    write_json(args.out/"manifest.json",dict(training_manifest_sha256=train_manifest_hash,
               checkpoints={k:v["best_sha256"] for k,v in complete.items()},
               sources={f:file_hash(f) for f in ("scripts/evaluate_controlled_replication.py","scripts/run_controlled_replication.py")},
               bootstrap="4000 molecule resamples after averaging noise replicates and fixed model seeds; does not estimate training-population uncertainty",
               clean_test_ids=manifest["clean_test_ids"],noise_test_ids=manifest["noise_test_ids"],
               criterion=manifest["practical_criterion"]))
    # Full clean test features, then exact same selected molecules at each noise condition.
    ids=manifest["clean_test_ids"]
    clean_pos=[ds[i].pos.numpy() for i in ids]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        clean_tda=np.stack(list(pool.map(topology_worker,clean_pos,chunksize=128)))
    clean_geometry=np.stack([geometry(ds[i].z.numpy(),pos) for i,pos in zip(ids,clean_pos)])
    clean_lookup={i:j for j,i in enumerate(ids)}
    rows=[]
    diagnostics=[]
    evaluations=[("full_clean",0.,0,ids,clean_pos,clean_tda,clean_geometry)]

    @torch.inference_mode()
    def evaluate(scope,sigma,noise_seed,ids,coords,tda,geom):
        clean_idx=[clean_lookup[i] for i in ids]
        tx=transform(tda,*scalers["tda"])
        gx=transform(geom,*scalers["geometry"])
        ctx=transform(clean_tda[clean_idx],*scalers["tda"])
        cgx=transform(clean_geometry[clean_idx],*scalers["geometry"])
        # Save exact unpadded noisy coordinates, descriptor values, membership and labels.
        sizes=np.array([len(c) for c in coords])
        offsets=np.r_[0,np.cumsum(sizes)]
        np.savez_compressed(args.out/f"inputs-{scope}-{sigma}-{noise_seed}.npz",ids=ids,
                            offsets=offsets,coordinates=np.concatenate(coords),tda=tda,geometry=geom,
                            atomic_numbers=np.concatenate([ds[i].z.numpy() for i in ids]),
                            labels=np.array([ds[i].y.view(-1)[4].item() for i in ids]))
        shuffle=np.random.default_rng(20260913).permutation(len(ids))
        sensitivity={s:[] for s in manifest["seeds"]}
        saturation={s:[] for s in manifest["seeds"]}
        for start in range(0,len(ids),64):
            sl=slice(start,start+64)
            batch_ids=ids[sl]
            values=[]
            for i,pos in zip(batch_ids,coords[sl]):
                d=ds[i].clone()
                d.pos=torch.from_numpy(pos)
                values.append((i,d))
            b=qm9_dense_collate(values)
            z,pos,mask=(getattr(b,k).cuda() for k in ("z","pos","mask"))
            for (seed,arm),model in models.items():
                t=None if arm=="egnn" else torch.from_numpy(tx[sl] if arm=="tda" else gx[sl] if arm=="geometry" else np.zeros((len(batch_ids),130),np.float32)).cuda()
                pred=model(z,pos,mask) if t is None else model(z,pos,mask,t)
                if not torch.isfinite(pred).all():
                    raise ValueError("Nonfinite prediction")
                conditions={"matched":pred}
                if sigma>0 and arm in ("tda","geometry"):
                    conditions["clean_auxiliary"]=model(z,pos,mask,torch.from_numpy(ctx[sl] if arm=="tda" else cgx[sl]).cuda())
                if arm=="tda":
                    shuffled=model(z,pos,mask,torch.from_numpy(tx[shuffle][sl]).cuda())
                    sensitivity[seed].extend((pred-shuffled).abs().cpu().tolist())
                    saturation[seed].extend((model.film(t).tanh().abs()>.9999).float().mean(1).cpu().tolist())
                if start==0:
                    transformed=model(z.flip(1),pos.flip(1)[...,[1,2,0]]+2,mask.flip(1)) if t is None else model(z.flip(1),pos.flip(1)[...,[1,2,0]]+2,mask.flip(1),t)
                    err=(pred-transformed).abs().max().item()
                    padded=(torch.nn.functional.pad(z,(0,3)),torch.nn.functional.pad(pos,(0,0,0,3)),torch.nn.functional.pad(mask,(0,3)))
                    padded_pred=model(*padded) if t is None else model(*padded,t)
                    padding_err=(pred-padded_pred).abs().max().item()
                    if max(err,padding_err)>2e-5:
                        raise ValueError(f"Symmetry/padding failure {arm}/{seed}: {err}, {padding_err}")
                    diagnostics.append(dict(scope=scope,sigma=sigma,noise_seed=noise_seed,arm=arm,seed=seed,symmetry_max_abs_error_eV=err,padding_max_abs_error_eV=padding_err))
                for condition,pr in conditions.items():
                    if not torch.isfinite(pr).all():
                        raise ValueError("Nonfinite auxiliary prediction")
                    for i,target,value in zip(batch_ids,b.y.tolist(),pr.cpu().tolist()):
                        rows.append(dict(scope=scope,sigma=sigma,noise_seed=noise_seed,seed=seed,arm=arm,condition=condition,molecule_id=i,target_eV=target,prediction_eV=value))
        for seed in manifest["seeds"]:
            diagnostics.append(dict(scope=scope,sigma=sigma,noise_seed=noise_seed,arm="tda",seed=seed,
                               saturation_fraction=float(np.mean(saturation[seed])),shuffled_mean_abs_prediction_change_eV=float(np.mean(sensitivity[seed]))))
        print("EVALUATED",scope,sigma,noise_seed,len(ids),flush=True)

    for values in evaluations:
        evaluate(*values)
    noise_ids=manifest["noise_test_ids"]
    idx=[clean_lookup[i] for i in noise_ids]
    evaluate("noise_subset",0.,0,noise_ids,[clean_pos[j] for j in idx],clean_tda[idx],clean_geometry[idx])
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for sigma in manifest["noise_sigmas"][1:]:
            for noise_seed in manifest["noise_seeds"]:
                coords=[perturb(ds[i].pos.numpy(),i,sigma,noise_seed)[0] for i in noise_ids]
                tda=np.stack(list(pool.map(topology_worker,coords,chunksize=128)))
                geom=np.stack([geometry(ds[i].z.numpy(),pos) for i,pos in zip(noise_ids,coords)])
                evaluate("noise_subset",sigma,noise_seed,noise_ids,coords,tda,geom)
    with (args.out/"predictions.csv.gz").open("wb") as raw:
        with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0) as zipped:
            with io.TextIOWrapper(zipped,encoding="utf-8",newline="") as text:
                writer=csv.DictWriter(text,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    write_json(args.out/"diagnostics.json",diagnostics)
    summary=summarize(rows,manifest["seeds"])
    summary["wall_seconds"]=time.perf_counter()-started
    write_json(args.out/"summary.json",summary)
    write_json(args.out/"complete.json",{p.name:file_hash(p) for p in args.out.iterdir() if p.is_file()})
    print("EVALUATION COMPLETE",summary["criterion_met"],summary["wall_seconds"],flush=True)


def summarize(rows,seeds):
    lookup={}
    for r in rows:
        key=tuple(r[k] for k in ("scope","sigma","noise_seed","seed","arm","condition","molecule_id"))
        if key in lookup:
            raise ValueError("Duplicate prediction identity")
        lookup[key]=abs(float(r["prediction_eV"])-float(r["target_eV"]))
    groups={}
    for scope,sigma in {(k[0],k[1]) for k in lookup}:
        subset=[k for k in lookup if k[0]==scope and k[1]==sigma]
        ids={k[6] for k in subset}
        replicates={0} if sigma==0 else {20260913,20260914,20260915}
        expected={(scope,sigma,n,s,a,c,i) for n in replicates for s in seeds for a in ARMS
                  for c in (["matched","clean_auxiliary"] if sigma>0 and a in ("tda","geometry") else ["matched"])
                  for i in ids}
        if set(subset)!=expected:
            raise ValueError("Incomplete model/molecule/noise/condition grid")
    for key,error in lookup.items():
        scope,sigma,noise_seed,seed,arm,condition,i=key
        groups.setdefault((scope,sigma,arm,condition),{}).setdefault((seed,i),[]).append(error)
    means,comparisons=[],[]
    for (scope,sigma,arm,condition),values in groups.items():
        seed_mae={str(s):float(np.mean([np.mean(v) for (seed,_),v in values.items() if seed==s])) for s in seeds}
        means.append(dict(scope=scope,sigma=sigma,arm=arm,condition=condition,seed_mae_eV=seed_mae,
                          mean_mae_eV=float(np.mean(list(seed_mae.values()))),seed_sd_eV=float(np.std(list(seed_mae.values()),ddof=1))))
    for scope,sigma in sorted({(r["scope"],r["sigma"]) for r in rows}):
        tda=groups[(scope,sigma,"tda","matched")]
        for arm in ("egnn","geometry","constant"):
            control=groups[(scope,sigma,arm,"matched")]
            if set(tda)!=set(control):
                raise ValueError("Unpaired model/molecule grid")
            differences={k:float(np.mean(v)-np.mean(control[k])) for k,v in tda.items()}
            ids=sorted({i for _,i in differences})
            molecule_means=[np.mean([differences[(seed,i)] for seed in seeds]) for i in ids]
            seed_diff={str(s):float(np.mean([v for (seed,_),v in differences.items() if seed==s])) for s in seeds}
            control_mean=np.mean([np.mean(v) for v in control.values()])
            comparisons.append(dict(scope=scope,sigma=sigma,control=arm,mean_difference_eV=float(np.mean(molecule_means)),
                               seed_differences_eV=seed_diff,relative_improvement=float(-np.mean(molecule_means)/control_mean),
                               fixed_models_molecule_ci95_eV=clustered_interval(molecule_means)))
    primary=[r for r in comparisons if r["scope"]=="noise_subset" and r["sigma"]==.1]
    return dict(mae=means,comparisons=comparisons,criterion_met=len(primary)==3 and all(r["relative_improvement"]>=.01 and all(v<0 for v in r["seed_differences_eV"].values()) for r in primary),
                prediction_rows=len(rows),uncertainty="molecule intervals condition on these 12 checkpoints; seed SD/differences reported separately")


if __name__=="__main__":
    main()
