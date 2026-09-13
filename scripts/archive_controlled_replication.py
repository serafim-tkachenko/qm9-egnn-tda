"""Verify and archive replication evidence; leave trained weights outside Git."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np

from scripts.diagnose_conditioning import fit_scaler
from src.validation import file_hash, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--evaluation",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--analysis",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise ValueError("Archive output must be new")
    manifest=json.loads((args.run/"manifest.json").read_text())
    completion=json.loads((args.run/"training_complete.json").read_text())
    preparation=json.loads((args.run/"preparation.json").read_text())
    if file_hash(args.run/"features.npz")!=preparation["features_sha256"]:
        raise ValueError("Training feature checksum mismatch")
    sources=dict(manifest["sources"])
    evaluation_manifest=json.loads((args.evaluation/"manifest.json").read_text())
    sources.update(evaluation_manifest["sources"])
    for path,sha in sources.items():
        if file_hash(path)!=sha:
            raise ValueError(f"Source changed since execution: {path}")
    histories={}
    orders={}
    for name,complete in completion.items():
        for file,key in (("best.pt","best_sha256"),("history.json","history_sha256")):
            if file_hash(args.run/name/file)!=complete[key]:
                raise ValueError(f"Training artifact mismatch: {name}/{file}")
        h=json.loads((args.run/name/"history.json").read_text())
        if [r["epoch"] for r in h]!=list(range(1,11)):
            raise ValueError("Expected all ten epochs")
        if min(r["val_mae_eV"] for r in h)!=complete["best_val_mae_eV"]:
            raise ValueError("Validation selection mismatch")
        seed=name.rsplit("-",1)[1]
        order=[r["order_sha256"] for r in h]
        if seed in orders and orders[seed]!=order:
            raise ValueError("Arms did not receive identical training order")
        orders[seed]=order
        histories[name]=h
    for name,sha in json.loads((args.evaluation/"complete.json").read_text()).items():
        if file_hash(args.evaluation/name)!=sha:
            raise ValueError(f"Evaluation artifact mismatch: {name}")
    (args.out/"training").mkdir(parents=True)
    shutil.copytree(args.evaluation,args.out/"evaluation")
    analysis=json.loads(args.analysis.read_text())
    if analysis["evaluation_complete_sha256"]!=file_hash(args.evaluation/"complete.json"):
        raise ValueError("Feature analysis is bound to another evaluation")
    shutil.copyfile(args.analysis,args.out/"feature_analysis.json")
    for name in ("manifest.json","preparation.json","training_complete.json"):
        shutil.copyfile(args.run/name,args.out/"training"/name)
    for name in completion:
        (args.out/"training"/name).mkdir()
        for file in ("complete.json","history.json"):
            shutil.copyfile(args.run/name/file,args.out/"training"/name/file)
    with np.load(args.run/"features.npz") as f:
        scaler={}
        for kind in ("tda","geometry"):
            mean,scale=fit_scaler(f[kind][:104664])
            np.testing.assert_array_equal(mean,f[kind+"_mean"])
            np.testing.assert_array_equal(scale,f[kind+"_scale"])
            scaler.update({kind+"_mean":mean,kind+"_scale":scale})
        np.savez_compressed(args.out/"training/scalers.npz",**scaler)
    write_json(args.out/"compute.json",dict(preprocessing_seconds=preparation["seconds"],
               summed_training_seconds=sum(r["training_seconds"] for h in histories.values() for r in h),
               summed_train_validation_seconds=sum(r["train_and_validation_seconds"] for h in histories.values() for r in h),
               evaluation_seconds=json.loads((args.evaluation/"summary.json").read_text())["wall_seconds"],
               gpu=manifest["gpu"],colab_runtime_used=False,
               timing_limit="Phase times exclude inter-epoch checkpoint I/O, some batch assembly and orchestration; concurrent software tests affected early timings"))
    write_json(args.out/"archive.json",dict(files={str(p.relative_to(args.out)).replace("\\","/"):file_hash(p) for p in args.out.rglob("*") if p.is_file()},
               external_artifacts="Best/resume checkpoints and full train/validation feature arrays remain in the local run directory; hashes are archived"))
    print("VERIFIED AND ARCHIVED",args.out)


if __name__=="__main__":
    main()
