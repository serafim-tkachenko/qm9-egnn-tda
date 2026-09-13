"""Verify exact perturbations and quantify descriptor shift in the completed run."""
import argparse
import json
from pathlib import Path
import numpy as np

from scripts.run_controlled_replication import geometry, topology_worker
from src.validation import file_hash, perturb, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--evaluation",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise ValueError("Analysis output must be new")
    for name,sha in json.loads((args.evaluation/"complete.json").read_text()).items():
        if file_hash(args.evaluation/name)!=sha:
            raise ValueError("Evaluation checksum mismatch")
    with np.load(args.run/"features.npz") as f:
        training={k:f[k][:104664] for k in ("tda","geometry")}
        mean={k:f[k+"_mean"] for k in training}
        scale={k:f[k+"_scale"] for k in training}
    with np.load(args.evaluation/"inputs-noise_subset-0.0-0.npz") as f:
        clean={k:f[k] for k in f.files}
    output={"source_sha256":file_hash(Path(__file__)),"features_sha256":file_hash(args.run/"features.npz"),
            "evaluation_complete_sha256":file_hash(args.evaluation/"complete.json"),"conditions":[]}
    checked=0
    for path in sorted(args.evaluation.glob("inputs-noise_subset-*.npz")):
        suffix=path.stem.removeprefix("inputs-noise_subset-")
        sigma,seed=suffix.split("-")
        sigma,seed=float(sigma),int(seed)
        with np.load(path) as f:
            values={k:f[k] for k in f.files}
        for k in ("ids","offsets","atomic_numbers","labels"):
            np.testing.assert_array_equal(values[k],clean[k])
        for j,i in enumerate(values["ids"]):
            sl=slice(values["offsets"][j],values["offsets"][j+1])
            expected=perturb(clean["coordinates"][sl],int(i),sigma,seed)[0]
            np.testing.assert_array_equal(expected,values["coordinates"][sl])
            np.testing.assert_array_equal(geometry(values["atomic_numbers"][sl],expected),values["geometry"][j])
            if j<8:
                np.testing.assert_array_equal(topology_worker(expected),values["tda"][j])
                checked+=1
        record={"sigma":sigma,"noise_seed":seed}
        for kind in training:
            x=values[kind].astype(np.float64)
            delta=(x-clean[kind])/scale[kind]
            z=(x-mean[kind])/scale[kind]
            outside=(x<training[kind].min(0)) | (x>training[kind].max(0))
            drift=np.sqrt(np.mean(delta**2,axis=0))
            top=np.argsort(drift)[-5:][::-1]
            record[kind]={"fraction_changed_molecules":float(np.any(x!=clean[kind],axis=1).mean()),
                          "mean_per_molecule_rms_standardized_change":float(np.sqrt(np.mean(delta**2,axis=1)).mean()),
                          "max_absolute_standardized_feature":float(np.abs(z).max()),
                          "fraction_entries_outside_training_range":float(outside.mean()),
                          "top_feature_rms_standardized_changes":[{"index":int(i),"rms":float(drift[i]),"training_scale":float(scale[kind][i])} for i in top]}
        output["conditions"].append(record)
    output["checks"]={"exact_coordinate_noise_and_geometry_rows":len(output["conditions"])*1024,
                      "recomputed_tda_vectors":checked,"fixed_membership_atoms_labels":True}
    write_json(args.out,output)
    print(json.dumps(output,indent=2))


if __name__=="__main__":
    main()
