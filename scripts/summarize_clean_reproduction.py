"""Check full clean validation/test outputs against preserved historical MAEs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.validation import file_hash, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--historical", default="results/compare_metrics.json")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    historical = json.loads(Path(args.historical).read_text())
    results = []
    for split, directory in (("val", args.validation), ("test", args.test)):
        root = Path(directory)
        manifest = json.loads((root / "manifest.json").read_text())
        complete = json.loads((root / "complete.json").read_text())
        summary = json.loads((root / "summary.json").read_text())["conditions"]
        config = manifest["arguments"]
        if config["split"] != split or config["size"] != 0 or config["sigmas"] != [0.]:
            raise ValueError("Expected a full, sigma-zero evaluation of the indicated split")
        for name, ext in (("predictions", ".csv"), ("inputs", ".npz")):
            if file_hash(root / (name + ext)) != complete[name + "_sha256"]:
                raise ValueError("Full-clean output checksum mismatch")
        with (root / "predictions.csv").open() as f:
            rows = list(csv.DictReader(f))
        if len(rows) != len(manifest["subset_ids"]) or [int(r["molecule_id"]) for r in rows] != manifest["subset_ids"]:
            raise ValueError("Expected each full-split molecule exactly once, in frozen order")
        metrics = {}
        for model, arm in (("baseline", "egnn"), ("fusion", "fusion_clean")):
            actual = float(np.mean([abs(float(r[arm])-float(r["target"])) for r in rows]))
            if abs(actual - summary[0][arm + "_mae"]) > 1e-12:
                raise ValueError("Summary disagrees with underlying predictions")
            saved = historical[model][split + "_mae"]
            metrics[model] = {"historical_mae_ev": saved, "reproduced_mae_ev": actual,
                              "absolute_difference_ev": abs(actual-saved)}
        results.append({"split": split, "n_molecules": len(rows), "metrics": metrics,
                        "compute": json.loads((root / "compute.json").read_text()),
                        "dataset_sha256": manifest["dataset_sha256"],
                        "split_manifest_sha256": manifest["split_manifest_sha256"],
                        "subset_sha256": manifest["subset_sha256"],
                        "checkpoint_sha256": {k: manifest["checkpoint_provenance"][k]["sha256"] for k in ("baseline", "fusion")},
                        "packages": manifest["packages"], "batch_size": config["batch_size"],
                        "precision": manifest["precision"], "device": config["device"],
                        "source_revision": manifest["git_revision"],
                        "artifacts_sha256": {name: file_hash(root / name) for name in ("manifest.json", "summary.json", "predictions.csv", "inputs.npz")}})
    result = {"purpose": "Retrospective clean reproduction; recomputed legacy descriptors; no retraining",
              "historical_metrics_sha256": file_hash(args.historical),
              "absolute_tolerance_ev": 2e-6, "splits": results,
              "passed": all(m["absolute_difference_ev"] <= 2e-6 for r in results for m in r["metrics"].values())}
    write_json(out, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise ValueError("Historical clean metric discrepancy exceeds tolerance")


if __name__ == "__main__":
    main()
