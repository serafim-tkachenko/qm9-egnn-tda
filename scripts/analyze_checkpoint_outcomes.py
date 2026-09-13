"""Describe saved checkpoint outcomes without selecting or refitting models."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", default="results/paired_pilot_2026-09-13")
    p.add_argument("--full-val", default="outputs/full-clean-val")
    p.add_argument("--full-test", default="outputs/full-clean-test")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root, out = Path(args.pilot), Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    complete = json.loads((root / "complete.json").read_text())
    if sha(root / "predictions.csv") != complete["predictions_sha256"]:
        raise ValueError("Pilot prediction checksum mismatch")
    with (root / "predictions.csv").open() as f:
        rows = list(csv.DictReader(f))
    conditions = []
    for sigma in sorted({float(r["sigma"]) for r in rows}):
        group = [r for r in rows if float(r["sigma"]) == sigma]
        if len({r["molecule_id"] for r in group}) != len(group):
            raise ValueError("This descriptive analysis expects one noise realization per molecule")
        delta = np.array([abs(float(r["fusion_perturbed"])-float(r["target"]))
                          - abs(float(r["egnn"])-float(r["target"])) for r in group])
        conditions.append({"sigma": sigma, "n": len(group),
                           "fusion_lower_error": int((delta < 0).sum()),
                           "egnn_lower_error": int((delta > 0).sum()), "ties": int((delta == 0).sum()),
                           "mean_delta_ev": float(delta.mean()), "median_delta_ev": float(np.median(delta)),
                           "delta_quantiles_0_10_50_90_100_ev": np.quantile(delta, [0, .1, .5, .9, 1]).tolist(),
                           "mean_delta_without_largest_absolute_pair_ev": float(np.delete(delta, np.argmax(abs(delta))).mean())})
    reproduction = json.loads((root / "full-clean-reproduction.json").read_text())
    full = []
    for split, directory in (("val", args.full_val), ("test", args.full_test)):
        path = Path(directory) / "summary.json"
        expected = next(r for r in reproduction["splits"] if r["split"] == split)
        if sha(path) != expected["artifacts_sha256"]["summary.json"]:
            raise ValueError("Full-clean summary checksum mismatch")
        summary = json.loads(path.read_text())["conditions"][0]
        full.append({"split": split, "n": summary["n_molecules"],
                     "fusion_minus_egnn_ev": summary["fusion_clean_minus_egnn"],
                     "ci95_ev": summary["fusion_clean_minus_egnn_ci95"], "source_summary_sha256": sha(path)})
    result = {"scope": "Post-pilot descriptive analysis; no fitting, filtering of primary results, or new model selection",
              "delta_sign": "fusion matched-input absolute error minus EGNN absolute error",
              "sensitivity_limit": "Deleting the largest absolute pair is an influence diagnostic only; primary estimates retain every molecule",
              "interval_limit": "Existing molecule-bootstrap intervals condition on selected checkpoints; validation was used for checkpoint selection",
              "predictions_sha256": complete["predictions_sha256"], "pilot": conditions, "full_clean": full}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
