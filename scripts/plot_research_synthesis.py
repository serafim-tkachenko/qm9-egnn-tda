"""Plot cited clean-QM9 reference values separately from our archived experiment."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/research_synthesis_2026-09-13"
SOURCE = ROOT / "results/controlled_replication_2026-09-13/evaluation/summary.json"


def main():
    references = json.loads((OUT / "references.json").read_text())
    summary = json.loads(SOURCE.read_text())
    points = references["points"]
    arms = ["egnn", "tda", "geometry", "constant"]
    labels = ["Our EGNN variant", "Standardized TDA", "Simple geometry", "Constant fusion"]
    rows = [next(r for r in summary["mae"] if r["scope"] == "full_clean"
                 and r["condition"] == "matched" and r["arm"] == arm) for arm in arms]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.7), layout="constrained")
    for i, point in enumerate(points):
        axes[0].plot(point["mae"], i, "o", color="#46667e", markersize=8)
        axes[0].annotate(str(point["mae"]), (point["mae"] + 8, i), va="center")
    axes[0].set(yticks=range(len(points)), yticklabels=[p["label"] for p in points],
                title="Published reference values\nDifferent protocols; uncertainty not supplied")
    for i, (row, color) in enumerate(zip(rows, ["#555d6b", "#217a99", "#c05b38", "#8064a2"])):
        mean, sd = row["mean_mae_eV"] * 1000, row["seed_sd_eV"] * 1000
        axes[1].errorbar(mean, i, xerr=sd, fmt="o", color=color, capsize=5)
        axes[1].annotate(f"{mean:.1f} ± {sd:.1f}", (mean, i), xytext=(0, 12),
                         textcoords="offset points", ha="center", fontsize=10)
    axes[1].set(yticks=range(4), yticklabels=labels,
                title="Our controlled run: 10 epochs\n13,084 test molecules; mean ± seed SD")
    for ax in axes:
        ax.set_xlim(0, 310)
        ax.set_ylim(-0.65, 4.65)
        ax.invert_yaxis()
        ax.set_xlabel("Clean gap MAE (meV; lower is better)")
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("QM9 context · equal units do not imply equal experimental conditions", fontsize=14)
    fig.savefig(OUT / "comparison.png", dpi=160)
    plt.close(fig)
    paths = [OUT / "references.json", SOURCE, Path(__file__), OUT / "comparison.png"]
    manifest = {"files": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in paths}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
