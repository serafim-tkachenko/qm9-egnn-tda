"""Render the archived conditioning diagnostic without loading a checkpoint."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.results / "summary.json").read_text())
    with (args.results / "steps.csv").open() as f:
        steps = list(csv.DictReader(f))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    colors = {"raw": "#c05b38", "standardized": "#217a99"}
    for arm, color in colors.items():
        values = summary[arm]
        x = [0, 1]
        saturation = [100 * values[s]["saturation_fraction"] for s in ("initial", "after_epoch")]
        sensitivity = [values[s]["shuffled_mean_abs_prediction_change_eV"] for s in ("initial", "after_epoch")]
        axes[0].plot(x, saturation, "o-", label=arm.title(), color=color, linewidth=2)
        axes[1].plot(x, sensitivity, "o-", color=color, linewidth=2)
        curve = [r for r in steps if r["arm"] == arm]
        axes[2].plot([int(r["step"])+1 for r in curve], [float(r["mse_eV2"]) for r in curve], color=color, alpha=.85)
    axes[0].set(ylabel="Saturated outputs (%)", ylim=(-3, 103), title="Scaling prevents early saturation")
    axes[0].legend(frameon=False)
    axes[1].set(ylabel="Mean absolute prediction change (eV)", title="Shuffled-descriptor sensitivity")
    axes[2].set(xlabel="Optimizer step", ylabel="Batch training MSE (eV²)", yscale="log", title="One epoch; no convergence claim")
    for ax in axes[:2]:
        ax.set_xticks([0, 1], ["Initialization", "After one epoch"])
    for ax in axes:
        ax.grid(alpha=.18)
    fig.suptitle("4,096 training molecules · identical initial weights/order · 256 validation molecules", fontsize=12)
    fig.savefig(args.results / "conditioning.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
