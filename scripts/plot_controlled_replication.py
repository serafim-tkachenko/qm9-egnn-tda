"""Plot actual learning curves and held-out errors from an archived replication."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS={"egnn":"#555d6b","tda":"#217a99","geometry":"#c05b38","constant":"#8064a2"}
LABELS={"egnn":"EGNN","tda":"TDA fusion","geometry":"Simple geometry","constant":"Constant fusion"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,required=True)
    args=p.parse_args()
    root=args.results
    summary=json.loads((root/"evaluation/summary.json").read_text())
    manifest=json.loads((root/"training/manifest.json").read_text())
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout="constrained")
    for arm,color in COLORS.items():
        histories=[json.loads((root/f"training/{arm}-{seed}/history.json").read_text()) for seed in manifest["seeds"]]
        curves=np.array([[r["val_mae_eV"] for r in h] for h in histories])
        x=np.arange(1,11)
        axes[0].plot(x,curves.mean(0),color=color,label=LABELS[arm])
        axes[0].fill_between(x,curves.min(0),curves.max(0),color=color,alpha=.1)
        points=sorted([r for r in summary["mae"] if r["scope"]=="noise_subset" and r["condition"]=="matched" and r["arm"]==arm],key=lambda r:r["sigma"])
        sigmas=[r["sigma"] for r in points]
        seed_values=np.array([list(r["seed_mae_eV"].values()) for r in points])
        axes[1].plot(sigmas,seed_values.mean(1),"o-",color=color)
        axes[1].fill_between(sigmas,seed_values.min(1),seed_values.max(1),color=color,alpha=.1)
        if arm != "egnn":
            sats=np.array([[r["saturation_fraction"] for r in h] for h in histories])*100
            axes[2].plot(x,sats.mean(0),color=color)
            axes[2].fill_between(x,sats.min(0),sats.max(0),color=color,alpha=.1)
    axes[0].set(xlabel="Training epoch",ylabel="Full-validation MAE (eV)",title="Learning curves: 3 seeds per arm")
    axes[0].legend(frameon=False,fontsize=9)
    axes[1].set(xlabel="Coordinate noise σ (Å)",ylabel="Test-subset MAE (eV)",title="1,024 paired molecules; matched inputs")
    axes[2].set(xlabel="Training epoch",ylabel="Saturated FiLM outputs (%)",ylim=(-2,102),title="Conditioning on 256 validation molecules")
    for ax in axes:
        ax.grid(alpha=.18)
    fig.suptitle("Controlled QM9 replication · bands show training-seed ranges, not confidence intervals",fontsize=12)
    fig.savefig(root/"replication.png",dpi=180)
    plt.close(fig)
    if (root/"archive.json").exists():
        from src.validation import file_hash, write_json
        archive=json.loads((root/"archive.json").read_text())
        archive["files"]["replication.png"]=file_hash(root/"replication.png")
        archive["figure_script_sha256"]=file_hash(Path(__file__))
        write_json(root/"archive.json",archive)


if __name__=="__main__":
    main()
