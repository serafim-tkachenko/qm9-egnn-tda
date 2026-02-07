from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


@dataclass
class History:
    epochs: List[int]
    train_loss: List[float]
    train_mae: List[float]
    val_mae: List[float]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_history_json(path: str | Path, history: History, extra: Optional[Dict] = None) -> None:
    payload = {
        "epochs": history.epochs,
        "train_loss": history.train_loss,
        "train_mae": history.train_mae,
        "val_mae": history.val_mae,
    }
    if extra:
        payload.update(extra)

    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def save_summary_csv(path: str | Path, rows: List[Dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        raise ValueError("save_summary_csv: rows is empty")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_loss(history: History, out_path: str | Path, title: str) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    plt.figure()
    plt.plot(history.epochs, history.train_loss, marker="o")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title(title)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_mae(history: History, out_path: str | Path, title: str) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    plt.figure()
    plt.plot(history.epochs, history.train_mae, marker="o", label="train MAE")
    plt.plot(history.epochs, history.val_mae, marker="o", label="val MAE")
    plt.xlabel("epoch")
    plt.ylabel("MAE")
    plt.title(title)
    plt.legend()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
