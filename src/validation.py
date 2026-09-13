"""Scientific checks shared by the paired checkpoint diagnostic.

The historical evaluator/cache remain available for reproduction. New runs use
explicit identities and never silently reinterpret an old numeric-ID cache.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from src.data.tda_features import TDACache, TDAConfig


DESCRIPTOR_VERSION = "legacy-adaptive-betti-unit-diameter-v1"
NOISE_VERSION = "sha256-pcg64-f32-v1"


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(value) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(digest_json({"shape": a.shape, "dtype": a.dtype.str}).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def versions():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "torch-geometric", "egnn-pytorch", "giotto-tda", "numpy",
             "scipy", "scikit-learn")}


def descriptor_spec():
    packages = versions()
    return {
        "version": DESCRIPTOR_VERSION, "betti_bins": 64, "max_homology_dim": 1,
        "normalization": "float32-center-then-unit-diameter",
        "grid": "fit-separately-per-molecule-and-homology-dimension",
        "entropy": "gtda-default-normalize-False-nan_fill_value-minus1",
        "implementation_sha256": file_hash(Path(__file__).parent / "data/tda_features.py"),
        "packages": {k: packages[k] for k in
                     ("giotto-tda", "numpy", "scipy", "scikit-learn")},
    }


def perturb(coords, molecule_id: int, sigma: float, noise_seed: int):
    """One CPU stream per molecule/replicate/sigma, unrelated to batch padding.

    Float32 perturbations are materialized once and used by both models and TDA.
    The serialized inputs, rather than library RNG promises, anchor reproduction.
    """
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) == 0:
        raise ValueError("Expected a nonempty, unpadded (N,3) point cloud")
    if not np.isfinite(coords).all() or not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Coordinates and sigma must be finite; sigma >= 0")
    identity = {"version": NOISE_VERSION, "molecule_id": int(molecule_id),
                "noise_seed": int(noise_seed), "sigma_hex": float(sigma).hex()}
    noise_id = digest_json(identity)
    rng = np.random.Generator(np.random.PCG64(int(noise_id[:32], 16)))
    noise = rng.standard_normal(coords.shape).astype(np.float32) * np.float32(sigma)
    noisy = coords.copy() if sigma == 0 else coords + noise
    return noisy, noise_id


class CheckedTDACache:
    """New cache namespace binds dataset bytes, descriptor and exact coordinates."""

    def __init__(self, root, dataset_sha256: str, spec=None):
        self.root = Path(root)
        self.spec = descriptor_spec() if spec is None else spec
        self.manifest = {"schema": 1, "dataset_sha256": dataset_sha256,
                         "descriptor": self.spec}
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "manifest.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != self.manifest:
                raise ValueError("TDA cache/config/dataset mismatch; choose a new directory")
        elif any(self.root.iterdir()):
            raise ValueError("Refusing an unversioned/nonempty TDA cache")
        else:
            write_json(path, self.manifest)
        self.computer = TDACache(TDAConfig(cache_dir=str(self.root), n_jobs=1))

    def get(self, molecule_id, coords):
        coords = np.asarray(coords, dtype=np.float32)
        coords_sha = array_hash(coords)
        identity = {"molecule_id": int(molecule_id), "coordinates_sha256": coords_sha,
                    "manifest_sha256": digest_json(self.manifest)}
        key = digest_json(identity)
        path = self.root / f"{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                vec = saved["vector"]
                metadata = json.loads(str(saved["metadata"]))
            if metadata != {**identity, "vector_sha256": array_hash(vec)}:
                raise ValueError(f"Corrupt or incompatible TDA entry: {path}")
        else:
            vec = self.computer.compute_vec(coords)
            metadata = {**identity, "vector_sha256": array_hash(vec)}
            np.savez_compressed(path, vector=vec, metadata=json.dumps(metadata))
        if vec.shape != (130,) or vec.dtype != np.float32 or not np.isfinite(vec).all():
            raise ValueError(f"Invalid descriptor for molecule {molecule_id}")
        return vec, metadata


def checked_state_dict(model, path, expected_sha256):
    if file_hash(path) != expected_sha256:
        raise ValueError(f"Checkpoint hash mismatch: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    # Legacy checkpoints contain only weights. Strict shape/key validation cannot
    # establish missing training provenance; the run manifest records that limit.
    if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a plain legacy tensor state_dict")
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Checkpoint contains nonfinite parameters")
    model.load_state_dict(state, strict=True)


def paired_summary(rows, bootstrap_seed=20260913, resamples=4000):
    """Pair by molecule and noise replicate, bootstrap molecules as clusters.

    With multiple noise replicates, average each molecule first. These intervals
    condition on the two checkpoints and do not estimate training-seed variation.
    """
    result = []
    if not rows:
        raise ValueError("No predictions")
    for sigma in sorted({float(r["sigma"]) for r in rows}):
        selected = [r for r in rows if float(r["sigma"]) == sigma]
        ids = sorted({int(r["molecule_id"]) for r in selected})
        replicates = sorted({int(r["noise_seed"]) for r in selected})
        lookup = {}
        for r in selected:
            key = (int(r["molecule_id"]), int(r["noise_seed"]))
            if key in lookup:
                raise ValueError("Duplicate molecule/noise replicate")
            lookup[key] = r
        if len(lookup) != len(ids) * len(replicates):
            raise ValueError("Missing paired molecule/noise replicates")
        errors = np.array([[[abs(float(lookup[i, seed][name]) - float(lookup[i, seed]["target"]))
                             for name in ("egnn", "fusion_clean", "fusion_perturbed")]
                            for seed in replicates] for i in ids], dtype=np.float64).mean(axis=1)
        if not np.isfinite(errors).all():
            raise ValueError("Nonfinite prediction/target")
        rng = np.random.default_rng(bootstrap_seed)
        # Bound memory even for retrospective full-test runs.
        boots = np.array([errors[rng.integers(0, len(ids), len(ids))].mean(axis=0)
                          for _ in range(resamples)])
        means = errors.mean(axis=0)
        item = {"sigma": sigma, "n_molecules": len(ids), "n_noise_seeds": len(replicates)}
        for j, name in enumerate(("egnn", "fusion_clean", "fusion_perturbed")):
            item[name + "_mae"] = float(means[j])
            item[name + "_mae_ci95"] = np.quantile(boots[:, j], [.025, .975]).tolist()
        for j, name in ((1, "fusion_clean"), (2, "fusion_perturbed")):
            item[name + "_minus_egnn"] = float(means[j] - means[0])
            item[name + "_minus_egnn_ci95"] = np.quantile(boots[:, j] - boots[:, 0], [.025, .975]).tolist()
        item["perturbed_minus_clean"] = float(means[2] - means[1])
        item["perturbed_minus_clean_ci95"] = np.quantile(boots[:, 2] - boots[:, 1], [.025, .975]).tolist()
        advantage = means[0] - means[1]
        item["fraction_of_paired_clean_topology_advantage_remaining"] = (
            float((means[0] - means[2]) / advantage) if advantage > 0 else None)
        result.append(item)
    return result
