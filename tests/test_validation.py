import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch_geometric.data import Data, InMemoryDataset

from src.data.collate import qm9_dense_collate
from src.data.tda_features import TDACache, TDAConfig
from src.eval_paired import frozen_split, predict
from src.models.egnn_gap import EGNNGapRegressor
from src.models.fusion_gap import EGNNTDAFiLMRegressor
from src.validation import (CheckedTDACache, array_hash, checked_state_dict,
                            DESCRIPTOR_VERSION, file_hash, paired_summary, perturb)

torch.set_num_threads(2)


def molecule(n, seed):
    rng = np.random.default_rng(seed)
    return Data(z=torch.tensor(([6, 1, 7, 8] * n)[:n]),
                pos=torch.from_numpy(rng.normal(size=(n, 3)).astype(np.float32)),
                y=torch.arange(19, dtype=torch.float32).reshape(1, -1))


def test_noise_pairing_order_padding_and_zero():
    molecules = [(17, molecule(4, 1)), (900, molecule(9, 2))]
    forward = {idx: perturb(d.pos, idx, .1, 5) for idx, d in molecules}
    torch.randn(117)  # Execution order and unrelated global RNG have no effect.
    for idx, data in reversed(molecules):
        noisy, noise_id = perturb(data.pos, idx, .1, 5)
        np.testing.assert_array_equal(noisy, forward[idx][0])
        assert noise_id == forward[idx][1]
        np.testing.assert_array_equal(perturb(data.pos, idx, 0, 5)[0], data.pos)
        assert not np.array_equal(noisy, perturb(data.pos, idx, .1, 6)[0])
    transformed = [(idx, Data(z=d.z, pos=torch.from_numpy(forward[idx][0]), y=d.y))
                   for idx, d in molecules]
    together = qm9_dense_collate(transformed)
    alone = qm9_dense_collate(transformed[:1])
    torch.testing.assert_close(together.pos[0, :4], alone.pos[0], rtol=0, atol=0)
    assert torch.count_nonzero(together.pos[0, 4:]) == 0
    assert together.y.tolist() == [4, 4]


def test_cache_recomputes_exact_noisy_coordinates_and_rejects_mismatch(tmp_path):
    cache = CheckedTDACache(tmp_path / "cache", "dataset-A")
    coords = molecule(8, 4).pos.numpy()
    noisy, _ = perturb(coords, 7, .1, 8)
    v, _ = cache.get(7, coords)
    nv, meta = cache.get(7, noisy)
    np.testing.assert_array_equal(nv, cache.computer.compute_vec(noisy))
    assert meta["coordinates_sha256"] == array_hash(noisy)
    assert not np.array_equal(v, nv)
    assert len(list(cache.root.glob("*.npz"))) == 2
    np.testing.assert_array_equal(cache.get(7, coords.copy())[0], v)
    with pytest.raises(ValueError, match="mismatch"):
        CheckedTDACache(cache.root, "dataset-B")
    with pytest.raises(ValueError, match="mismatch"):
        CheckedTDACache(cache.root, "dataset-A", {"version": "changed-grid"})
    (tmp_path / "legacy").mkdir()
    np.save(tmp_path / "legacy/000007.npy", v)
    with pytest.raises(ValueError, match="unversioned"):
        CheckedTDACache(tmp_path / "legacy", "dataset-A")


def test_cache_detects_tampering(tmp_path):
    cache = CheckedTDACache(tmp_path, "A")
    coords = molecule(5, 4).pos.numpy()
    cache.get(8, coords)
    path = next(tmp_path.glob("*.npz"))
    with np.load(path) as saved:
        vec, metadata = saved["vector"].copy(), saved["metadata"].copy()
    vec[0] += 1
    np.savez_compressed(path, vector=vec, metadata=metadata)
    with pytest.raises(ValueError, match="Corrupt"):
        cache.get(8, coords)


def test_empty_h1_and_degenerate_diagrams_are_finite(tmp_path):
    computer = TDACache(TDAConfig(cache_dir=str(tmp_path), n_jobs=1))
    for coords in (np.zeros((1, 3)), np.zeros((4, 3)),
                   np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])):
        vec = computer.compute_vec(coords)
        assert vec.shape == (130,) and np.isfinite(vec).all()
        assert np.count_nonzero(vec[64:128]) == 0
        assert vec[-1] == -1  # Historical giotto empty-diagram entropy sentinel.
    for coords in (np.zeros((0, 3)), np.full((4, 3), np.nan)):
        with pytest.raises(ValueError):
            computer.compute_vec(coords)


def test_model_symmetries_masks_and_batch_size(tmp_path):
    torch.manual_seed(7)
    models = [EGNNGapRegressor(emb_dim=16, depth=2, mlp_hidden=32).eval(),
              EGNNTDAFiLMRegressor(tda_dim=130, emb_dim=16, depth=2,
                                   film_hidden=32, head_hidden=32).eval()]
    computer = TDACache(TDAConfig(cache_dir=str(tmp_path), n_jobs=1))
    d = molecule(8, 12)
    vec = computer.compute_vec(d.pos.numpy())[None]
    original = predict(models, qm9_dense_collate([(2, d)]), vec, vec, "cpu")
    q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
    for transform in (lambda x: x + torch.tensor([1., -2., 3.]),
                      lambda x: x @ torch.tensor(q, dtype=torch.float32),
                      lambda x: x * torch.tensor([-1., 1., 1.])):
        modified = d.clone()
        modified.pos = transform(d.pos)
        v = computer.compute_vec(modified.pos.numpy())[None]
        np.testing.assert_allclose(v, vec, rtol=1e-5, atol=1e-5)
        pred = predict(models, qm9_dense_collate([(2, modified)]), v, v, "cpu")
        np.testing.assert_allclose(pred, original, rtol=1e-4, atol=2e-5)
    perm = torch.tensor([2, 7, 4, 3, 1, 0, 6, 5])
    permuted = Data(z=d.z[perm], pos=d.pos[perm], y=d.y)
    vp = computer.compute_vec(permuted.pos.numpy())[None]
    np.testing.assert_allclose(vp, vec, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(predict(models, qm9_dense_collate([(2, permuted)]), vp, vp, "cpu"),
                               original, rtol=1e-4, atol=2e-5)
    other = molecule(12, 11)
    vectors = np.concatenate([vec, computer.compute_vec(other.pos.numpy())[None]])
    batch = qm9_dense_collate([(2, d), (4, other)])
    batch.pos[0, 8:] = 1234  # Explicitly check masked coordinates cannot affect the output.
    together = predict(models, batch, vectors, vectors, "cpu")
    np.testing.assert_allclose(together[:1], original, rtol=1e-4, atol=2e-5)


def test_metrics_hand_computable_and_complete_pairing():
    rows = [dict(sigma=.1, molecule_id=i, noise_seed=8, target=t,
                 egnn=b, fusion_clean=c, fusion_perturbed=n)
            for i, t, b, c, n in [(0, 1, 0, 1, 3), (1, 3, 0, 2, 1)]]
    r = paired_summary(rows, resamples=100)[0]
    assert r["egnn_mae"] == 2
    assert r["fusion_clean_mae"] == .5
    assert r["fusion_perturbed_mae"] == 2
    assert r["fusion_clean_minus_egnn"] == -1.5
    assert r["perturbed_minus_clean"] == 1.5
    assert r["fraction_of_paired_clean_topology_advantage_remaining"] == 0
    with pytest.raises(ValueError, match="Duplicate"):
        paired_summary(rows + rows)
    with pytest.raises(ValueError, match="Missing"):
        paired_summary(rows + [{**rows[0], "noise_seed": 9}])


def test_checkpoint_hash_and_architecture_fail_closed(tmp_path):
    model = EGNNGapRegressor(emb_dim=16, depth=2, mlp_hidden=32)
    path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)
    checked_state_dict(model, path, file_hash(path))
    with pytest.raises(ValueError, match="hash mismatch"):
        checked_state_dict(model, path, "wrong")
    with pytest.raises(RuntimeError):
        checked_state_dict(EGNNGapRegressor(), path, file_hash(path))


def test_frozen_splits_preserve_population(tmp_path):
    path = tmp_path / "split.json"
    split = frozen_split(path, "dataset-A", list(range(100)))
    torch.manual_seed(999)
    assert frozen_split(path, "dataset-A", list(range(100))) == split
    with pytest.raises(ValueError, match="identity mismatch"):
        frozen_split(path, "dataset-B", list(range(100)))
    split["val"], split["test"] = split["test"], split["val"]
    path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="legacy checkpoint"):
        frozen_split(path, "dataset-A", list(range(100)))


def test_cli_writes_auditable_paired_inputs_with_synthetic_fixture(tmp_path):
    """Exercise serialization, provenance and CLI; these are NOT QM9 results."""
    data_path = tmp_path / "data_v3.pt"
    InMemoryDataset.save([molecule(6 + i % 3, i) for i in range(20)], data_path)
    provenance = {"descriptor_version": DESCRIPTOR_VERSION,
                  "architecture": "source-defaults-width128-depth4-film256-head256",
                  "egnn_pytorch_version": "0.2.8", "dataset_sha256": file_hash(data_path)}
    for name, model in (("baseline", EGNNGapRegressor()), ("fusion", EGNNTDAFiLMRegressor(130))):
        path = tmp_path / f"{name}.pt"
        torch.save(model.state_dict(), path)
        provenance[name] = {"path": str(path), "sha256": file_hash(path)}
    prov_path = tmp_path / "provenance.json"
    prov_path.write_text(json.dumps(provenance))
    out = tmp_path / "result"
    command = [sys.executable, "-m", "src.eval_paired", "--data", str(data_path),
               "--provenance", str(prov_path), "--split-manifest", str(tmp_path / "split.json"),
               "--cache", str(tmp_path / "cache"), "--out", str(out), "--size", "2",
               "--batch-size", "1", "--device", "cpu"]
    subprocess.run(command, check=True, capture_output=True, text=True,
                   cwd=Path(__file__).resolve().parents[1], env={**os.environ, "MPLBACKEND": "Agg"})
    summary = json.loads((out / "summary.json").read_text())["conditions"]
    assert summary[0]["fusion_clean_mae"] == summary[0]["fusion_perturbed_mae"]
    assert summary[1]["n_molecules"] == 2
    assert (out / "inputs.npz").exists() and (out / "complete.json").exists()
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0 and "immutable" in repeated.stderr
