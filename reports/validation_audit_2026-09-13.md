# Validation audit — 13 September 2026

Starting revision: `1f75ea29aff2b975b78929dcf1e2eaf59ab547d5` on `master`.
The local repository was empty with no user changes; remote master matched the
handover revision. The supplied documentation cleanup patch was not applied
upstream, passed `git apply --check`, and is now included. Historical numeric
result files and figures remain unchanged.

| Confirmed issue | Impact |
| --- | --- |
| Clean topology after coordinate corruption | Fusion retains additional clean structural information |
| Sequential independent noise draws | Perturbations are unpaired; added comparison noise has unmeasured effect size |
| Per-molecule Betti fitting | Columns use adaptive grids, not a shared filtration axis |
| Numeric-ID-only cache | No configuration, data or perturbation compatibility checks |
| Weight-only checkpoints | No embedded configuration, descriptor version, split or environment |
| One seed controls split and training | Changing initialization also changes evaluation membership |
| Fusion: 994,517 parameters; EGNN: 895,189 | Extra capacity confounds a topology-specific interpretation |
| One documented ten-epoch configuration | No convergence demonstration or training-seed uncertainty |

Installed PyG 2.7.0 identifies target 4 in eV and converts raw energy targets from
Hartree. The collator has no further target transform. Saved preprocessing records
still need checking before evaluation. Only the topology branch centers and scales
coordinates to unit diameter. Installed giotto-tda 0.6.2 confirms per-fit grids;
empty H1 defaults to entropy -1 and zero Betti counts.

## Recovery status at implementation checkpoint

Located the original Drive project, processed `data_v3.pt` (329,137,334 bytes),
preprocessing records, result summaries, checkpoints and topology-cache folder.
Both checkpoints were recovered locally and load strictly into default models:

| File | SHA-256 |
| --- | --- |
| `best_egnn.pt` | `3678bab48fe8b817e59bfbde08b03c85e67700f098f58dd70e31a04a7ac89138` |
| `best_fusion.pt` | `0493bf12c3a9c76e0a2bc17fc47265a0dcd3ea63b0c4382e127e9c3f1077797b` |

Local RTX 3080 Ti: 12,288 MiB VRAM, driver 610.47; system RAM about 31.8 GiB.
PyTorch 2.10.0+cu128 recognizes the GPU with CUDA available. Free disk was about
13.2 GiB initially and 7.5 GiB after isolated environment installation. Desktop
graphics workloads were present; no separate GPU compute job was visible.
Availability snapshots are not throughput benchmarks.

Nine tests pass on Windows/Python 3.10.3: noise pairing, masking/batching,
symmetries, cache mismatch/corruption, checkpoint compatibility, fixed membership,
hand-computable aggregation, degenerate diagrams and a synthetic CLI run.
Synthetic fixtures are software checks, not molecular performance results.

The large dataset could not yet be materialized locally: the connector exceeded
its transfer-frame limit and the application browser blocked the binary download.
Colab with mounted Drive is the next route. At this implementation checkpoint,
no corrected QM9 pilot, clean full-split reproduction or representative training
benchmark has completed; no Colab runtime or new training has been used. There
is no measured multi-seed compute estimate. Append execution results when available.

See the [frozen protocol and failure gates](../docs/paired_evaluation.md).

Sources: [PyG QM9](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.QM9.html),
[BettiCurve](https://giotto-ai.github.io/gtda-docs/latest/modules/generated/diagrams/representations/gtda.diagrams.BettiCurve.html),
[EGNN paper](https://arxiv.org/abs/2102.09844). The latest giotto documentation renders
as 0.5.1; behavior was checked against installed 0.6.2 source.
