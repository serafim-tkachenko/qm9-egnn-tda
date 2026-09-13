# Validation audit — 13 September 2026

Starting revision: `1f75ea29aff2b975b78929dcf1e2eaf59ab547d5` on `master`.
Historical numeric result files and figures remain unchanged. The completed
[paired pilot and diagnosis](paired_pilot_2026-09-13.md) provide the execution
record and decision for this cycle.

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
| Saturated FiLM on all 256 checked descriptors | Shuffling/constant real TDA changes predictions by less than 1e-6 eV; individual topology use is not demonstrated |

Installed PyG 2.7.0 identifies target 4 in eV and converts raw energy targets from
Hartree. The collator has no further target transform. Both saved preprocessing
records were checked and contain the string `None`. Only the topology branch centers and scales
coordinates to unit diameter. Installed giotto-tda 0.6.2 confirms per-fit grids;
empty H1 defaults to entropy -1 and zero Betti counts.

## Recovery and execution

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

Nine tests pass on Windows/Python 3.10.3 and Colab/Python 3.11.16: noise pairing, masking/batching,
symmetries, cache mismatch/corruption, checkpoint compatibility, fixed membership,
hand-computable aggregation, degenerate diagrams and a synthetic CLI run.
Synthetic fixtures are software checks, not molecular performance results.

The 130,831-molecule processed dataset was recovered through mounted Colab Drive,
then transferred locally as four compressed parts. All part hashes and the final
329,137,334-byte dataset SHA-256 were verified. Original descriptors were staged
without changing the fixed selection after a large-directory read timeout.
All 256 recomputed vectors match the original files exactly; all 40 real-molecule
symmetry/padding checks pass (maximum prediction difference 1.43e-6 eV).

The 256-molecule paired pilot and old-clean evaluator comparison completed in
Colab CPU. A representative RTX 3080 Ti optimizer-step benchmark and the topology
reliance check completed locally. The report separates measured timings from
training extrapolations. Colab was shut down after verifying persistent exports.
No new trained model was produced.

Both full clean splits were subsequently reproduced on the RTX 3080 Ti: 13,083
validation and 13,084 test molecules. All four original MAEs agree within
7.22e-9 eV, below the 2e-6 eV comparison tolerance. The published
[reproduction check](../results/paired_pilot_2026-09-13/full-clean-reproduction.json)
records their provenance, output checksums and measured timings.

See the [frozen protocol and failure gates](../docs/paired_evaluation.md).

Sources: [PyG QM9](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.QM9.html),
[BettiCurve](https://giotto-ai.github.io/gtda-docs/latest/modules/generated/diagrams/representations/gtda.diagrams.BettiCurve.html),
[EGNN paper](https://arxiv.org/abs/2102.09844). The latest giotto documentation renders
as 0.5.1; behavior was checked against installed 0.6.2 source.
