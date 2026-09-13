# EGNN with topology features on QM9

This project tests whether persistent-homology features help an EGNN predict the QM9 HOMO–LUMO gap. I built a training and evaluation pipeline around `egnn_pytorch`, then added a FiLM-conditioned model using Betti curves and persistence entropy.

The saved single-seed comparison gives test MAE **0.2051 eV for EGNN and 0.2023 eV for EGNN + TDA**. This small difference has no estimate of variation across training seeds and does not establish a reliable advantage from topology.

**Evaluation correction, 13 September 2026:** the coordinate-noise experiment leaves the fusion model's cached topology features clean. The two models also receive independently drawn noise. Its reported advantage therefore applies to that asymmetric-input setup; it does not demonstrate robustness when geometry and topology both come from the same noisy measurement. The original numbers are retained below. No corrected experiment has been run as part of this documentation review.

## Method

The baseline embeds atomic numbers, applies four EGNN layers and pools the node embeddings to predict the gap. The fusion model modulates the pooled representation with 130 features: 64 Betti bins for each of H0 and H1, plus two persistence entropies. Topology is computed from centered molecular coordinates scaled to unit diameter.

Both models use a random 80/10/10 split, seed 42, ten training epochs, batch size 64 and AdamW with learning rate 0.001. Checkpoints are selected by validation MAE. The [method notes](docs/methods.md) describe implementation details and unresolved descriptor/cache issues.

## Saved results

These values come from [compare_table.csv](results/compare_table.csv) and [compare_robustness.csv](results/compare_robustness.csv). MAE is in eV; noise sigma is in the original coordinate units.

| Input condition | EGNN test MAE | EGNN + TDA test MAE |
| --- | ---: | ---: |
| Clean coordinates and topology | 0.2051 | 0.2023 |
| Coordinate noise, sigma 0.01; fusion topology remains clean | 0.2058 | 0.2024 |
| Coordinate noise, sigma 0.05; fusion topology remains clean | 0.2344 | 0.2137 |
| Coordinate noise, sigma 0.10; fusion topology remains clean | 0.3904 | 0.3014 |

The [experiment record](reports/experiment_log.md) retains the separate baseline run, topology analysis and training history. The old noise plot in `figures/compare_robustness.png` visualizes this same asymmetric comparison, not a corrected robustness test.

## Run the existing pipeline

```bash
git clone https://github.com/serafim-tkachenko/qm9-egnn-tda.git
cd qm9-egnn-tda
uv sync --frozen
uv run python -m src.train
uv run python -m scripts.build_tda_cache
uv run python -m src.train_fusion
uv run python -m src.eval
```

These are the existing entry points, including the evaluation limitations above. They train models and write to the default result paths; use a separate checkout/output location to preserve historical results. Dataset files, trained checkpoints and the topology cache are outside Git. The repository contains selected summary results and figures, so a clone alone cannot reproduce the saved metrics. Inspect the configurations and available GPU before training. This documentation review did not install the environment or execute these commands.

## Next experiment

The [paired protocol](docs/paired_evaluation.md), `src.eval_paired` and recovery
checks implement the next diagnostic. The [validation audit](reports/validation_audit_2026-09-13.md)
records recovered artifacts, software checks and execution status.

First restore the checkpoints and record paired per-molecule errors for three conditions: noisy EGNN; fusion with noisy coordinates and clean topology; and fusion with topology recomputed from those same noisy coordinates. Reuse identical perturbations across arms and keep clean labels fixed. This tests input corruption, not the true electronic properties of newly distorted molecules.

Then decide whether a retrained, multi-seed comparison is warranted. It needs a fixed data split, independent training seeds, a capacity control and a simple geometric-descriptor baseline. The current Betti grids are fitted separately for each molecule; comparing a shared, training-defined grid requires rebuilding the cache and retraining. More architectures or larger datasets should wait until this comparison is understood.

## Code and references

- [Data and topology](src/data), [models](src/models), [training](src/train.py), [fusion training](src/train_fusion.py), [evaluation](src/eval.py).
- [EGNN paper](https://arxiv.org/abs/2102.09844) and [egnn-pytorch implementation](https://github.com/lucidrains/egnn-pytorch).
- [PyG QM9 targets and units](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.datasets.QM9.html) and [giotto-tda BettiCurve](https://giotto-ai.github.io/gtda-docs/latest/modules/generated/diagrams/representations/gtda.diagrams.BettiCurve.html).
