# EGNN with topology features on QM9

This project tests whether persistent-homology features help an EGNN predict the QM9 HOMO–LUMO gap. I built a training and evaluation pipeline around `egnn_pytorch`, then added a FiLM-conditioned model using Betti curves and persistence entropy.

The saved single-seed comparison gives test MAE **0.2051 eV for EGNN and 0.2023 eV for EGNN + TDA**. This small difference has no estimate of variation across training seeds and does not establish a reliable advantage from topology.

**Paired validation, 13 September 2026:** a corrected 256-molecule development pilot gives noisy-input MAE **0.3803 eV for EGNN and 0.3282 eV for fusion** at sigma 0.10 Å. Clean and recomputed topology give essentially identical fusion predictions. A follow-up check finds saturated FiLM activations: shuffling real descriptors or using one constant real descriptor changes predictions by less than 1e-6 eV. This does not demonstrate a benefit from molecule-specific topology. See the [pilot, checks and stop decision](reports/paired_pilot_2026-09-13.md).

The original noise comparison below used clean cached topology and independently drawn coordinate noise. It remains a historical asymmetric-input experiment; the corrected pilot uses the same perturbed coordinates in all three conditions.

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

These are the existing entry points, including the evaluation limitations above. They train models and write to the default result paths; use a separate output location to preserve historical results. Dataset files, trained checkpoints and the topology cache are outside Git, so a clone alone cannot reproduce the saved metrics. The corrected evaluation uses the separate [validation environment and commands](docs/paired_evaluation.md) or [Colab notebook](notebooks/paired_validation.ipynb).

## Validation and next decision

The [paired protocol](docs/paired_evaluation.md), `src.eval_paired` and recovery
checks implement the diagnostic. The [validation audit](reports/validation_audit_2026-09-13.md)
records recovered artifacts, software checks and execution status.

The [archived pilot](results/paired_pilot_2026-09-13) includes per-molecule predictions, exact inputs, checksums, recovery checks and measured compute. Labels stay fixed: this tests input corruption, not the electronic properties of newly distorted molecules.

The current cycle stops without a new training experiment. Further work should first address FiLM saturation, then compare independent training seeds on the fixed split with a capacity control and simple geometric descriptors. The Betti grids are fitted separately for each molecule; a shared, training-defined grid requires rebuilding the cache and retraining.

## Code and references

- [Data and topology](src/data), [models](src/models), [training](src/train.py), [fusion training](src/train_fusion.py), [evaluation](src/eval.py).
- [EGNN paper](https://arxiv.org/abs/2102.09844) and [egnn-pytorch implementation](https://github.com/lucidrains/egnn-pytorch).
- [PyG QM9 targets and units](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.datasets.QM9.html) and [giotto-tda BettiCurve](https://giotto-ai.github.io/gtda-docs/latest/modules/generated/diagrams/representations/gtda.diagrams.BettiCurve.html).
