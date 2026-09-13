# Original QM9 experiment record

## 13 September 2026 — paired evaluator implementation

Added a paired evaluator and artifact-recovery checks. The
[audit](validation_audit_2026-09-13.md) records hashes, environment checks and
execution status; the [protocol](../docs/paired_evaluation.md) freezes the initial
subset and three conditions. Nine software tests pass. Synthetic fixtures are
not molecular performance results. Original numeric artifacts remain unchanged.

This is a summary of the saved experiment outputs. The tables retain the original results; the interpretation was corrected on 13 September 2026 after inspecting the evaluation and topology code. This documentation review did not rerun training or evaluation.

## Correction to the noise interpretation

`src/eval.py` adds noise to coordinates but loads fusion's topology vector from the unchanged per-molecule cache. `src/data/tda_features.py` builds that cache from the dataset's clean coordinates. Baseline and fusion evaluations also draw separate noise samples.

The earlier interpretation attributed the increasing advantage under coordinate noise to stable topology. That conclusion is not established by this experiment: fusion retains a clean structural input while its coordinate input is corrupted. The original comparison below remains evidence about that specific setup. It is not a paired test of two methods using the same noisy measurement.

The code also fits Betti grids separately for each molecule and does not validate cache configuration on reuse. See the [method notes](../docs/methods.md). These require explicit treatment in a new experiment, not silent changes to the historical outputs.

## Baseline run

The saved EGNN run reports validation MAE 0.2056 eV and clean test MAE 0.2051 eV.

| Coordinate noise sigma | Test MAE, eV |
| ---: | ---: |
| 0.00 | 0.2051 |
| 0.01 | 0.2058 |
| 0.05 | 0.2342 |
| 0.10 | 0.3886 |

Sources: [baseline metrics](../results/baseline_metrics.json), [noise results](../results/baseline_robustness.csv), [figure](../figures/baseline_robustness.png). The standalone noise values differ slightly from the later comparison below. They are separate saved evaluations; the exact cause of the difference has not been independently reconstructed. Do not average or interchange them.

## Topology analysis

The original analysis used 5,000 molecules, H0/H1 persistence, 64 Betti bins per dimension and two entropy values. It reported a 93.4% H1 nonzero rate. This describes the computed point-cloud representation; it does not validate a chemical-ring interpretation or show predictive value.

Sources: [nonzero rate](../results/tda_h1_nonzero_rate.txt), [mean Betti curves](../figures/tda_betti_mean_std.png), [examples](../figures/tda_examples.png), [H1 histogram](../figures/tda_h1_max_hist.png).

## Fusion training

The FiLM-conditioned model used ten epochs, batch size 64, learning rate 0.001 and seed 42. The topology vector had 130 dimensions.

| Saved training metric | Value |
| --- | ---: |
| Best validation MAE, eV | 0.2009 |
| Final train MAE, eV | 0.2144 |
| Final validation MAE, eV | 0.2100 |
| Final training loss, MSE | 0.0838 |

The best validation checkpoint precedes the final epoch. Full epoch values remain in [training history](../results/fusion_train_history.json) and [summary](../results/fusion_summary.csv), with [loss](../figures/fusion_train_loss.png) and [MAE](../figures/fusion_mae_curves.png) plots. Ten epochs and one training seed do not establish convergence or reliable differences between methods.

## Model comparison

| Model | Validation MAE, eV | Clean test MAE, eV |
| --- | ---: | ---: |
| EGNN | 0.2056 | 0.2051 |
| EGNN + TDA | 0.2009 | 0.2023 |

Sources: [comparison metrics](../results/compare_metrics.json), [comparison table](../results/compare_table.csv). The observed clean-test difference is about 1.4%, without training-seed uncertainty or a parameter-matched control.

| Coordinate noise sigma | EGNN test MAE, eV | EGNN + TDA test MAE, eV | Observed relative reduction |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.2051 | 0.2023 | 1.4% |
| 0.01 | 0.2058 | 0.2024 | 1.7% |
| 0.05 | 0.2344 | 0.2137 | 8.9% |
| 0.10 | 0.3904 | 0.3014 | 22.8% |

**Fusion topology remains clean in every row. Noise is not paired across models.** Sources: [noise comparison](../results/compare_robustness.csv), [original plot](../figures/compare_robustness.png). These percentages describe the saved asymmetric-input comparison, not a corrected robustness result.

## What the next run must resolve

Restore the original checkpoints and caches. Run paired perturbations for the baseline and fusion, comparing clean auxiliary topology with topology recomputed from the same noisy coordinates. Keep the old feature definition for checkpoint compatibility, record per-molecule results and inspect failures before retraining.

If further work is justified, train on a fixed split with independent model seeds and compare topology against both extra capacity and simple geometric descriptors. A revised common-grid representation needs its own versioned cache and retraining. Preserve these original CSVs and write the corrected experiment to a new directory, with actual execution dates and a configuration manifest.
