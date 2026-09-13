# Original QM9 experiment record

## 13 September 2026 — literature synthesis and project pause

Compared the completed experiment with original EGNN, TopNets, molecular
multiparameter persistence and stability literature. Added a source-linked
comparison figure, protocol table, interpretation limits and restart criteria in
[the final synthesis](research_synthesis_2026-09-13.md). Recommended pausing the
current pipeline. No new training or changes to historical numerical artifacts.

## 13 September 2026 — completed four-arm replication

All twelve models completed ten epochs: EGNN, standardized legacy TDA fusion,
simple-geometry fusion and trained constant fusion, each with seeds 42/43/44.
The fixed split and training-only scalers were preserved, with shared
encoder/head initialization and minibatch order within each seed.

At paired noise 0.10 Å on 1,024 test molecules with three noise draws, mean MAEs
were 0.28973, 1.32849, 0.40324 and 0.26858 eV respectively. TDA also had higher
full-clean test error than each control in all three seeds. Its clean-auxiliary
noise score was 0.27106 eV, emphasizing the difference between the two scenarios.
The predeclared performance criterion failed. Descriptor sensitivity was
restored, but several runs had late optimization spikes, so the result is
specific to this representation and fixed training budget.

The [report](controlled_replication_2026-09-13.md) and
[archive](../results/controlled_replication_2026-09-13) include all histories,
335,184 predictions, exact evaluation inputs, scalers, hashes, feature checks
and measured compute. Thirteen tests passed. Local GPU training plus validation
summed to 80.16 minutes; no Colab runtime was used. Original results remain below.

## 13 September 2026 — new conditioning training diagnostic

After the checkpoint analysis, two newly initialized fusion arms were trained
for one epoch on the first 4,096 training IDs, with identical weights and batch
order. Scaling used those training molecules only. Raw features reached 85.710%
saturated FiLM outputs; standardized features remained at 0% and retained a
descriptor response. Both arms passed finite-gradient and symmetry/padding
checks. The local RTX 3080 Ti completed the diagnostic in 19.25 seconds.

The [diagnostic report](conditioning_diagnostic_2026-09-13.md) and
[archived inputs, predictions and logs](../results/conditioning_diagnostic_2026-09-13)
record the actual experiment. The conditioning gate passes; four-arm multi-seed
replication remains unexecuted. One-epoch validation MAEs are diagnostic only.

## 13 September 2026 — paired checkpoint validation

The [paired pilot](paired_pilot_2026-09-13.md) evaluated 256 frozen validation
molecules with identical perturbations. At sigma 0.10 Å, MAE was 0.380251 eV for
EGNN and 0.328248 eV for fusion with either clean or recomputed TDA. The FiLM
branch is saturated on the checked descriptors; shuffling or replacing them with
one real vector barely changes predictions. A molecule-specific topology benefit
is therefore unsupported. The cycle stops before new training. Recovery checks,
old-clean agreement, nine software tests and measured compute accompany the
result. Original numeric artifacts remain unchanged.

Follow-up analysis reports noisy per-molecule wins of 139/256 and distinguishes
the median error improvement from the larger mean improvement. The reproduced
full-test clean paired interval crosses zero. The
[next-step protocol](../docs/controlled_replication.md) separates a short
conditioning diagnostic from any multi-seed claim; no new training was run for
this analysis.

The tables below retain the original experiment outputs. New evaluation artifacts
are stored separately in `results/paired_pilot_2026-09-13`.

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

## What further training would need to resolve

The original checkpoints and selected cache files have been recovered and the
paired pilot completed. Any further training must address conditioning saturation
before interpreting descriptor robustness as topology use.

If further work is justified, train on a fixed split with independent model seeds and compare topology against both extra capacity and simple geometric descriptors. A revised common-grid representation needs its own versioned cache and retraining. Preserve these original CSVs and write the corrected experiment to a new directory, with actual execution dates and a configuration manifest.
