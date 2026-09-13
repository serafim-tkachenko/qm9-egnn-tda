# Controlled QM9 replication — 13 September 2026

**Final context:** [literature comparison and pause decision](research_synthesis_2026-09-13.md).
This is an internal replication across training seeds, not a reproduction of
the published EGNN QM9 benchmark. The final synthesis documents that distinction.

**The standardized legacy TDA model failed the predeclared comparison.** It had
higher clean and matched-noise test MAE than all three controls in every training
seed. Standardization restored descriptor sensitivity, but that sensitivity was
harmful under coordinate corruption. This is a negative result for the tested
representation, optimizer and ten-epoch budget, not a rejection of topology in
general.

![Learning curves, paired noise results and conditioning](../results/controlled_replication_2026-09-13/replication.png)

## Main result

Entries are mean MAE ± sample standard deviation across **three training seeds**,
in eV. Clean test uses all **13,084** test molecules. Noise uses the preselected
first **1,024** test IDs and averages three fixed perturbations per molecule.
The two columns therefore have different populations; use the next table to
compare noise levels on identical molecules.

| Model | Full clean test | Matched inputs, noise 0.10 Å |
| --- | ---: | ---: |
| EGNN | 0.21718 ± 0.01558 | 0.28973 ± 0.01062 |
| Standardized TDA fusion | 0.25769 ± 0.02136 | 1.32849 ± 0.09382 |
| Simple-geometry fusion | 0.23648 ± 0.02892 | 0.40324 ± 0.05650 |
| Trained constant fusion | 0.21402 ± 0.02572 | 0.26858 ± 0.01234 |

The working criterion, frozen before training, required TDA MAE at 0.10 Å to be
at least 1% below **each** control, with lower error in all three seeds. It was
not met. TDA's noisy MAE was approximately **4.59 times EGNN's**, with worse
results in every seed.

| Noise σ, Å; same 1,024 molecules | EGNN | TDA fusion | Simple geometry | Constant fusion |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.21771 | 0.25106 | 0.23978 | 0.21194 |
| 0.01 | 0.21800 | 0.29909 | 0.23975 | 0.21226 |
| 0.05 | 0.22956 | 0.95376 | 0.26920 | 0.22149 |
| 0.10 | 0.28973 | 1.32849 | 0.40324 | 0.26858 |

These are means of the individual models' MAEs, **not ensemble-prediction MAEs**.
The plot's bands show seed minima/maxima, not confidence intervals.

## Pairing and uncertainty

At 0.10 Å, paired TDA-minus-control MAE differences are:

| Control | Mean difference, eV | 95% molecule-bootstrap interval, eV | Per-seed differences: 42 / 43 / 44 |
| --- | ---: | --- | --- |
| EGNN | +1.03875 | [0.98868, 1.08928] | +0.92614 / +1.12392 / +1.06619 |
| Simple geometry | +0.92525 | [0.88290, 0.97007] | +0.85815 / +0.94383 / +0.97376 |
| Constant fusion | +1.05990 | [1.01025, 1.11060] | +0.96031 / +1.15504 / +1.06435 |

The 4,000 bootstrap resamples cluster by molecule, after averaging its fixed
noise replicates and model seeds. Intervals condition on these twelve fitted
models and fixed noise draws. They do not estimate training-population
uncertainty; seed differences and standard deviations are reported separately.
Intervals are unadjusted across comparisons. Three seeds remain exploratory.

On the full clean test set, TDA-minus-EGNN is +0.04052 eV with conditional
molecule interval [0.03788, 0.04324] eV. TDA also has higher clean error than
geometry and constant fusion in every seed. Constant fusion's slightly lower
mean than EGNN should not be promoted to a general superiority claim from
these three runs.

## Clean auxiliary topology changes the story

Both requested scenarios were evaluated with the same noisy coordinates and
unchanged target labels:

| Auxiliary representation at 0.10 Å | Recomputed from noisy coordinates | Retained from clean coordinates |
| --- | ---: | ---: |
| TDA | 1.32849 eV | 0.27106 eV |
| Simple geometry | 0.40324 eV | 0.31444 eV |

The TDA difference is large. A clean auxiliary descriptor substantially changes
this experiment's information conditions; it cannot substitute for the matched
measurement-error result. This is why both conditions must remain labeled and
why the original asymmetric robustness interpretation was unsafe. These new
models are distinct from the historical saturated checkpoint.

## Conditioning and descriptor shift

The selected TDA checkpoints have **12.97%, 8.92% and 19.45%** saturated outputs
on the full clean test set, using the absolute-tanh threshold 0.9999. Shuffling
clean descriptors changes predictions by **0.9073, 1.0525 and 1.0440 eV** on
average. The historical checkpoint changed by less than 1e-6 eV in its pilot.
The new models clearly respond to individual descriptors; responsiveness alone
is not robustness or predictive benefit.

For noise seed 20260913 at 0.10 Å, mean per-molecule RMS feature change, measured
in training-standard-deviation units, is 1.0569 for TDA and 0.8937 for simple
geometry. These summaries span different feature sets and do not directly rank
their predictive importance. The largest TDA changes occur at zero-based H0
Betti indices 48, 55, 54, 49 and 56, around 2.74–3.17 training standard deviations
RMS. Only 0.30% of TDA entries lie outside their univariate training ranges in
this replicate. The failure therefore cannot be explained merely by counting
out-of-range scalar values.

All 10,240 saved subset cases reproduced their exact coordinate perturbations,
atom identities, labels and simple descriptors. Eight TDA vectors per condition
(80 total) independently recomputed exactly. This rules out the checked input
alignment/cache failure modes. The adaptive Betti grids, diameter normalization,
feature scaling and learned sensitivity have not been separately ablated;
attributing the failure to any single component would go beyond these checks.

## Training and limitations

The [frozen protocol](../docs/controlled_replication.md) used 104,664 training and
13,083 validation molecules; split seed 42; model seeds 42/43/44; batch size 64;
ten epochs; AdamW with learning rate 0.001 and weight decay 0.01; MSE loss;
float32 with TF32 disabled. Scalers were fit on the full training split only.
All arms share initial encoder/head weights within each seed and identical
epoch orders. TDA and constant fusion also share initial FiLM weights. This
does not make their initial *effective predictions* identical to EGNN.

Parameter counts are 895,189 (EGNN), 994,517 (TDA/constant), and 994,502 (geometry).
Geometry uses five atom counts and five unordered pair-distance statistics.
Constant fusion receives the training mean TDA vector, represented as zero
after standardization, throughout training and evaluation. It has the same
fusion implementation but not the same effective capacity as variable-input
fusion. Constant FiLM can be absorbed into the head's first affine layer.

| Selected epoch by validation MAE | Seed 42 | Seed 43 | Seed 44 |
| --- | ---: | ---: | ---: |
| EGNN | 7 | 10 | 10 |
| TDA | 9 | 9 | 10 |
| Simple geometry | 10 | 8 | 9 |
| Constant fusion | 9 | 10 | 8 |

All training values remained finite, but several runs were **not optimization
stable**. Geometry seed 43 had an epoch-10 FiLM gradient maximum of 2,953.8 and
validation MAE 0.6293; TDA seed 44 spiked at epoch 7; constant seed 44 ended at
0.5089 validation MAE. Earlier selected checkpoints were retained. Other curves
also fluctuate. No clipping, learning-rate tuning, extra epochs or descriptor
redesign was introduced after seeing outcomes. These failures and the limited
budget constrain conclusions about each architecture's attainable accuracy.

The test set has already been used in historical work, so this is retrospective
validation. Noise models measurement corruption with fixed clean gap labels;
it does not estimate quantum energies of newly distorted molecules.

## Compute, checks and reproduction

Execution used the local RTX 3080 Ti; **no Colab runtime was allocated**. Feature
preparation took 94.55 seconds, training phases summed to 76.71 minutes,
training plus validation to 80.16 minutes, and evaluation to 87.15 seconds.
Phase timings exclude some startup, batch assembly, checkpoint I/O and
orchestration; concurrent software checks affected early timing measurements.

All **13 software tests passed**. Selected-checkpoint rigid-transform/atom-order
checks had maximum absolute prediction difference 4.30e-6 eV; extra padding
changed predictions by 0 in the checked batches. Source/output hashes,
training-only scalers, all epoch orders and validation selection were verified.
The archive contains **335,184 predictions**, exact evaluation inputs, scaler
values, twelve histories, checkpoint hashes, feature analysis and the figure.
Weights, resume states and full train/validation feature arrays remain outside
Git in the local output directory. Historical artifacts are unchanged.

```bash
python -m scripts.run_controlled_replication --data data/qm9/processed/data_v3.pt --split artifacts/recovered/split42.json --out outputs/controlled-new --workers 4
python -m scripts.evaluate_controlled_replication --data data/qm9/processed/data_v3.pt --split artifacts/recovered/split42.json --run outputs/controlled-new --out outputs/controlled-evaluation-new --workers 4
python -m scripts.analyze_replication_features --run outputs/controlled-new --evaluation outputs/controlled-evaluation-new --out outputs/controlled-feature-analysis-new.json
python -m pytest -q
```

The training runner resumes matching configurations at saved epoch boundaries;
evaluation and analysis require new output locations. See the
[artifact archive](../results/controlled_replication_2026-09-13) for the frozen
manifests and exact results.

## Decision

Do not claim topology-specific robustness for this pipeline or expand its
training budget to rescue the present result. The completed experiment provides
a useful negative finding: after recovering descriptor sensitivity, the newly
trained models were strongly vulnerable to descriptor perturbations. The
historical saturated checkpoint had been insensitive to those inputs; it was
a different fitted model, not a counterfactual version of these new checkpoints.

If further work is pursued, first study optimization stability using validation
only and equal tuning budgets across arms. A shared-grid or other stable
vectorization should be a separately versioned representation experiment, with
the simple-geometry and constant controls retained. This run has not established
which redesign would work, and no additional training experiment is included in
these results.
