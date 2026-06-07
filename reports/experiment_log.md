# Experiment Log

## Experiment 001 — EGNN baseline

**Goal:**
Train an E(n)-Equivariant Graph Neural Network baseline for HOMO–LUMO gap prediction on QM9.

**Setup:**
Model: EGNN
Dataset: QM9
Target: HOMO–LUMO gap
Metric: Mean Absolute Error

**Artifacts:**

* `results/baseline_metrics.json`
* `results/baseline_robustness.csv`
* `figures/baseline_robustness.png`

**Result:**

| Metric         |  Value |
| -------------- | -----: |
| Validation MAE | 0.2056 |
| Test MAE       | 0.2051 |

**Baseline robustness:**

| Noise sigma | Test MAE |
| ----------: | -------: |
|        0.00 |   0.2051 |
|        0.01 |   0.2058 |
|        0.05 |   0.2342 |
|        0.10 |   0.3886 |

**Interpretation:**
The baseline trains stably and provides a reasonable reference point for evaluating the contribution of topological features.

As expected, the baseline becomes less accurate when molecular coordinates are perturbed, because the model relies directly on coordinate-based geometric information.

**Notes:**
No severe overfitting was observed. The EGNN already captures much of the relevant local molecular geometry.

---

## Experiment 002 — Dataset-level topology analysis

**Goal:**
Check whether QM9 molecules contain nontrivial topological signal before using persistent homology features for prediction.

**Setup:**
Input: molecular 3D point clouds
Topological features: (H_0), (H_1), Betti curves, persistence entropy
Sample: 5000 molecules
Betti bins: 64
Maximum homology dimension: 1
Final TDA feature dimension: 130

**Artifacts:**

* `figures/tda_betti_mean_std.png`
* `figures/tda_examples.png`
* `figures/tda_h1_max_hist.png`
* `results/tda_h1_nonzero_rate.txt`

**Result:**

| Metric          | Value |
| --------------- | ----: |
| H1 nonzero rate | 93.4% |

**Interpretation:**
The (H_0) curves capture the merging of atomic components across filtration scales. The (H_1) signal is weaker but non-degenerate, suggesting that many QM9 molecules contain some loop-like geometric structure that can be captured by persistent homology.

**Notes:**
Because QM9 molecules are relatively small, the topological signal is limited. This is one reason why a large standard prediction improvement should not be expected.

---

## Experiment 003 — EGNN + TDA with FiLM conditioning

**Goal:**
Test whether persistent homology features improve molecular property prediction when fused with EGNN graph embeddings.

**Setup:**

| Parameter                  |             Value |
| -------------------------- | ----------------: |
| Model                      | EGNN + TDA (FiLM) |
| Epochs                     |                10 |
| Batch size                 |                64 |
| Learning rate              |             0.001 |
| Seed                       |                42 |
| TDA bins                   |                64 |
| TDA max homology dimension |                 1 |
| TDA feature dimension      |               130 |

**Artifacts:**

* `results/fusion_train_history.json`
* `results/fusion_summary.csv`
* `figures/fusion_train_loss.png`
* `figures/fusion_mae_curves.png`

**Training summary:**

| Metric               |  Value |
| -------------------- | -----: |
| Best validation MAE  | 0.2009 |
| Final train MAE      | 0.2144 |
| Final validation MAE | 0.2100 |
| Final train loss     | 0.0838 |

**Training dynamics:**

| Stage    | Train loss | Train MAE | Validation MAE |
| -------- | ---------: | --------: | -------------: |
| Epoch 1  |     0.6476 |    0.5992 |         0.4809 |
| Epoch 2  |     0.2143 |    0.3485 |         0.3113 |
| Epoch 3  |     0.1592 |    0.2995 |         0.2810 |
| Epoch 8  |     0.0842 |    0.2185 |         0.2009 |
| Epoch 9  |     0.0760 |    0.2081 |         0.2019 |
| Epoch 10 |     0.0838 |    0.2144 |         0.2100 |

**Interpretation:**
The fusion model trains stably. Adding FiLM conditioning with TDA features does not appear to destabilize optimization.

The best validation MAE was reached before the final epoch, and the final validation MAE increased slightly, so the best checkpoint should be used for evaluation.

---

## Experiment 004 — Standard prediction comparison

**Goal:**
Compare EGNN against EGNN + TDA on standard in-distribution HOMO–LUMO prediction.

**Artifacts:**

* `results/compare_metrics.json`
* `results/compare_table.csv`

**Result:**

| Model             | Validation MAE | Test MAE |
| ----------------- | -------------: | -------: |
| EGNN              |         0.2056 |   0.2051 |
| EGNN + TDA (FiLM) |         0.2009 |   0.2023 |

**Interpretation:**
The topology-aware model improves test MAE from 0.2051 to 0.2023, approximately 1.4%.

This is a weak/modest improvement. It should not be interpreted as strong evidence that persistent homology substantially improves HOMO–LUMO prediction on QM9.

**Notes:**
The result suggests that topology may provide complementary information, but the standard accuracy gain alone is not strong enough to claim a meaningful predictive improvement.

---

## Experiment 005 — Robustness comparison: EGNN vs EGNN + TDA

**Goal:**
Evaluate whether topological features improve robustness when molecular coordinates are perturbed.

**Setup:**
Models compared:

* EGNN
* EGNN + TDA with FiLM conditioning

Perturbation: Gaussian coordinate noise
Metric: Test MAE

**Artifacts:**

* `results/compare_robustness.csv`
* `figures/compare_robustness.png`

**Result:**

| Noise sigma | EGNN test MAE | EGNN + TDA test MAE | Relative error reduction |
| ----------: | ------------: | ------------------: | -----------------------: |
|        0.00 |        0.2051 |              0.2023 |                     1.4% |
|        0.01 |        0.2058 |              0.2024 |                     1.7% |
|        0.05 |        0.2344 |              0.2137 |                     8.9% |
|        0.10 |        0.3904 |              0.3014 |                    22.8% |

**Interpretation:**
The topology-aware model becomes more useful as coordinate noise increases.

At (σ=0.10), EGNN + TDA reduces error by approximately 22.8%. This is the strongest result of the project. Persistent homology features appear to provide a more stable global descriptor when precise coordinate information is degraded.

**Notes:**
This robustness result is more interesting than the standard prediction result. It suggests that topological features may be useful for stability and robustness rather than direct in-distribution accuracy gains on small molecular datasets.

---

## Overall conclusion

Topology provides only a modest improvement in standard HOMO–LUMO prediction accuracy on QM9, but a substantially stronger improvement in robustness under coordinate perturbation.

The project should be treated as an exploratory scientific ML investigation rather than a performance breakthrough. The main value is that it tests a plausible hypothesis, documents a weak/modest accuracy result honestly, and identifies a more promising robustness direction.

The most interesting future work is to test whether topology improves robustness, generalization, or out-of-distribution performance on larger and more structurally complex molecular datasets.