# Topology-Aware Equivariant Neural Networks for Molecular Property Prediction (HOMO–LUMO gap)

## Abstract
This project studies how **geometric deep learning** and **topological data analysis (TDA)** can be combined to improve molecular property prediction. We focus on predicting the **HOMO–LUMO gap** on the QM9 dataset, using an **E(n)-Equivariant Graph Neural Network (EGNN)** as a baseline and a **topology-aware EGNN** augmented with **persistent homology features** fused via **FiLM conditioning**.

We show that incorporating topological priors:

1. Slightly improves predictive accuracy.
2. Significantly improves robustness to coordinate noise.

---

## 1. Motivation

Molecules are inherently **geometric objects**: atoms live in 3D space, and physical properties must be **invariant** under rotations and translations. Standard neural networks do not encode these symmetries, leading to inefficient learning.

Geometric Deep Learning addresses this by enforcing symmetry constraints directly in the model. However, geometry alone does not fully capture **global structural properties** such as:

- rings,
- cycles,
- connectivity across scales.

These are naturally described by **Topology**.

This project explores whether **topological descriptors** extracted via persistent homology can complement equivariant neural networks.

---

## 2. Mathematical Background

### 2.1 Euclidean Symmetry

The Euclidean group in 3D is:

$E(3) = \mathbb{R}^3 \rtimes SO(3)$

A valid molecular predictor must satisfy:

$f(\{x_i\}) = f(\{R x_i + t\})$

for all rotations $R$ and translations $t$.

---

### 2.2 Equivariance vs Invariance

- **Equivariance**: intermediate representations transform predictably.
- **Invariance**: final outputs remain unchanged.

EGNNs guarantee equivariance at each layer and invariance after pooling.

---

## 3. E(n)-Equivariant Graph Neural Networks (EGNN)

Each molecule is modeled as a fully-connected graph:

- nodes: atoms,
- edges: pairwise interactions.

At layer $\ell$:

$m_{ij} = \phi_m(h_i, h_j, ||x_i - x_j||^2)$  

$h_i^{\ell+1} = \phi_h(h_i, \sum_j m_{ij})$  

$x_i^{\ell+1} = x_i + \sum_j (x_i - x_j)\phi_x(m_{ij})$

These updates preserve E(3)-equivariance.

A graph-level embedding is obtained by masked mean pooling.

---

## 4. Topological Data Analysis

### 4.1 Persistent Homology

Given atomic coordinates as a point cloud:

$X = \{x_1, \dots, x_N\} \subset \mathbb{R}^3$

We build a **Vietoris–Rips filtration** and track:

- $H_0$: connected components
- $H_1$: loops (rings)

Each feature has a birth–death pair $(b, d)$.

---

### 4.2 Topological Features Used

We summarize persistence diagrams using:

- **Betti curves** (64 bins per homology dimension)
- **Persistence entropy**

Total TDA feature dimension: **130**.

---

## 5. Visual Evidence of TDA

### Molecular point cloud (example)
![Point cloud](figures/tda_pointcloud_xy.png)

**Interpretation**

This plot shows the atomic coordinates of a QM9 molecule projected onto the x–y plane.

Key observations:

- The molecule contains only a few atoms.
- The structure appears sparse and slightly asymmetric.
- There are no visible ring structures.

This is expected because:

- QM9 molecules are small (≤9 heavy atoms).
- Many are simple chain-like organic fragments.
- They often lack complex topological features.

This explains why the **topological signal is relatively weak** in this dataset and why accuracy improvements are modest.

---

### Betti curves from persistent homology
![Betti curves](figures/tda_betti_curves.png)

**Interpretation**

The Betti curves show the evolution of topological features across the filtration.

Observations:

- $H_0$ remains constant at a small value.
- $H_1$ stays at zero across all scales.

Meaning:

- $H_0$ counts connected components.
- The constant value corresponds roughly to the number of atoms.
- $H_1 = 0$ indicates **no loops or rings**.

This is typical for many QM9 molecules and explains why TDA only slightly improves accuracy but improves robustness.

---

## 6. Fusion Model: EGNN + TDA (FiLM)

Let:

- $h \in \mathbb{R}^d$: EGNN graph embedding
- $t \in \mathbb{R}^{d_t}$: TDA vector

FiLM conditioning:

$(\gamma, \beta) = \text{MLP}(t)$  

$h' = (1 + \gamma) \odot h + \beta$

---

## 7. Dataset

**QM9** dataset (~130k molecules).  
Target: HOMO–LUMO gap.  
Metric: MAE.

---

## What is the HOMO–LUMO Gap?

The HOMO–LUMO gap is:

$\Delta E = E_{\text{LUMO}} - E_{\text{HOMO}}$

It measures the minimum energy required to excite an electron and determines chemical reactivity, optical absorption, and electronic behavior.

---

## 8. Training Dynamics

![Fusion loss](figures/fusion_train_loss.png)

**Interpretation**

- Training loss decreases steadily.
- Optimization is stable.
- FiLM conditioning works correctly.

---

![Fusion MAE](figures/fusion_mae_curves.png)

**Interpretation**

- Training and validation MAE decrease together.
- No severe overfitting.
- Topology acts as a mild regularizer.

---

## 9. Results

| Model | Val MAE | Test MAE |
|------|--------|---------|
| EGNN | 0.2056 | 0.2051 |
| **EGNN + TDA (FiLM)** | **0.2009** | **0.2023** |

**Analysis**

- Accuracy improves by ~1.4%.
- Improvement is modest because QM9 molecules are small and topologically simple.

---

## Robustness to Noise

![Robustness](figures/compare_robustness.png)

At noise level $\sigma = 0.10$:

| Model | Test MAE |
|------|----------|
| EGNN | 0.3904 |
| EGNN+TDA | **0.3014** |

≈23% reduction in error due to topological stability.

---

## 10. Main Conclusions

1. Geometry is essential for molecular prediction.
2. Topology provides complementary global structure.
3. Topology improves robustness to noise.
4. FiLM is an effective fusion mechanism.

---

## Reproducibility

```bash
python -m src.train
python -m scripts.build_tda_cache
python -m src.train_fusion
python -m src.eval
```
