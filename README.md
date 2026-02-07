# topo-egnn-qm9

**Topology-aware E(n)-Equivariant Graph Neural Networks for HOMO--LUMO
Gap Prediction**

------------------------------------------------------------------------

## 1. Problem Setting and Motivation

Modern molecular machine learning problems involve data that is
inherently geometric. A molecule is naturally represented as a set of
atoms embedded in three-dimensional Euclidean space. Each atom has: - a
type (atomic number), - a position in 3D space.

Many physically meaningful molecular properties are invariant under
rigid transformations: - translations, - rotations, - reflections (in
some contexts).

If a molecule is rotated or shifted in space, its energy, dipole moment,
or orbital gap should remain unchanged.

Standard neural networks do not encode these symmetries explicitly. As a
result, they must learn these invariances from data, which: - increases
sample complexity, - leads to unstable training, - reduces
generalization.

Geometric deep learning addresses this by designing models that are
equivariant or invariant to the symmetry group of the data.

In this project, we combine:

1.  **E(n)-equivariant neural networks (EGNN)**\
    for local geometric reasoning.
2.  **Topological Data Analysis (TDA)**\
    for global, coordinate-invariant structural descriptors.

We apply this hybrid approach to the **HOMO--LUMO gap prediction**
problem on the QM9 dataset.

------------------------------------------------------------------------

## 2. Mathematical Background

### 2.1 Euclidean Symmetry Group

Let the coordinates of atoms be: x_i ∈ ℝ³, for i = 1,...,N.

The Euclidean group in 3D is: E(3) = ℝ³ ⋊ SO(3),

where: - SO(3) is the rotation group, - ℝ³ represents translations.

A transformation g ∈ E(3) acts on coordinates as:

x_i → R x_i + t,

where: R ∈ SO(3), t ∈ ℝ³.

A molecular property y is physically valid only if:

y({x_i}) = y({R x_i + t}).

This defines an **invariance constraint**.

------------------------------------------------------------------------

### 2.2 Equivariance and Invariance

A function f is equivariant under group action g if:

f(g·x) = g·f(x).

It is invariant if:

f(g·x) = f(x).

In molecular prediction tasks: - intermediate representations should be
equivariant, - final outputs should be invariant.

------------------------------------------------------------------------

## 3. Graph Representation of Molecules

A molecule is represented as a graph:

G = (V, E),

where: - V: atoms (nodes), - E: interactions (edges).

Each node i has: - feature vector h_i, - coordinate x_i ∈ ℝ³.

Initial features: h_i⁰ = Embedding(Z_i),

where Z_i is the atomic number.

------------------------------------------------------------------------

## 4. E(n)-Equivariant Graph Neural Networks

### 4.1 Message Passing Formulation

At layer ℓ, the EGNN computes messages:

m_ij\^(ℓ) = φ_m(h_i\^(ℓ), h_j\^(ℓ), \|\|x_i\^(ℓ) − x_j\^(ℓ)\|\|²)

Feature update:

h_i\^(ℓ+1) = φ_h(h_i\^(ℓ), Σ_j m_ij\^(ℓ))

Coordinate update:

x_i\^(ℓ+1) = x_i\^(ℓ) + Σ_j (x_i\^(ℓ) − x_j\^(ℓ)) φ_x(m_ij\^(ℓ))

These equations guarantee equivariance because: - only relative
coordinates are used, - updates are constructed using scalar functions
of distances.

------------------------------------------------------------------------

### 4.2 Graph-Level Representation

After L layers:

h_G = Pool({h_i\^(L)})

Common pooling:

h_G = (1/N) Σ_i h_i\^(L)

Final prediction:

ŷ = f(h_G)

------------------------------------------------------------------------

## 5. Topological Data Analysis

### 5.1 Point Cloud Representation

Each molecule: X = {x₁,...,x_N} ⊂ ℝ³

This is treated as a point cloud.

------------------------------------------------------------------------

### 5.2 Vietoris--Rips Filtration

For scale ε, define:

VR(ε) = {σ ⊆ X : \|\|x_i − x_j\|\| ≤ ε for all i,j in σ}

As ε increases: - components merge, - loops appear and disappear.

------------------------------------------------------------------------

### 5.3 Persistent Homology

For each topological dimension k:

H₀ → connected components\
H₁ → loops (rings)

Each feature has: (birth, death) = (b_i, d_i)

This forms the persistence diagram:

D_k = {(b_i, d_i)}.

Persistence:

p_i = d_i − b_i

indicates importance of a feature.

------------------------------------------------------------------------

### 5.4 Persistence Image Representation

Each diagram is converted into a vector:

I(x, y) = Σ_i w(b_i, d_i) · N((x,y); (b_i,d_i), σ²I)

where: - N is a Gaussian kernel, - w is a weighting function.

The image is discretized into a fixed grid and flattened into a vector.

Properties: - stable under noise, - invariant to rigid
transformations, - fixed dimensional.

------------------------------------------------------------------------

## 6. Hybrid EGNN + TDA Model

We combine learned geometric features with topological descriptors.

Let:

h_G = EGNN(G) t_G = TDA(X)

Fusion:

z_G = concat(h_G, t_G)

Prediction:

ŷ = f(z_G)

This model combines: - local, learned geometric interactions, - global,
invariant topological structure.

------------------------------------------------------------------------

## 7. Dataset and Target

Dataset: **QM9**

Each sample: - atomic numbers Z_i, - coordinates x_i, - multiple
quantum-chemical targets.

Target: HOMO--LUMO gap.

Learning problem:

f(Z, X) → gap

Metric: Mean Absolute Error (MAE).

------------------------------------------------------------------------

## 8. Models Compared

1.  MLP baseline
2.  TDA-only model
3.  EGNN baseline
4.  EGNN + TDA fusion

------------------------------------------------------------------------

## 9. Experimental Protocol

Training: - optimizer: AdamW - loss: MSE - metric: MAE - fixed seed

Robustness test:

x_i' = x_i + ε\
ε \~ N(0, σ²I)

Test at multiple noise levels.

------------------------------------------------------------------------

## 10. Expected Contributions

We test:

1.  Do equivariant models outperform non-geometric baselines?
2.  Do topological features improve performance?
3.  Do TDA features improve robustness?

------------------------------------------------------------------------

## 11. Future Work

-   element-aware persistent homology,
-   topology-driven architecture search,
-   extension to MD17 dynamics,
-   theoretical analysis of topology--energy relationships.
