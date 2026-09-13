# Method and implementation notes

These notes describe the checked-in implementation reviewed on 13 September 2026. They distinguish its actual behavior from the changes needed for the next comparison.

## Data and model

`src/data/qm9_data.py` loads PyG QM9 and makes an 80/10/10 random split. The collator selects target index 4, the HOMO–LUMO gap, without an additional target transform. [PyG documents this target in eV](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.datasets.QM9.html). Dense batches pad atomic numbers and positions and carry a node mask. Original molecule indices identify cached topology vectors.

The baseline wraps `egnn_pytorch.EGNN_Network` with atomic-number embeddings, width 128 and depth 4. It uses the full graph, masked mean pooling and a regression head with hidden width 256. This project implements the surrounding pipeline and fusion model; it does not implement the underlying EGNN message-passing library from scratch.

The [EGNN construction](https://arxiv.org/abs/2102.09844) respects rotations, translations, reflections and node permutations. The relevant Euclidean group includes reflections: E(3) = R³ ⋊ O(3). Mask handling and numerical invariance still need to be checked in any modified implementation.

The fusion model uses an MLP with hidden width 256 to map the 130 topology values to 128-dimensional gamma and beta vectors. Both pass through tanh, then modulate the pooled embedding as `h' = (1 + gamma) * h + beta` before the regression head. Fusion adds parameters, so comparison with the baseline alone cannot isolate the contribution of topology.

## Topology representation

`src/data/tda_features.py` centers each point cloud and divides coordinates by the molecule's maximum pairwise distance. It computes Vietoris–Rips persistence in H0 and H1, then concatenates 64 Betti values per dimension and two persistence entropies. This produces 130 values and discards absolute molecular scale. The topology branch uses point-cloud geometry, without atom types or explicit chemical bonds.

The existing implementation calls `BettiCurve.fit_transform` separately for each molecule. Its filtration grid is therefore molecule-specific: bin 20 need not represent the same filtration distance across molecules. [BettiCurve's documented sampling behavior](https://giotto-ai.github.io/gtda-docs/latest/modules/generated/diagrams/representations/gtda.diagrams.BettiCurve.html) explains how the fitted grid is obtained. This can be treated as an adaptive descriptor, but it is not a shared-scale Betti representation. It is not, by itself, proof of train/test leakage.

For a shared-scale comparison, define and freeze the grid using training data or a prespecified physical range, save its parameters, rebuild features and retrain. Feeding changed descriptors into the old checkpoint would conflate an evaluation repair with a representation change.

The cache currently uses only numeric molecule IDs and skips existing `.npy` files. It does not validate dataset version, preprocessing or feature settings. A new experiment needs a separate cache identity containing those settings, and a perturbation identity for noisy features. The original cache must remain available for reproducing old checkpoints.

## Training and evaluation

The default training configuration uses seed 42, ten epochs, batch size 64, AdamW at learning rate 0.001, MSE training loss and validation MAE for checkpoint selection. The same seed controls the split and training randomness. A multi-seed comparison must separate those settings so that changing initialization does not also change the data split.

In `src/eval.py`, baseline and fusion evaluation draw Gaussian coordinate noise independently. Fusion then loads the cached descriptor computed from the clean molecule. This measures two-input prediction when only one input is corrupted. It cannot isolate whether topology recomputed from noisy coordinates improves robustness.

The next diagnostic should give both models identical perturbations and compare fusion with clean versus recomputed noisy topology. Preserve the existing feature definition for this first checkpoint diagnostic. Save errors by molecule and perturbation, excluding padded nodes from topology extraction. Keep clean target labels fixed and describe the task as input-corruption sensitivity, not prediction of recalculated energies for distorted molecules.

These are conclusions from source inspection. Checkpoints, raw cache files and GPU execution were not independently verified during this documentation review.
