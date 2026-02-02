# topo-egnn-qm9

Topology-aware E(n)-Equivariant Graph Neural Networks for HOMO–LUMO gap prediction on QM9.

## Project idea
We compare:
- EGNN baseline (equivariant 3D encoder)
- EGNN + TDA (persistent homology features via persistence images)
- Robustness under coordinate noise

## Setup (uv)

pip install uv
uv venv
source .venv/bin/activate
uv sync
