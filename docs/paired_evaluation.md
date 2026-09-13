# Paired checkpoint diagnostic

This retrospective development evaluation studies both clean topology supplied
alongside noisy geometry and topology recomputed from the same noisy measurement.
Targets remain the original QM9 gaps, not recalculated energies at distorted coordinates.

## Frozen protocol

- Save the original seed-42 80/10/10 split independently of training RNG. Select
  the first 256 validation indices in original random split order, before outcomes.
- Sigma 0 and 0.10 angstrom, noise seed 20260913. SHA-256 of molecule ID, sigma
  and seed selects a NumPy PCG64 stream. Materialize float32 unpadded coordinates
  once; all three arms consume exactly those inputs.
- Compare EGNN, fusion with clean topology, and fusion with recomputed perturbed
  topology. Retain unit-diameter normalization and per-molecule adaptive Betti
  grids. No representation redesign or retraining in this cycle.
- Float32 inference, TF32 off, batch size 64; record any device or batch changes.
- Save predictions, targets, errors, noise IDs, exact input arrays, input and
  descriptor hashes, checkpoint hashes, package/source versions and timings.
- Report MAE and paired fusion-minus-EGNN errors (negative favors fusion).
  Use 4,000 percentile bootstrap samples with seed 20260913, resampling molecules
  after averaging any noise replicates within molecule. This conditions on the
  checkpoints and does not estimate training-seed uncertainty.
- Report the fraction of the **paired clean-topology** advantage remaining only
  if that reference advantage is positive. This descriptive ratio is unstable
  near zero and is not a comparison with the old unpaired full-test percentage.

## Recovery gate

`scripts.prepare_paired` verifies the recovered checkpoint SHA-256 hashes and
strict architecture, loads processed QM9 without downloading raw data, checks
saved preprocessing records, freezes the split, and compares all 256 clean
descriptors with their legacy cache files. Betti counts must match exactly;
entropy allows 1e-6 absolute/relative rounding tolerance. Plain legacy state
dictionaries cannot establish missing training metadata or original package
versions. Hashes bind recovered artifacts, not undocumented historical settings.

The first eight validation molecules undergo translation, rotation, reflection,
permutation and extra-padding checks with recovered models. Prediction tolerance
is 1e-4 eV absolute plus 1e-5 relative, much smaller than the historical 0.20 eV
MAE. Finite precision and discontinuous Betti binning may still cause failures;
these are recorded and block expansion. Actual grids for three preselected
molecules are exported. Empty H1 retains the giotto-tda 0.6.2 convention: zero
Betti counts and entropy -1. Nonfinite inputs/descriptors fail explicitly.

## Commands

Use Python 3.10 or 3.11 and install an appropriate PyTorch 2.10 build first:

```bash
python -m pip install -r requirements-validation.txt
python -m pytest -q
python -m scripts.prepare_paired --original-root /path/to/qm9-egnn-tda --out outputs/recovery
python -m src.eval_paired --data /path/to/qm9-egnn-tda/data/qm9/processed/data_v3.pt --provenance outputs/recovery/provenance.json --split-manifest outputs/recovery/split42.json --cache artifacts/validation_cache/pilot --out outputs/paired-pilot --device cuda
```

On Colab add `--compute-environment colab`. Preserve the source revision and
environment recorded by each run. Use separate paths from historical files.
Existing output directories are refused. Completed prediction batches persist
incrementally; incomplete runs lack `complete.json`. After interruption, choose
a new output directory and reuse the checked cache.

For clean full-split reproduction, use separate outputs with `--size 0 --split val
--sigmas 0`, then `--split test`. These remain retrospective evaluations.
The historical `src.eval` remains available and must not be used for a corrected
robustness claim. Frozen membership is checked against the original seed-42 split.

## Decision rule

Inspect recovery/symmetry checks before the table and per-molecule errors. Do not
select favorable subsets, noise seeds or representations after observing outcomes.
Stop expansion on unresolved artifact or representation failures. After a valid
pilot, decide whether to stop with a corrected claim, expand paired evaluation,
or retrain with controls. A topology-specific claim needs capacity and simple
geometric-descriptor controls plus independent training seeds on a fixed split.
