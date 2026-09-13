# Proposed controlled replication

Status: a follow-up protocol, not executed results. The completed checkpoint
validation is in the [pilot report](../reports/paired_pilot_2026-09-13.md).

The next question is whether a fusion model can learn useful molecule-specific
conditioning once saturation is controlled, and whether topology improves on
simple geometric information under the same budget. Repeating the old saturated
model across more seeds would not resolve that question.

## First step: a bounded training diagnostic

Keep the current adaptive Betti grids, diameter normalization, EGNN architecture,
loss and original split. Compare raw TDA with training-standardized TDA using the
same initialization seed and minibatch order. Fit per-feature mean and standard
deviation on training molecules only; replace scales below 1e-6 by 1 and persist
the fitted values with dataset/descriptor hashes. Never fit scaling on validation
or noisy evaluation inputs.

Use the first 4,096 training IDs in the frozen split and one epoch for the initial
software/training check. This subset is selected without outcomes. Log feature
ranges, pre-tanh quantiles, saturated-output fraction, conditioning-branch gradient
norms, losses and per-molecule prediction sensitivity at initialization and after
the epoch. Use the existing 256 validation molecules for diagnostics only.

Require finite inputs, gradients and predictions, preserved masks/symmetries,
and correct train-only preprocessing. Flag a saturated-output fraction above 95%
(`abs(tanh) > 0.9999`) for diagnosis before a larger run. This is an operational
warning threshold, not a definition of predictive usefulness. If standardization
does not resolve saturation, test a smaller FiLM output-layer initialization as a
separate change; do not silently combine both changes or redesign the Betti grid.
Keep logs from failed configurations. The one-epoch check cannot establish an
accuracy advantage or convergence.

## Comparison after the diagnostic passes

| Arm | Question | Budget/control |
| --- | --- | --- |
| EGNN | Reference predictor | Original encoder/head |
| Fusion with standardized legacy TDA | Does learned individual topology help? | Same encoder and checkpoint-selection rule |
| Fusion with simple geometry | Does topology add more than cheap size/distance information? | Nearly equal parameter count to TDA fusion |
| Fusion with one fixed training-derived descriptor | Can constant conditioning explain the architecture's gain? | Same fusion implementation; this is a constant-conditioning control, not equal effective capacity |

For the simple descriptor, predeclare five atom counts (H, C, N, O, F) and five
statistics of unpadded, unordered pair distances in Å: mean, standard deviation,
minimum, maximum and root mean square. Define the no-pair case as zeros. Fit its
scaling on training data only. With the current architecture, ten inputs and
FiLM hidden width 371 give 994,502 parameters, versus 994,517 for 130-input TDA
with width 256. Verify these counts in code before training. The matching is of
parameter count; it does not make the representations equally expressive.

For the constant arm, choose the training mean descriptor once, before outcomes,
and use it for every molecule in both training and evaluation. This differs from
the completed post-hoc replacement stress test: the control must be trained with
its constant input. Do not interpret the failed all-zero inference perturbation
of the historical checkpoint as this trained control.

Freeze split seed 42 independently of model seeds 42, 43 and 44. Use the same
training budget (initially ten epochs), batch size 64, float32/TF32-off precision,
optimizer settings and validation-MAE checkpoint rule across arms. Report each
seed, learning curves and actual wall time. Three seeds are an initial comparison,
not strong evidence of stability; ten epochs do not establish convergence. Any
budget extension must apply consistently and be chosen using validation only.

Evaluate paired coordinate noise at 0, 0.01, 0.05 and 0.10 Å with predeclared noise
seeds 20260913, 20260914 and 20260915. For the primary corruption comparison,
recompute distance/TDA features from the exact same noisy coordinates. Retain
clean auxiliary features as a separately labeled secondary condition. Keep labels
fixed, and report per-molecule/replicate predictions. Separate molecule/noise
uncertainty from variation across training seeds. Choose any practical-effect
threshold on validation before inspecting new test results; record it in the run
manifest. The previously used test set remains retrospective.

## Cost and decision

The existing fixed-batch benchmark suggests about 33 minutes of optimizer work
for two models × three seeds × ten epochs. Four arms would be roughly twice that,
before topology extraction, validation, I/O and tuning. This is an extrapolation
from batch size 64 padded to 25 atoms, not a promised runtime. Measure the actual
epoch during the diagnostic; use the local RTX 3080 Ti first. No Colab GPU needs
to wait for CPU feature extraction.

Continue only if the conditioning path responds to descriptors and the controlled
comparison can answer the stated question. If TDA does not improve on simple
geometry across seeds, report that outcome. A stable negative result is sufficient;
larger models or datasets are not required. A shared-grid representation would be
a subsequent, separately versioned experiment.
