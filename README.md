# Artifact Overview

This artifact contains a minimal code-only snapshot for the paper. It is
organized by experiment component and excludes raw datasets, model checkpoints,
raw traces, generated binaries, and experiment results.

## Layout

- `attack_evaluation_self/ciphersteal/`: code used for the CipherSteal-style
  trace collection, execution-matrix runs, and collision/leakage analysis.
- `attack_evaluation_self/hypertheft/`: code used for the HyperTheft-style
  attack evaluation, mismatch-trace experiments, local model modules, and latent
  analysis.
- `modified_framework_files/glow/`: Glow source files modified for the defense
  and trace-collection experiments.
- `modified_framework_files/tvm/`: TVM source files modified for the defense
  and trace-collection experiments.
- `leakage_analysis/`: standalone collision and higher-order trace-analysis
  scripts.
- `trace_collection/`: trace-collection and execution-matrix helper scripts.

## Placeholders

Local machine paths have been anonymized. Paths beginning with `/path/to/...`
are placeholders and should be replaced with the corresponding local workspace,
dataset, Glow, TVM, trace, or experiment directory before running scripts.

## Excluded Files

The artifact intentionally excludes datasets, private traces, model weights,
checkpoints, generated logs, generated tables, figures, compiled libraries, and
other bulky or sensitive outputs. These files should be regenerated locally or
obtained through the proper dataset/model channels when needed.
