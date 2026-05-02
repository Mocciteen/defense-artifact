# Ciphertext-Trace Defense Artifact

This repository contains the core code for evaluating ciphertext-trace defenses
for neural-network inference.

## Components

`llvm_memory_patch_pass/` implements the defense.  The protected object is a
concrete selected memory store; function names and operator patterns are only
selectors for locating stores.

`benchmark_comp/` evaluates accuracy and overhead with externally prepared
benchmark executables or transformed bundles.  It includes the generic benchmark
server wrapper and optional adapters for external baselines.

`utility_test/` is a smaller accuracy/overhead helper for the four backend
classes used in the artifact: Glow bundle, Glow image-classifier, TVM VM, and
TVM AOT.

`mem_trace_pintool/` collects ciphertext write traces.  Generated trace binaries
and IP maps are outputs, not source files.

`trace_leakage_R/` analyzes collected traces.  It contains pattern-leakage
analysis and adjacent-change leakage analysis.

`trace_to_image_pip/` evaluates trace-to-image recovery.  It supports T-only and
T+I evaluation.  T+I testing requires a GAN prior and positive projection steps.

`trace_to_label_pip/` evaluates trace-to-label attacks from prepared tensor
corpora.

`trace_to_func_pip/` evaluates the trace-to-function / hypernetwork attack from
prepared task corpora.

`mismatch_baseline/` provides attack-agnostic synthetic mismatch traces.  These
traces are used for trace-to-image mismatch recovery and trace-to-function
mismatch evaluation.  Trace-to-label evaluation is independent and does not need
a mismatch-specific baseline.

## External Inputs

Prepare these outside the repository:

- datasets and prepared tensor corpora;
- victim models, generated bundles, and benchmark executables;
- attack checkpoints and optional GAN checkpoints;
- off/on trace directories and pair lists;
- benchmark manifests and local run configurations.

## Outputs

Typical generated outputs include accuracy summaries, overhead summaries,
leakage metrics, recovered images, attack metrics, and mismatch baseline
summaries.
