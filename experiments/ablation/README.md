# Ablation experiment suite

This directory adds an independent experiment driver. Existing source files
and result files are not modified.

The strict implementation is built separately:

```bash
./experiments/ablation/build_strict_ablation.sh
```

It supports `STG_ABLATION_MODE=full`, `no_importance`,
`no_sm_allocation`, and `no_stream_placement`. `STG_ABLATION_STREAMS`
fixes the Stream count for the first three modes (default: 3), so the
importance and SM-allocation comparisons do not also change the Stream count.
The no-placement mode intentionally uses one Stream because disabling Stream
parallelism means all tasks share one execution Stream.

The four executable variants are used as a staged ablation:

| Variant | Executable | Interpretation |
|---|---|---|
| `no_dependency_scheduling` | `sequential_method` | No dependency-aware Stream placement |
| `stream_placement` | `existing_method` | Dependency-aware Stream placement without Green Context |
| `stream_placement_green_context` | `existing_method_green_context` | Existing placement plus SM partitioning |
| `proposed_full` | `proposed_method` | Proposed importance-aware placement and SM allocation |

The third row is an **existing-method Green Context comparison**, not an
isolated SM-allocation-only implementation. The suite deliberately preserves
that distinction instead of labeling it as a stronger ablation than the
available executables support.

## Run

From the repository root:

```bash
./experiments/ablation/run_ablation.sh
```

Useful overrides:

```bash
RUNS=33 WARMUP_RUNS=3 \
  ./experiments/ablation/run_ablation.sh \
  synthetic_mixed_chain_parallel synthetic_sequential_chain
```

Results are written below `results/ablation/` by default:

- `ablation_results.csv`: one row per variant and benchmark
- `ablation_summary.csv`: speedup and confidence intervals relative to the
  no-scheduling variant
- `ablation_summary.md`: a paper-ready summary table
- `raw/<variant>/`: raw logs and per-method CSV files

The reported `gpu_submit_wait_ms` is a host wall-clock interval around task
submission, executor waiting, cleanup, and `cudaDeviceSynchronize()`. The
`gpu_kernel_ms` value is computed separately with CUDA Events from the first
task event to the last task event. Compare both values from raw logs with:

```bash
python3 experiments/measurement_validation.py \
  results/ablation/raw results/ablation/measurement_validation.csv
```

The difference must be reported as host/runtime overhead, not silently
described as GPU kernel execution time.

Set `RESULTS_DIR=/absolute/path` to use another output directory. The script
requires the four existing executables to have already been built.

## Scope and interpretation

This suite measures the same `gpu_submit_wait_ms` metric and validation fields
as the repository's standard benchmark runner. It does not claim that the
third row isolates SM allocation. To isolate that factor, a separate executable
with Stream placement held constant and only the SM allocation policy changed
must be added in a future experiment.
