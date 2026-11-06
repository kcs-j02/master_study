# Real-application ablation evaluation

This independent runner evaluates the six KESCO-derived applications with the
strict ablation executable:

- `vector_square`
- `black_scholes`
- `machine_learning`
- `hits`
- `image_processing`
- `deep_learning`

Each condition uses the same STG and input data. The default is 33 runs with
3 warmups and a fixed three-Stream comparison:

```bash
./experiments/application/run_application_ablation.sh
```

Smoke test:

```bash
RUNS=2 WARMUP_RUNS=1 \
RESULTS_DIR=/tmp/application_ablation \
./experiments/application/run_application_ablation.sh
```

Outputs are stored below `results/application_ablation_corrected/` by default. The
runner stops if any KESCO validation field is missing or if any validation
run fails. The summary compares every ablation condition with `full`; it does
not treat a failed validation run as a performance result.

## Scheduling overhead

The full-variant logs also include Stage 1--4 CPU-side scheduling times. To
summarize the post-warmup overhead:

```bash
python3 experiments/measure_overhead.py \
  results/application_ablation_corrected/application_ablation_results.csv \
  results/application_ablation_corrected/application_ablation_summary.csv
```

`scheduling_overhead_ms` is the sum of STG analysis, task-importance
calculation, Stream placement, and SM allocation. It is reported separately
from GPU execution and is not included in the performance claim without
explicitly stating the end-to-end measurement scope.
