# Scale and seed evaluation

This independent suite generates additional synthetic STGs without modifying
existing benchmarks or source files. It evaluates:

- chain graphs,
- fully parallel graphs,
- random DAGs with seeds,
- multiple task counts.

Run from the repository root:

```bash
./experiments/scale_seed/run_scale_seed.sh
```

The default design is 3 graph families, 4 sizes (`10 20 50 100`), 3 seeds,
4 strict ablation variants, and 33 runs with 3 warmups. For a smoke test:

```bash
RUNS=2 WARMUP_RUNS=1 \
GRAPH_SIZES="10" GRAPH_SEEDS="1" \
RESULTS_DIR=/tmp/scale_seed \
./experiments/scale_seed/run_scale_seed.sh
```

The output is `scale_seed_results.csv`. Report the mean and 95% confidence
interval for each graph family and size. For random DAGs, aggregate across
seeds rather than treating one seed as representative.
