#!/usr/bin/env bash
set -euo pipefail

readonly benchmark_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 "$benchmark_root/benchmark_evaluation/scripts/measure_sm_active.py" "$@"

readonly result_dir="$benchmark_root/../results/benchmark_evaluation/latest-sm-active"
python3 "$benchmark_root/benchmark_evaluation/scripts/plot_sm_active_comparison.py" \
  "$result_dir/sm_active_results.csv" \
  "$result_dir/kesco_sm_active_comparison.png" \
  --group kesco
python3 "$benchmark_root/benchmark_evaluation/scripts/plot_sm_active_comparison.py" \
  "$result_dir/sm_active_results.csv" \
  "$result_dir/stg_sm_active_comparison.png" \
  --group stg
