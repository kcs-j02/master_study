#!/usr/bin/env bash
set -euo pipefail
readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly project_dir="$(cd -- "$script_dir/.." && pwd)"
readonly study_dir="$(cd -- "$project_dir/../.." && pwd)"
readonly figure_root="$study_dir/results/method_comparison_figures"
readonly stg_figure_dir="$figure_root/RESULT_stg"
readonly benchmark_figure_dir="$figure_root/result_BENCHMARK"

if [[ -n "${RESULTS_DIR:-}" ]]; then
  root="$RESULTS_DIR"
else
  root="$study_dir/results/benchmark_evaluation/latest-all"
  if [[ -d "$root" ]]; then
    find "$root" -mindepth 1 -depth -delete
  fi
fi
readonly root
mkdir -p "$root"

for method in sequential_method existing_method existing_method_green_context proposed_method; do
  RESULTS_DIR="$root/$method" "$script_dir/run_single_method.sh" "$method"
done

{
  head -n 1 "$root/sequential_method/results.csv"
  for method in sequential_method existing_method existing_method_green_context proposed_method; do
    tail -n +2 "$root/$method/results.csv"
  done
} >"$root/all_results.csv"

mkdir -p "$stg_figure_dir" "$benchmark_figure_dir"

awk -F, 'NR == 1 || $1 ~ /^synthetic_/' \
  "$root/all_results.csv" >"$stg_figure_dir/all_stg_results.csv"
awk -F, 'NR == 1 || $1 !~ /^synthetic_/' \
  "$root/all_results.csv" >"$benchmark_figure_dir/all_benchmark_results.csv"

MPLCONFIGDIR="$root/.matplotlib" \
  python3 "$script_dir/plot_benchmark_comparison.py" \
    "$stg_figure_dir/all_stg_results.csv" \
    "$stg_figure_dir/all_stg_execution_time_comparison.png"

MPLCONFIGDIR="$root/.matplotlib" \
  python3 "$script_dir/plot_benchmark_comparison.py" \
    "$benchmark_figure_dir/all_benchmark_results.csv" \
    "$benchmark_figure_dir/all_benchmark_execution_time_comparison.png"

echo "combined result: $root/all_results.csv"
echo "published STG CSV: $stg_figure_dir/all_stg_results.csv"
echo "published STG graph: $stg_figure_dir/all_stg_execution_time_comparison.png"
echo "published other-benchmark CSV: $benchmark_figure_dir/all_benchmark_results.csv"
echo "published other-benchmark graph: $benchmark_figure_dir/all_benchmark_execution_time_comparison.png"
