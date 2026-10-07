#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 BENCHMARK" >&2
  exit 2
fi

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly project_dir="$(cd -- "$script_dir/.." && pwd)"
readonly study_dir="$(cd -- "$project_dir/../.." && pwd)"
readonly benchmark="$1"
# shellcheck source=benchmark_catalog.sh
source "$script_dir/benchmark_catalog.sh"

if ! is_known_benchmark "$benchmark"; then
  echo "unknown benchmark: $benchmark" >&2
  echo "available: ${ALL_BENCHMARKS[*]}" >&2
  exit 2
fi

readonly figure_root="$study_dir/results/method_comparison_figures"
if [[ "$(benchmark_group "$benchmark")" == "STG" ]]; then
  readonly figure_dir="$figure_root/RESULT_stg"
else
  readonly figure_dir="$figure_root/result_BENCHMARK"
fi

if [[ -n "${RESULTS_DIR:-}" ]]; then
  root="$RESULTS_DIR"
else
  root="$study_dir/results/benchmark_evaluation/latest-$benchmark"
  if [[ -d "$root" ]]; then
    find "$root" -mindepth 1 -depth -delete
  fi
fi
readonly root
mkdir -p "$root"

methods=(
  sequential_method
  existing_method
  existing_method_green_context
  proposed_method
)

for method in "${methods[@]}"; do
  RESULTS_DIR="$root/$method" \
    "$script_dir/run_single_method.sh" "$method" "$benchmark"
done

{
  head -n 1 "$root/sequential_method/results.csv"
  for method in "${methods[@]}"; do
    tail -n +2 "$root/$method/results.csv"
  done
} >"$root/all_results.csv"

MPLCONFIGDIR="$root/.matplotlib" \
  python3 "$script_dir/plot_benchmark_comparison.py" \
    "$root/all_results.csv" \
    "$root/${benchmark}_method_comparison.png"

mkdir -p "$figure_dir"
cp -- "$root/all_results.csv" \
  "$figure_dir/${benchmark}_results.csv"
cp -- "$root/${benchmark}_method_comparison.png" \
  "$figure_dir/${benchmark}_execution_time_comparison.png"

echo "combined result: $root/all_results.csv"
echo "comparison graph: $root/${benchmark}_method_comparison.png"
echo "published CSV: $figure_dir/${benchmark}_results.csv"
echo "published graph: $figure_dir/${benchmark}_execution_time_comparison.png"
