#!/usr/bin/env bash
set -euo pipefail

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly study_dir="$(cd -- "$script_dir/../.." && pwd)"
readonly output_dir="$study_dir/results/method_comparison_figures/kesco_sm_scaling"
readonly executable="$script_dir/benchmark_kesco_sm_scaling"
readonly warmups="${WARMUP_RUNS:-5}"
readonly runs="${RUNS:-21}"

mkdir -p "$output_dir"
nvcc -O2 -std=c++20 "$script_dir/benchmark_kesco_sm_scaling.cu" \
  -I/home/kobayashi/taskflow -o "$executable"
"$executable" "$output_dir/kesco_sm_scaling.csv" "$warmups" "$runs"
python3 "$script_dir/plot_kesco_sm_scaling.py" \
  "$output_dir/kesco_sm_scaling.csv" "$output_dir"

echo "CSV: $output_dir/kesco_sm_scaling.csv"
echo "graph: $output_dir/kesco_benchmark_sm_scaling.png"
echo "slowdown graph: $output_dir/kesco_benchmark_sm_slowdown.png"
