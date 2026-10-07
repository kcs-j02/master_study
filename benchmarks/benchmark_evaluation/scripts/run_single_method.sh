#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )); then
  echo "usage: $0 METHOD [BENCHMARK ...]" >&2
  exit 2
fi

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly master_dir="/home/kobayashi/main/master_study"
readonly method="$1"
shift
# shellcheck source=benchmark_catalog.sh
source "$script_dir/benchmark_catalog.sh"

case "$method" in
  sequential_method)
    readonly executable="$master_dir/sequential_method/main"
    ;;
  existing_method)
    readonly executable="$master_dir/existing_method/main"
    ;;
  existing_method_green_context)
    readonly executable="$master_dir/existing_method_green_context/main"
    ;;
  proposed_method)
    readonly executable="$master_dir/proposed_method/main"
    ;;
  *)
    echo "unknown method: $method" >&2
    exit 2
    ;;
esac

if [[ ! -x "$executable" ]]; then
  echo "executable not found: $executable" >&2
  echo "The source directory was intentionally not modified or rebuilt." >&2
  exit 1
fi

readonly runs="${RUNS:-12}"
readonly warmup_runs="${WARMUP_RUNS:-2}"
if [[ ! "$runs" =~ ^[1-9][0-9]*$ || ! "$warmup_runs" =~ ^[0-9]+$ ]] ||
   (( warmup_runs >= runs )); then
  echo "RUNS must be positive and WARMUP_RUNS must satisfy 0 <= WARMUP_RUNS < RUNS" >&2
  exit 2
fi

if [[ -n "${RESULTS_DIR:-}" ]]; then
  result_dir="$RESULTS_DIR"
else
  result_dir="$master_dir/results/benchmark_evaluation/latest-$method"
  if [[ -d "$result_dir" ]]; then
    find "$result_dir" -mindepth 1 -depth -delete
  fi
fi
readonly result_dir
mkdir -p "$result_dir/logs" "$result_dir/work"
readonly csv="$result_dir/results.csv"
printf 'benchmark,method,status,runs,warmup_runs,mean_gpu_submit_wait_ms,min_gpu_submit_wait_ms,max_gpu_submit_wait_ms,note\n' >"$csv"

if (( $# > 0 )); then
  benchmarks=("$@")
else
  benchmarks=("${ALL_BENCHMARKS[@]}")
fi

for benchmark in "${benchmarks[@]}"; do
  if ! is_known_benchmark "$benchmark"; then
    echo "unknown benchmark: $benchmark" >&2
    exit 2
  fi
  benchmark_group_name="$(benchmark_group "$benchmark")"
  stg="$master_dir/benchmarks/$benchmark_group_name/BENCHMARK_$benchmark/$benchmark.stg"
  if [[ ! -f "$stg" ]]; then
    echo "benchmark input not found: $stg" >&2
    exit 1
  fi
  values_file="$result_dir/work/$benchmark.values"
  : >"$values_file"
  echo "[$method] $benchmark ($runs runs, $warmup_runs warmup)" >&2

  for ((run = 1; run <= runs; ++run)); do
    log="$result_dir/logs/${benchmark}_run$(printf '%02d' "$run").log"
    (
      cd "$result_dir/work"
      STG_DISABLE_STREAM_PLOT=1 "$executable" "$stg"
    ) >"$log" 2>&1

    value="$(awk '$1 == "gpu_submit_wait_ms:" {v=$2} END {if (v == "") exit 1; print v}' "$log")"
    echo "  run $run/$runs: $value ms" >&2
    if (( run > warmup_runs )); then
      printf '%s\n' "$value" >>"$values_file"
    fi
  done

  stats="$(awk '
    NR == 1 {min=$1; max=$1}
    {sum+=$1; if ($1<min) min=$1; if ($1>max) max=$1}
    END {if (NR == 0) exit 1; printf "%.6f,%.6f,%.6f", sum/NR, min, max}
  ' "$values_file")"
  printf '%s,%s,ok,%s,%s,%s,%s\n' \
    "$benchmark" "$method" "$runs" "$warmup_runs" "$stats" \
    'shared task-graph benchmark input' >>"$csv"
done

echo "result: $csv"
