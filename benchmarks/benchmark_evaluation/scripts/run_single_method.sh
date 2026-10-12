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

readonly runs="${RUNS:-33}"
readonly warmup_runs="${WARMUP_RUNS:-3}"
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
printf 'benchmark,method,status,runs,warmup_runs,mean_gpu_submit_wait_ms,min_gpu_submit_wait_ms,max_gpu_submit_wait_ms,median_gpu_submit_wait_ms,stddev_gpu_submit_wait_ms,ci95_gpu_submit_wait_ms,note\n' >"$csv"

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
    if ! (
      cd "$result_dir/work"
      STG_DISABLE_STREAM_PLOT=1 "$executable" "$stg"
    ) >"$log" 2>&1; then
      echo "benchmark process failed: $method / $benchmark / run $run" >&2
      echo "log: $log" >&2
      tail -n 40 "$log" >&2
      exit 1
    fi

    value="$(awk '$1 == "gpu_submit_wait_ms:" {v=$2} END {if (v == "") exit 1; print v}' "$log")"
    echo "  run $run/$runs: $value ms" >&2
    if (( run > warmup_runs )); then
      printf '%s\n' "$value" >>"$values_file"
    fi
  done

  sorted_values_file="$result_dir/work/$benchmark.sorted.values"
  sort -n "$values_file" >"$sorted_values_file"
  stats="$(awk '
    function t_critical_95(df, z, z2, z3, z5, z7) {
      if (df <= 1) return 12.706
      if (df == 2) return 4.303
      z=1.959963984540054
      z2=z*z; z3=z2*z; z5=z3*z2; z7=z5*z2
      return z + (z3+z)/(4*df) + (5*z5+16*z3+3*z)/(96*df*df) + (3*z7+19*z5+17*z3-15*z)/(384*df*df*df)
    }
    NR == 1 {min=$1}
    {value[NR]=$1; sum+=$1; sumsq+=$1*$1; max=$1}
    END {
      if (NR == 0) exit 1
      mean=sum/NR
      median=(NR%2 ? value[(NR+1)/2] : (value[NR/2]+value[NR/2+1])/2)
      stddev=(NR>1 ? sqrt((sumsq-NR*mean*mean)/(NR-1)) : 0)
      ci95=(NR>1 ? t_critical_95(NR-1)*stddev/sqrt(NR) : 0)
      printf "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f", mean,min,max,median,stddev,ci95
    }
  ' "$sorted_values_file")"
  printf '%s,%s,ok,%s,%s,%s,%s\n' \
    "$benchmark" "$method" "$runs" "$warmup_runs" "$stats" \
    'shared task-graph benchmark input' >>"$csv"
done

echo "result: $csv"
