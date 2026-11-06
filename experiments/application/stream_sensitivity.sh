#!/usr/bin/env bash
set -euo pipefail

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly root_dir="$(cd -- "$script_dir/../.." && pwd)"
readonly output_dir="${RESULTS_DIR:-$root_dir/results/application_stream_sensitivity}"
readonly binary="$root_dir/experiments/ablation/strict_ablation"
readonly runs="${RUNS:-12}"
readonly warmup_runs="${WARMUP_RUNS:-2}"

if [[ ! "$runs" =~ ^[1-9][0-9]*$ ||
      ! "$warmup_runs" =~ ^[0-9]+$ ||
      "$warmup_runs" -ge "$runs" ]]; then
  echo "RUNS must be positive and 0 <= WARMUP_RUNS < RUNS" >&2
  exit 2
fi

mkdir -p "$output_dir/raw"
if [[ ! -x "$binary" ]]; then
  "$root_dir/experiments/ablation/build_strict_ablation.sh" "$binary" >/dev/null
fi

printf 'benchmark,stream_count,status,runs,warmup_runs,mean_gpu_kernel_ms,ci95_gpu_kernel_ms\n' \
  >"$output_dir/stream_sensitivity.csv"

for benchmark in image_processing deep_learning; do
  stg="$root_dir/benchmarks/KESCO/BENCHMARK_$benchmark/$benchmark.stg"
  for stream_count in 1 2 3 4 5; do
    values="$output_dir/.${benchmark}_${stream_count}.values"
    : >"$values"
    for ((run = 1; run <= runs; ++run)); do
      log="$output_dir/raw/${benchmark}_s${stream_count}_run$(printf '%02d' "$run").log"
      STG_ABLATION_MODE=full \
        STG_ABLATION_STREAMS="$stream_count" \
        STG_DISABLE_STREAM_PLOT=1 \
        "$binary" "$stg" >"$log" 2>&1
      kernel_ms="$(awk '$1 == "gpu_kernel_ms:" {v=$2} END {print v}' "$log")"
      [[ -n "$kernel_ms" ]] || {
        echo "measurement missing: $log" >&2
        exit 1
      }
      if (( run > warmup_runs )); then
        printf '%s\n' "$kernel_ms" >>"$values"
      fi
    done
    stats="$(awk '
      {kernel[NR]=$1; kernel_sum+=$1; kernel_sq+=$1*$1}
      END {
        kernel_mean=kernel_sum/NR
        stddev=sqrt((kernel_sq-NR*kernel_mean*kernel_mean)/(NR-1))
        ci95=2.262157163*stddev/sqrt(NR)
        printf "%.6f,%.6f", kernel_mean,ci95
      }' "$values")"
    printf '%s,%s,ok,%s,%s,%s\n' \
      "$benchmark" "$stream_count" "$runs" "$warmup_runs" "$stats" \
      >>"$output_dir/stream_sensitivity.csv"
  done
done

rm -f "$output_dir"/.*.values
echo "stream sensitivity results: $output_dir/stream_sensitivity.csv"
