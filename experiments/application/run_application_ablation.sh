#!/usr/bin/env bash
set -euo pipefail

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly root_dir="$(cd -- "$script_dir/../.." && pwd)"
readonly output_dir="${RESULTS_DIR:-$root_dir/results/application_ablation}"
readonly binary="$output_dir/strict_ablation"
readonly runs="${RUNS:-33}"
readonly warmup_runs="${WARMUP_RUNS:-3}"
readonly stream_count="${ABLATION_STREAMS:-3}"

if [[ ! "$runs" =~ ^[1-9][0-9]*$ ||
      ! "$warmup_runs" =~ ^[0-9]+$ ||
      "$warmup_runs" -ge "$runs" ]]; then
  echo "RUNS must be positive and 0 <= WARMUP_RUNS < RUNS" >&2
  exit 2
fi

if [[ ! "$stream_count" =~ ^[1-9]$ ]] || (( stream_count > 5 )); then
  echo "ABLATION_STREAMS must be between 1 and 5" >&2
  exit 2
fi

mkdir -p "$output_dir/raw"
"$root_dir/experiments/ablation/build_strict_ablation.sh" "$binary" >/dev/null

declare -a benchmarks=(
  vector_square
  black_scholes
  machine_learning
  hits
  image_processing
  deep_learning
)
declare -a modes=(full no_importance no_sm_allocation no_stream_placement)

printf 'variant,benchmark,status,runs,warmup_runs,validation_status,validation_runs,validation_checks_per_run,max_validation_abs_error,mean_gpu_kernel_ms,min_gpu_kernel_ms,max_gpu_kernel_ms,median_gpu_kernel_ms,stddev_gpu_kernel_ms,ci95_gpu_kernel_ms,note\n' \
  >"$output_dir/application_ablation_results.csv"

for benchmark in "${benchmarks[@]}"; do
  stg="$root_dir/benchmarks/KESCO/BENCHMARK_$benchmark/$benchmark.stg"
  if [[ ! -f "$stg" ]]; then
    echo "benchmark input not found: $stg" >&2
    exit 1
  fi

  for mode in "${modes[@]}"; do
    mode_dir="$output_dir/raw/$mode"
    mkdir -p "$mode_dir/logs" "$mode_dir/work"
    values="$mode_dir/work/$benchmark.values"
    validations="$mode_dir/work/$benchmark.validation"
    : >"$values"
    : >"$validations"

    for ((run = 1; run <= runs; ++run)); do
      log="$mode_dir/logs/${benchmark}_run$(printf '%02d' "$run").log"
      if ! STG_ABLATION_MODE="$mode" \
        STG_ABLATION_STREAMS="$stream_count" \
        STG_DISABLE_STREAM_PLOT=1 \
        "$binary" "$stg" >"$log" 2>&1; then
        echo "benchmark failed: $mode / $benchmark / run $run" >&2
        tail -n 50 "$log" >&2
        exit 1
      fi

      value="$(awk '$1 == "gpu_kernel_ms:" {v=$2} END {print v}' "$log")"
      validation_status="$(awk '$1 == "validation_status:" {v=$2} END {print v}' "$log")"
      validation_checks="$(awk '$1 == "validation_checks:" {v=$2} END {print v}' "$log")"
      validation_error="$(awk '$1 == "validation_max_abs_error:" {v=$2} END {print v}' "$log")"
      if [[ -z "$value" || -z "$validation_status" ||
            -z "$validation_checks" || -z "$validation_error" ]]; then
        echo "required measurement or validation field missing: $log" >&2
        exit 1
      fi
      printf '%s,%s,%s\n' "$validation_status" "$validation_checks" \
        "$validation_error" >>"$validations"
      if (( run > warmup_runs )); then
        printf '%s\n' "$value" >>"$values"
      fi
    done

    stats="$(sort -n "$values" | awk '
      {value[NR]=$1; sum+=$1; sumsq+=$1*$1; max=$1}
      NR == 1 {min=$1}
      END {
        mean=sum/NR
        median=(NR%2 ? value[(NR+1)/2] : (value[NR/2]+value[NR/2+1])/2)
        stddev=(NR>1 ? sqrt((sumsq-NR*mean*mean)/(NR-1)) : 0)
        ci95=(NR>1 ? 2.045229642*stddev/sqrt(NR) : 0)
        printf "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f", mean,min,max,median,stddev,ci95
      }')"
    validation_summary="$(awk -F, '
      BEGIN {status="PASS"; checks=""; max_error=0}
      {
        if ($1 != "PASS") status="FAIL"
        if (checks == "") checks=$2
        if ($2 != checks) status="FAIL"
        if (($3 + 0) > max_error) max_error=$3 + 0
      }
      END {printf "%s,%d,%s,%.12g", status, NR, checks, max_error}
    ' "$validations")"
    row_status=ok
    [[ "$validation_summary" == FAIL,* ]] && row_status=validation_failed
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$mode" "$benchmark" "$row_status" "$runs" "$warmup_runs" \
      "$validation_summary" "$stats" \
      "KESCO application; fixed stream count $stream_count" \
      >>"$output_dir/application_ablation_results.csv"
  done
done

python3 "$root_dir/experiments/ablation/analyze_ablation.py" \
  "$output_dir/application_ablation_results.csv" \
  "$output_dir/application_ablation_summary.csv" \
  "$output_dir/application_ablation_summary.md"

echo "application ablation results: $output_dir/application_ablation_results.csv"
