#!/usr/bin/env bash
set -euo pipefail

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly root_dir="$(cd -- "$script_dir/../.." && pwd)"
readonly output_dir="${RESULTS_DIR:-$root_dir/results/scale_seed}"
readonly graph_dir="$output_dir/graphs"
readonly binary="$root_dir/experiments/ablation/strict_ablation"
readonly runs="${RUNS:-33}"
readonly warmup_runs="${WARMUP_RUNS:-3}"
readonly stream_count="${ABLATION_STREAMS:-3}"

if [[ ! -x "$binary" ]]; then
  "$root_dir/experiments/ablation/build_strict_ablation.sh" "$binary" >/dev/null
fi

mkdir -p "$output_dir"
python3 "$script_dir/generate_stg.py" \
  --output-dir "$graph_dir" \
  --sizes ${GRAPH_SIZES:-10 20 50 100} \
  --seeds ${GRAPH_SEEDS:-1 2 3}

printf 'variant,graph_kind,task_count,seed,status,runs,warmup_runs,mean_gpu_submit_wait_ms,min_gpu_submit_wait_ms,max_gpu_submit_wait_ms,median_gpu_submit_wait_ms,stddev_gpu_submit_wait_ms,ci95_gpu_submit_wait_ms\n' \
  >"$output_dir/scale_seed_results.csv"

for graph in "$graph_dir"/*.stg; do
  filename="${graph##*/}"
  stem="${filename%.stg}"
  kind="${stem%%_n*}"
  remainder="${stem#*_n}"
  task_count="${remainder%%_seed*}"
  seed="${remainder#*_seed}"

  for mode in full no_importance no_sm_allocation no_stream_placement; do
    values="$output_dir/.${stem}_${mode}.values"
    : >"$values"
    for ((run = 1; run <= runs; ++run)); do
      log="$output_dir/.${stem}_${mode}_run$(printf '%02d' "$run").log"
      STG_ABLATION_MODE="$mode" \
        STG_ABLATION_STREAMS="$stream_count" \
        STG_DISABLE_STREAM_PLOT=1 \
        "$binary" "$graph" >"$log" 2>&1
      value="$(awk '$1 == "gpu_submit_wait_ms:" {v=$2} END {print v}' "$log")"
      if [[ -z "$value" ]]; then
        echo "gpu_submit_wait_ms missing: $log" >&2
        exit 1
      fi
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
    printf '%s,%s,%s,%s,ok,%s,%s,%s\n' \
      "$mode" "$kind" "$task_count" "$seed" "$runs" "$warmup_runs" "$stats" \
      >>"$output_dir/scale_seed_results.csv"
    rm -f "$values"
  done
done

rm -f "$output_dir"/.*.log
echo "scale/seed results: $output_dir/scale_seed_results.csv"
