#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MASTER_DIR="$(cd "$ROOT_DIR/.." && pwd)"

TASKFLOW_INC="/home/kobayashi/taskflow"

SEQUENTIAL_DIR="$MASTER_DIR/STG"
SEQUENTIAL_MAIN="$SEQUENTIAL_DIR/main"

PROPOSED_DIR="$ROOT_DIR"
PROPOSED_MAIN="$PROPOSED_DIR/main"

NSYS="/usr/local/cuda/bin/nsys"
NSYS_TMP_ROOT="$HOME/tmp/nsys"
NSYS_RESULT_DIR="$ROOT_DIR/nsys_reports"

mkdir -p "$NSYS_TMP_ROOT"
mkdir -p "$NSYS_RESULT_DIR"

RUNS=12
WARMUP_RUNS=2

STGS=(
  "$MASTER_DIR/sample_mixed_chain_parallel.stg"
  "$MASTER_DIR/sample_fully_parallel.stg"
  "$MASTER_DIR/sample_Multiple_long_branches.stg"
  "$MASTER_DIR/sample_random.stg"
  "$MASTER_DIR/sample_trial.stg"
)

RESULT_NAMES=()
RESULT_SEQ=()
RESULT_PROPOSED=()
RESULT_SPEEDUP=()
RESULT_SM=()

echo "Building Sequential..."

cd "$SEQUENTIAL_DIR"
rm -f main

nvcc -O2 -std=c++20 main.cu \
  -I"${TASKFLOW_INC}" \
  -o main

echo "Building Proposed..."

cd "$PROPOSED_DIR"
rm -f main

nvcc -O2 -std=c++20 main.cu \
  -I"${TASKFLOW_INC}" \
  -o main

echo "Build completed."
echo

measure_average() {
  local executable="$1"
  local stg="$2"
  local method_name="$3"

  local sum="0"
  local count=0
  local logfile
  local time_ms
  local average

  for ((run=1; run<=RUNS; run++)); do
    logfile=$(mktemp)

    "$executable" "$stg" > "$logfile" 2>&1

    time_ms=$(
      grep 'gpu_submit_wait_ms:' "$logfile" \
        | tail -n1 \
        | awk '{print $2}'
    )

    rm -f "$logfile"

    echo "[$method_name] run $run/$RUNS : ${time_ms} ms" >&2

    if (( run > WARMUP_RUNS )); then
      sum=$(
        awk \
          -v a="$sum" \
          -v b="$time_ms" \
          'BEGIN { printf "%.10f", a + b }'
      )
      ((count+=1))
    fi
  done

  average=$(
    awk \
      -v sum="$sum" \
      -v count="$count" \
      'BEGIN { printf "%.3f", sum / count }'
  )

  echo "$average"
}

measure_sm_utilization() {
  local stg="$1"

  local name
  local report_prefix
  local report_file
  local sqlite_file
  local sm_utilization

  name=$(basename "$stg" .stg)

  report_prefix="$NSYS_RESULT_DIR/${name}_proposed"
  report_file="${report_prefix}.nsys-rep"
  sqlite_file="${report_prefix}.sqlite"

  rm -f "$report_file"
  rm -f "$sqlite_file"

  echo "[Nsight] profiling $name ..." >&2

  TMPDIR="$NSYS_TMP_ROOT" \
  "$NSYS" profile \
    --force-overwrite=true \
    --sample=none \
    --cpuctxsw=none \
    --trace=cuda \
    --gpu-metrics-devices=0 \
    --gpu-metrics-frequency=10000 \
    --output="$report_prefix" \
    "$PROPOSED_MAIN" "$stg" \
    > /dev/null 2>&1

  TMPDIR="$NSYS_TMP_ROOT" \
  "$NSYS" export \
    --type=sqlite \
    --force-overwrite=true \
    --output="$sqlite_file" \
    "$report_file" \
    > /dev/null 2>&1

  sm_utilization=$(
    python3 - "$sqlite_file" <<'PY'
import sqlite3
import sys

sqlite_file = sys.argv[1]

con = sqlite3.connect(sqlite_file)
cur = con.cursor()

rows = cur.execute("""
SELECT
    g.timestamp,
    g.value
FROM GPU_METRICS AS g
JOIN TARGET_INFO_GPU_METRICS AS info
    USING (metricId)
WHERE info.metricName LIKE 'SMs Active%'
ORDER BY g.timestamp
""").fetchall()

con.close()

if not rows:
    print("N/A")
    sys.exit(0)

values = [float(row[1]) for row in rows]

active_indices = [
    i for i, value in enumerate(values)
    if value > 0.0
]

if not active_indices:
    print("0.00")
    sys.exit(0)

start = active_indices[0]
end = active_indices[-1]

target_values = values[start:end + 1]

avg = sum(target_values) / len(target_values)

print(f"{avg:.2f}")
PY
  )

  echo "[Nsight] saved:" >&2
  echo "  $report_file" >&2
  echo "  $sqlite_file" >&2

  echo "$sm_utilization"
}

echo
echo "================================================================================================================================"
echo " Proposed Method Evaluation"
echo "================================================================================================================================"

printf "%-35s %22s %22s %15s %20s\n" \
  "STG" \
  "Sequential Time [ms]" \
  "Proposed Time [ms]" \
  "Speedup [x]" \
  "SM Utilization [%]"

printf "%-35s %22s %22s %15s %20s\n" \
  "-----------------------------------" \
  "----------------------" \
  "----------------------" \
  "---------------" \
  "--------------------"

for stg in "${STGS[@]}"; do
  if [ ! -f "$stg" ]; then
    echo "STG file not found: $stg" >&2
    continue
  fi

  name=$(basename "$stg" .stg)

  echo >&2
  echo "================================================================" >&2
  echo "STG: $name" >&2
  echo "================================================================" >&2

  sequential_time=$(
    measure_average \
      "$SEQUENTIAL_MAIN" \
      "$stg" \
      "Sequential"
  )

  proposed_time=$(
    measure_average \
      "$PROPOSED_MAIN" \
      "$stg" \
      "Proposed"
  )

  speedup=$(
    awk \
      -v seq="$sequential_time" \
      -v proposed="$proposed_time" \
      'BEGIN {
          if (proposed > 0) {
              printf "%.3f", seq / proposed
          } else {
              printf "0.000"
          }
      }'
  )

  sm_utilization=$(
    measure_sm_utilization "$stg"
  )

  if [ "$sm_utilization" = "N/A" ]; then
    printf "%-35s %22s %22s %14sx %20s\n" \
      "$name" \
      "$sequential_time" \
      "$proposed_time" \
      "$speedup" \
      "N/A"
  else
    printf "%-35s %22s %22s %14sx %19s%%\n" \
      "$name" \
      "$sequential_time" \
      "$proposed_time" \
      "$speedup" \
      "$sm_utilization"
  fi

  RESULT_NAMES+=("$name")
  RESULT_SEQ+=("$sequential_time")
  RESULT_PROPOSED+=("$proposed_time")
  RESULT_SPEEDUP+=("$speedup")
  RESULT_SM+=("$sm_utilization")
done

echo
echo
echo "################################################################################################################################"
echo "# FINAL SUMMARY"
echo "################################################################################################################################"
echo

printf "%-35s %22s %22s %15s %20s\n" \
  "STG" \
  "Sequential Time [ms]" \
  "Proposed Time [ms]" \
  "Speedup [x]" \
  "SM Utilization [%]"

printf "%-35s %22s %22s %15s %20s\n" \
  "-----------------------------------" \
  "----------------------" \
  "----------------------" \
  "---------------" \
  "--------------------"

for ((i=0; i<${#RESULT_NAMES[@]}; i++)); do
  if [ "${RESULT_SM[$i]}" = "N/A" ]; then
    printf "%-35s %22s %22s %14sx %20s\n" \
      "${RESULT_NAMES[$i]}" \
      "${RESULT_SEQ[$i]}" \
      "${RESULT_PROPOSED[$i]}" \
      "${RESULT_SPEEDUP[$i]}" \
      "N/A"
  else
    printf "%-35s %22s %22s %14sx %19s%%\n" \
      "${RESULT_NAMES[$i]}" \
      "${RESULT_SEQ[$i]}" \
      "${RESULT_PROPOSED[$i]}" \
      "${RESULT_SPEEDUP[$i]}" \
      "${RESULT_SM[$i]}"
  fi
done

echo
echo "################################################################################################################################"
echo

echo "Execution Time:"
echo "  Runs    : $RUNS"
echo "  Warmup  : $WARMUP_RUNS"
echo "  Average : $((RUNS - WARMUP_RUNS)) runs"

echo

echo "Speedup:"
echo "  Sequential Time / Proposed Time"

echo

echo "SM Utilization:"
echo "  Average SMs Active [%]"
echo "  Measurement range: first active sample to last active sample"
echo "  Zero samples inside Proposed execution are included"

echo

echo "Nsight Systems Reports:"
echo "  $NSYS_RESULT_DIR"

echo

echo "Open example:"
echo "  nsys-ui $NSYS_RESULT_DIR/sample_fully_parallel_proposed.nsys-rep"

echo