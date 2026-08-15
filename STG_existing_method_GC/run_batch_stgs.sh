#!/usr/bin/env bash
set -euo pipefail


# ============================================================
# Directory
# ============================================================

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MASTER_DIR="$(cd "$ROOT_DIR/.." && pwd)"

TASKFLOW_INC="/home/kobayashi/taskflow"

# Sequential
SEQUENTIAL_DIR="$MASTER_DIR/STG"
SEQUENTIAL_MAIN="$SEQUENTIAL_DIR/main"

# Existing Method + Green Context
EMGC_DIR="$ROOT_DIR"
EMGC_MAIN="$EMGC_DIR/main"

# Nsight Systems
NSYS="/usr/local/cuda/bin/nsys"
NSYS_TMP_ROOT="$HOME/tmp/nsys"

# Nsight結果保存場所
NSYS_RESULT_DIR="$ROOT_DIR/nsys_reports"

mkdir -p "$NSYS_TMP_ROOT"
mkdir -p "$NSYS_RESULT_DIR"


# ============================================================
# Evaluation settings
# ============================================================

RUNS=12
WARMUP_RUNS=2


# ============================================================
# STG files
# ============================================================

STGS=(
  "$MASTER_DIR/sample_mixed_chain_parallel.stg"
  "$MASTER_DIR/sample_fully_parallel.stg"
  "$MASTER_DIR/sample_Multiple_long_branches.stg"
  "$MASTER_DIR/sample_random.stg"
  "$MASTER_DIR/sample_trial.stg"
)


# ============================================================
# Final result arrays
# ============================================================

RESULT_NAMES=()
RESULT_SEQ=()
RESULT_EMGC=()
RESULT_SPEEDUP=()
RESULT_SM=()


# ============================================================
# Build Sequential
# ============================================================

echo "Building Sequential..."

cd "$SEQUENTIAL_DIR"

rm -f main

nvcc -O2 -std=c++20 main.cu \
  -I"${TASKFLOW_INC}" \
  -o main

echo "Sequential build completed."


# ============================================================
# Build EM-GC
# ============================================================

echo "Building EM-GC..."

cd "$EMGC_DIR"

rm -f main

nvcc -O2 -std=c++20 main.cu \
  -I"${TASKFLOW_INC}" \
  -o main

echo "EM-GC build completed."

echo


# ============================================================
# Execution time measurement
#
# 12 runs
# First 2 runs = warmup
# Average of remaining 10 runs
# ============================================================

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


    # --------------------------------------------------------
    # gpu_submit_wait_ms
    # --------------------------------------------------------

    time_ms=$(
      grep 'gpu_submit_wait_ms:' "$logfile" \
        | tail -n1 \
        | awk '{print $2}'
    )


    rm -f "$logfile"


    echo "[$method_name] run $run/$RUNS : ${time_ms} ms" >&2


    # --------------------------------------------------------
    # Skip warmup
    # --------------------------------------------------------

    if (( run > WARMUP_RUNS )); then

      sum=$(
        awk \
          -v a="$sum" \
          -v b="$time_ms" \
          'BEGIN {
              printf "%.10f", a + b
          }'
      )

      ((count+=1))

    fi

  done


  # ----------------------------------------------------------
  # Average
  # ----------------------------------------------------------

  average=$(
    awk \
      -v sum="$sum" \
      -v count="$count" \
      'BEGIN {
          printf "%.3f", sum / count
      }'
  )


  echo "$average"
}


# ============================================================
# SM Utilization measurement
#
# EM-GCをNsight Systemsで別に1回実行
#
# .nsys-rep と .sqlite は保存する
# ============================================================

measure_sm_utilization() {

  local stg="$1"

  local name
  local report_prefix
  local report_file
  local sqlite_file
  local sm_utilization


  name=$(basename "$stg" .stg)


  # ==========================================================
  # Saved filenames
  # ==========================================================

  report_prefix="$NSYS_RESULT_DIR/${name}_emgc"

  report_file="${report_prefix}.nsys-rep"
  sqlite_file="${report_prefix}.sqlite"


  # 古い結果を削除
  rm -f "$report_file"
  rm -f "$sqlite_file"


  echo "[Nsight] profiling $name ..." >&2


  # ==========================================================
  # Nsight Systems profile
  # ==========================================================

  TMPDIR="$NSYS_TMP_ROOT" "$NSYS" profile \
    --force-overwrite=true \
    --sample=none \
    --cpuctxsw=none \
    --trace=cuda \
    --gpu-metrics-devices=0 \
    --gpu-metrics-frequency=10000 \
    --output="$report_prefix" \
    "$EMGC_MAIN" "$stg" \
    > /dev/null 2>&1


  # ==========================================================
  # nsys-rep -> SQLite
  # ==========================================================

  TMPDIR="$NSYS_TMP_ROOT" "$NSYS" export \
    --type=sqlite \
    --force-overwrite=true \
    --output="$sqlite_file" \
    "$report_file" \
    > /dev/null 2>&1


  # ==========================================================
  # Calculate SM Utilization
  # ==========================================================

  sm_utilization=$(
    python3 - "$sqlite_file" <<'PY'

import sqlite3
import sys


sqlite_file = sys.argv[1]


# ============================================================
# Open SQLite
# ============================================================

con = sqlite3.connect(sqlite_file)
cur = con.cursor()


# ============================================================
# Get SMs Active
# ============================================================

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


# ============================================================
# No data
# ============================================================

if not rows:

    print("N/A")
    sys.exit(0)


values = [float(row[1]) for row in rows]


# ============================================================
# Detect active regions
#
# value > 0 が連続している区間を抽出
# ============================================================

regions = []

start = None


for i, value in enumerate(values):

    # Active start
    if value > 0.0 and start is None:

        start = i


    # Active end
    elif value <= 0.0 and start is not None:

        regions.append(
            (start, i - 1)
        )

        start = None


# 最後までactiveの場合
if start is not None:

    regions.append(
        (start, len(values) - 1)
    )


# ============================================================
# No active region
# ============================================================

if len(regions) == 0:

    print("0.00")
    sys.exit(0)


# ============================================================
# Initial sequential executionを除外
#
# regions[0]
#   = 最初の逐次実行
#
# regions[1]
#   = EM-GC本実行開始
# ============================================================

if len(regions) < 2:

    print("N/A")
    sys.exit(0)


# ============================================================
# EM-GC本実行区間
#
# 2番目のactive区間開始
#       ～
# 最後のactive区間終了
#
# 途中の0%は含める
# ============================================================

target_start = regions[1][0]
target_end   = regions[-1][1]


target_values = values[
    target_start:
    target_end + 1
]


if not target_values:

    print("N/A")
    sys.exit(0)


# ============================================================
# Average SMs Active
# ============================================================

avg = sum(target_values) / len(target_values)


print(f"{avg:.2f}")

PY
  )


  echo "[Nsight] saved:" >&2
  echo "  $report_file" >&2
  echo "  $sqlite_file" >&2


  echo "$sm_utilization"
}


# ============================================================
# Header
# ============================================================

echo
echo "================================================================================================================================"
echo " EM-GC Evaluation"
echo "================================================================================================================================"

printf "%-35s %22s %20s %15s %20s\n" \
  "STG" \
  "Sequential Time [ms]" \
  "EM-GC Time [ms]" \
  "Speedup [x]" \
  "SM Utilization [%]"

printf "%-35s %22s %20s %15s %20s\n" \
  "-----------------------------------" \
  "----------------------" \
  "--------------------" \
  "---------------" \
  "--------------------"


# ============================================================
# Evaluate each STG
# ============================================================

for stg in "${STGS[@]}"; do


  # ----------------------------------------------------------
  # Check STG
  # ----------------------------------------------------------

  if [ ! -f "$stg" ]; then

    echo "STG file not found: $stg" >&2
    continue

  fi


  name=$(basename "$stg" .stg)


  echo >&2
  echo "================================================================" >&2
  echo "STG: $name" >&2
  echo "================================================================" >&2


  # ==========================================================
  # Sequential
  #
  # 12 runs
  # 2 warmup
  # 10-run average
  # ==========================================================

  sequential_time=$(
    measure_average \
      "$SEQUENTIAL_MAIN" \
      "$stg" \
      "Sequential"
  )


  # ==========================================================
  # EM-GC
  #
  # 12 runs
  # 2 warmup
  # 10-run average
  # ==========================================================

  emgc_time=$(
    measure_average \
      "$EMGC_MAIN" \
      "$stg" \
      "EM-GC"
  )


  # ==========================================================
  # Speedup
  #
  # Sequential / EM-GC
  # ==========================================================

  speedup=$(
    awk \
      -v seq="$sequential_time" \
      -v emgc="$emgc_time" \
      'BEGIN {

          if (emgc > 0) {

              printf "%.3f", seq / emgc

          } else {

              printf "0.000"

          }
      }'
  )


  # ==========================================================
  # SM Utilization
  # ==========================================================

  sm_utilization=$(
    measure_sm_utilization "$stg"
  )


  # ==========================================================
  # Current result
  # ==========================================================

  if [ "$sm_utilization" = "N/A" ]; then

    printf "%-35s %22s %20s %14sx %20s\n" \
      "$name" \
      "$sequential_time" \
      "$emgc_time" \
      "$speedup" \
      "N/A"

  else

    printf "%-35s %22s %20s %14sx %19s%%\n" \
      "$name" \
      "$sequential_time" \
      "$emgc_time" \
      "$speedup" \
      "$sm_utilization"

  fi


  # ==========================================================
  # Save result for final summary
  # ==========================================================

  RESULT_NAMES+=("$name")
  RESULT_SEQ+=("$sequential_time")
  RESULT_EMGC+=("$emgc_time")
  RESULT_SPEEDUP+=("$speedup")
  RESULT_SM+=("$sm_utilization")

done


# ============================================================
# FINAL SUMMARY
# ============================================================

echo
echo
echo "################################################################################################################################"
echo "# FINAL SUMMARY"
echo "################################################################################################################################"
echo


printf "%-35s %22s %20s %15s %20s\n" \
  "STG" \
  "Sequential Time [ms]" \
  "EM-GC Time [ms]" \
  "Speedup [x]" \
  "SM Utilization [%]"


printf "%-35s %22s %20s %15s %20s\n" \
  "-----------------------------------" \
  "----------------------" \
  "--------------------" \
  "---------------" \
  "--------------------"


for ((i=0; i<${#RESULT_NAMES[@]}; i++)); do


  if [ "${RESULT_SM[$i]}" = "N/A" ]; then

    printf "%-35s %22s %20s %14sx %20s\n" \
      "${RESULT_NAMES[$i]}" \
      "${RESULT_SEQ[$i]}" \
      "${RESULT_EMGC[$i]}" \
      "${RESULT_SPEEDUP[$i]}" \
      "N/A"

  else

    printf "%-35s %22s %20s %14sx %19s%%\n" \
      "${RESULT_NAMES[$i]}" \
      "${RESULT_SEQ[$i]}" \
      "${RESULT_EMGC[$i]}" \
      "${RESULT_SPEEDUP[$i]}" \
      "${RESULT_SM[$i]}"

  fi

done


# ============================================================
# Information
# ============================================================

echo
echo "################################################################################################################################"
echo

echo "Execution Time:"
echo "  Runs    : $RUNS"
echo "  Warmup  : $WARMUP_RUNS"
echo "  Average : $((RUNS - WARMUP_RUNS)) runs"

echo

echo "Speedup:"
echo "  Sequential Time / EM-GC Time"

echo

echo "SM Utilization:"
echo "  Average SMs Active [%]"
echo "  Initial sequential active region excluded"
echo "  Zero samples during EM-GC execution included"

echo

echo "Nsight Systems Reports:"
echo "  $NSYS_RESULT_DIR"

echo
echo "Open example:"
echo "  nsys-ui $NSYS_RESULT_DIR/sample_fully_parallel_emgc.nsys-rep"

echo